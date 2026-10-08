"""Evaluate a LeRobot-trained MolmoAct2 checkpoint on RoboEval tasks.

Runs in the `roboeval` conda env (simulation, success checks, metrics, videos). LeRobot inference runs
in a separate process in the `molmoact2` env (`molmoact2_policy_server.py`), which this script starts
and talks to over a local socket, because the two envs' numpy / mujoco / gymnasium pins conflict.

The policy was trained on absolute end-effector poses (see Molmoact2_RoboEval
scripts/roboeval_dataset_generation/compute_ee_poses.py), so observations and actions follow that
dataset exactly: 14-D [left xyz, left euler-xyz, right xyz, right euler-xyz, left grip, right grip],
3 cameras at 256x256, 20 Hz control.

Usage (from the RoboEval repo root, `roboeval` env):
    MUJOCO_GL=egl python evaluation/eval_molmoact2.py --config evaluation/configs/molmoact2_roboeval.yaml
"""

import argparse
import json
import os
import subprocess
import time
import warnings
from collections import defaultdict
from multiprocessing.connection import Client
from pathlib import Path

import imageio
import numpy as np
import yaml

import roboeval.envs
from roboeval.action_modes import JointPositionActionMode
from roboeval.robots.configs.panda import BimanualPanda
from roboeval.utils.observation_config import CameraConfig, ObservationConfig
from roboeval.utils.shared import find_class_in_module

AUTHKEY = b"molmoact2-roboeval"
SERVER_SCRIPT = Path(__file__).resolve().parent / "molmoact2_policy_server.py"
CAMERAS = ("head", "left_wrist", "right_wrist")
RESOLUTION = 256
CONTROL_FREQUENCY = 20  # the training data was recorded at 20 fps
# robot.qpos = [left arm 7, left fingers 2, right arm 7, right fingers 2]
ARM_QPOS = np.r_[0:7, 9:16]
EULER_DIMS = [3, 4, 5, 9, 10, 11]
# Must match WRAP_LOW in Molmoact2_RoboEval/scripts/roboeval_dataset_generation/compute_ee_poses.py:
# index into the 14-D EE vector -> lower bound of its training-data euler range [low, low + 2pi).
WRAP_LOW = {3: 0.0, 9: 0.0, 5: -5.1, 11: -5.1}


def encode(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {"data": arr.tobytes(), "dtype": arr.dtype.str, "shape": arr.shape}


def decode(packed: dict) -> np.ndarray:
    return np.frombuffer(packed["data"], dtype=packed["dtype"]).reshape(packed["shape"])


def start_policy_server(cfg: dict) -> tuple[subprocess.Popen, object]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu"]))
    log = open(Path(cfg["output_dir"]) / "policy_server.log", "a")
    server = subprocess.Popen(
        [cfg["policy_python"], str(SERVER_SCRIPT), cfg["checkpoint"], cfg["dataset_root"], str(cfg["port"])],
        env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    print("Starting policy server (model loading takes a few minutes) ...", flush=True)
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"Policy server exited; see {log.name}")
        try:
            return server, Client(("localhost", cfg["port"]), authkey=AUTHKEY)
        except ConnectionRefusedError:
            time.sleep(5)


def make_env(variant: str):
    env_cls = find_class_in_module(roboeval.envs, variant)
    if env_cls is None:
        raise ValueError(f"Unknown RoboEval task variant {variant!r}")
    return env_cls(
        action_mode=JointPositionActionMode(absolute=True, ee=True, floating_base=True, floating_dofs=[]),
        observation_config=ObservationConfig(
            cameras=[CameraConfig(name=c, rgb=True, depth=False, resolution=(RESOLUTION, RESOLUTION)) for c in CAMERAS]
        ),
        render_mode=None,
        robot_cls=BimanualPanda,
        control_frequency=CONTROL_FREQUENCY,
    )


def ee_state(env) -> np.ndarray:
    """Current 14-D absolute EE state, in the same convention as the training data."""
    robot = env.robot
    state = np.concatenate([robot.forward_kinematics(robot.qpos[ARM_QPOS]), robot.qpos_grippers]).astype(np.float32)
    for d, low in WRAP_LOW.items():
        state[d] = np.mod(state[d] - low, 2 * np.pi) + low
    return state


def to_json(value):
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_episode(env, conn, task: str, seed: int, max_steps: int, record: bool) -> tuple[dict, list]:
    obs, info = env.reset(seed=seed)
    conn.send({"cmd": "reset"})
    conn.recv()
    frames, success, steps = [], False, 0
    for steps in range(1, max_steps + 1):
        images = {c: obs[f"rgb_{c}"] for c in CAMERAS}  # uint8 CHW
        if record:
            frames.append(np.concatenate([np.moveaxis(images[c], 0, -1) for c in CAMERAS], axis=1))
        conn.send({"cmd": "act", "images": {c: encode(i) for c, i in images.items()},
                   "state": encode(ee_state(env)), "task": task})
        action = decode(conn.recv()["action"]).astype(np.float64)
        if not np.isfinite(action).all():
            raise RuntimeError(f"Non-finite action from policy: {action}")
        action[EULER_DIMS] = np.mod(action[EULER_DIMS] + np.pi, 2 * np.pi) - np.pi  # RoboEval's [-pi, pi)
        action = np.clip(action, env.action_space.low, env.action_space.high)  # e.g. gripper slightly < 0
        obs, reward, terminated, truncated, info = env.step(action)
        if env.success:
            success = True
            break
        if terminated or truncated:
            break
    return {"success": success, "steps": steps, "metrics": to_json(info)}, frames


def summarize(episodes: list[dict], out_dir: Path) -> None:
    by_variant = defaultdict(list)
    for e in episodes:
        by_variant[e["variant"]].append(e["success"])
    by_family = defaultdict(list)
    for v, s in by_variant.items():
        by_family[v.split("Position")[0].split("Orientation")[0].split("Obstacle")[0]].extend(s)
    rate = lambda s: sum(s) / len(s)
    summary = {
        "overall": {"success_rate": rate([e["success"] for e in episodes]), "episodes": len(episodes)},
        "per_family": {f: {"success_rate": rate(s), "episodes": len(s)} for f, s in sorted(by_family.items())},
        "per_variant": {v: {"success_rate": rate(s), "episodes": len(s)} for v, s in sorted(by_variant.items())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["| Task variant | Success | Episodes |", "|---|---|---|"]
    lines += [f"| {v} | {r['success_rate']:.0%} | {r['episodes']} |" for v, r in summary["per_variant"].items()]
    lines += ["", "| Task family | Success | Episodes |", "|---|---|---|"]
    lines += [f"| {f} | {r['success_rate']:.0%} | {r['episodes']} |" for f, r in summary["per_family"].items()]
    lines += ["", f"**Overall: {summary['overall']['success_rate']:.1%} over {len(episodes)} episodes**"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to the evaluation config YAML")
    cfg = yaml.safe_load(parser.parse_args().config.read_text())
    out_dir = Path(cfg["output_dir"])
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    # Training instruction and longest demo per variant, from the training dataset's sidecar.
    sources = [json.loads(l) for l in (Path(cfg["dataset_root"]) / "meta/roboeval_source.jsonl").read_text().splitlines()]
    task_of = {s["variant"]: s["task"] for s in sources}
    longest = defaultdict(int)
    for s in sources:
        longest[s["variant"]] = max(longest[s["variant"]], s["length"])
    variants = cfg.get("tasks") or sorted(task_of)

    episodes_path = out_dir / "episodes.jsonl"
    done = [json.loads(l) for l in episodes_path.read_text().splitlines()] if episodes_path.exists() else []
    finished = {(e["variant"], e["episode"]) for e in done}

    server, conn = start_policy_server(cfg)
    try:
        for variant in variants:
            todo = [i for i in range(cfg["episodes_per_task"]) if (variant, i) not in finished]
            if not todo:
                continue
            env = make_env(variant)
            max_steps = int(np.ceil(cfg["max_steps_factor"] * longest[variant]))
            for i in todo:
                seed = cfg["seed"] + i
                record = i < cfg["videos_per_task"]
                t0 = time.time()
                result, frames = run_episode(env, conn, task_of[variant], seed, max_steps, record)
                result = {"variant": variant, "episode": i, "seed": seed, "max_steps": max_steps,
                          "seconds": round(time.time() - t0, 1), **result}
                with episodes_path.open("a") as f:
                    f.write(json.dumps(result) + "\n")
                done.append(result)
                if record:
                    imageio.mimsave(out_dir / "videos" / f"{variant}_ep{i:02d}.mp4", frames, fps=CONTROL_FREQUENCY)
                print(f"[{variant} ep {i}] success={result['success']} steps={result['steps']}/{max_steps} "
                      f"({result['seconds']} s)", flush=True)
            env.close()
            summarize(done, out_dir)
    finally:
        conn.send({"cmd": "close"})
        server.wait(timeout=60)
    summarize(done, out_dir)
    print((out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
