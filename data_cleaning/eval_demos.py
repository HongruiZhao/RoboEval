"""Replay every original RoboEval demo and record whether it ends in success.

Writes one JSON line per demo to --out (resumable: already-evaluated files are skipped).
"""
import argparse
import json
import os
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/home/hongrui/nas/dataset/roboeval/roboeval_demos/1.0.0/BimanualPanda")
CHECK_EVERY = 10  # steps between "succeeded at any point" checks

_envs = {}


def _patch_drag_over_and_lift_tray():
    """Scratch-only fix: the preset spawns LighterBreakfastTray but the env looks up BreakfastTray."""
    import roboeval.envs.lift_tray as lt
    from roboeval.envs.props.items import LighterBreakfastTray

    original = lt.DragOverAndLiftTray._initialize_env
    if getattr(original, "_patched", False):
        return

    def _initialize_env(self):
        get_props = self._preset.get_props
        self._preset.get_props = lambda cls: get_props(LighterBreakfastTray if cls is lt.BreakfastTray else cls)
        try:
            original(self)
        finally:
            self._preset.get_props = get_props

    _initialize_env._patched = True
    lt.DragOverAndLiftTray._initialize_env = _initialize_env


def _env_for(demo):
    key = demo.metadata.environment_data.env_name
    if key not in _envs:
        for old in _envs.values():  # keep one env per worker to bound memory
            old.close()
        _envs.clear()
        _envs[key] = demo.metadata.get_env(500, render_mode=None)
    return _envs[key]


def evaluate(path):
    warnings.filterwarnings("ignore")
    from roboeval.demonstrations.demo import Demo
    _patch_drag_over_and_lift_tray()

    t0 = time.time()
    rec = {"file": str(path), "variant": path.parts[len(ROOT.parts)]}
    try:
        demo = Demo.from_safetensors(path)
        env = _env_for(demo)
        env.reset(seed=demo.seed)
        ever = False
        steps = demo.timesteps
        for i, step in enumerate(steps):
            env.step(step.executed_action, fast=True)
            if not ever and (i % CHECK_EVERY == 0) and env.success:
                ever = True
        final = bool(env.success)
        rec.update(steps=len(steps), final_success=final, ever_success=ever or final,
                   final_fail=bool(env.fail), healthy=bool(env.is_healthy))
    except Exception as e:  # keep going; report per-file errors
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    files = sorted(ROOT.glob("*/*/lightweight/*.safetensors"))
    done = set()
    if args.out.exists():
        done = {json.loads(l)["file"] for l in args.out.read_text().splitlines() if l.strip()}
    todo = [f for f in files if str(f) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(files)} demos total, {len(done)} done, {len(todo)} to evaluate", flush=True)

    with Pool(args.workers, maxtasksperchild=200) as pool, args.out.open("a") as out:
        for n, rec in enumerate(pool.imap_unordered(evaluate, todo, chunksize=1), 1):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if n % 100 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
