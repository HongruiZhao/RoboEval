"""LeRobot MolmoAct2 policy server for RoboEval evaluation.

Runs in the `molmoact2` conda env (LeRobot, numpy 2) and is started by `eval_molmoact2.py`, which runs
the simulation in the `roboeval` env (numpy 1.26). The two talk over a local
`multiprocessing.connection` socket; arrays are sent as raw bytes + dtype + shape because numpy 2
pickles do not load under numpy 1.26.

Requests (dicts):
    {"cmd": "reset"}                                      -> {"ok": True}
    {"cmd": "act", "images": {cam: arr}, "state": arr, "task": str}
                                                           -> {"action": arr}
    {"cmd": "close"}                                      -> server exits

Usage:
    python molmoact2_policy_server.py CHECKPOINT_DIR DATASET_ROOT PORT
"""

import sys
from multiprocessing.connection import Listener

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors

AUTHKEY = b"molmoact2-roboeval"


def decode(packed: dict) -> np.ndarray:
    return np.frombuffer(packed["data"], dtype=packed["dtype"]).reshape(packed["shape"]).copy()


def encode(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    return {"data": arr.tobytes(), "dtype": arr.dtype.str, "shape": arr.shape}


def load_policy(checkpoint: str, dataset_root: str):
    """Load a LeRobot-saved MolmoAct2 checkpoint with its saved pre/post-processors."""
    pretrained = f"{checkpoint}/pretrained_model"
    cfg = PreTrainedConfig.from_pretrained(pretrained)
    cfg.pretrained_path = pretrained
    cfg.inference_action_mode = "continuous"
    meta = LeRobotDatasetMetadata("roboeval_lerobot_ee", root=dataset_root)
    policy = make_policy(cfg=cfg, ds_meta=meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=pretrained,
        preprocessor_overrides={"device_processor": {"device": str(cfg.device)}},
    )
    return policy, preprocessor, postprocessor


def main() -> None:
    checkpoint, dataset_root, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
    policy, preprocessor, postprocessor = load_policy(checkpoint, dataset_root)

    with Listener(("localhost", port), authkey=AUTHKEY) as listener:
        print(f"Policy server ready on port {port}", flush=True)
        with listener.accept() as conn:
            while True:
                request = conn.recv()
                if request["cmd"] == "close":
                    break
                if request["cmd"] == "reset":
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
                    conn.send({"ok": True})
                    continue
                batch = {
                    f"observation.images.{cam}": torch.from_numpy(decode(img)).float().div(255).unsqueeze(0)
                    for cam, img in request["images"].items()
                }
                batch["observation.state"] = torch.from_numpy(decode(request["state"])).float().unsqueeze(0)
                batch["task"] = [request["task"]]
                with torch.inference_mode():
                    action = postprocessor(policy.select_action(preprocessor(batch)))
                conn.send({"action": encode(action.float().cpu().numpy()[0])})


if __name__ == "__main__":
    main()
