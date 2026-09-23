"""Repair stale ``env_name`` metadata in the downloaded demonstration dataset.

The released demos were recorded before the environment classes were renamed, so
most files still carry names such as ``LiftPotStatic`` or ``PackBoxRotationEnv``
that no longer exist. ``DemoStore`` sidesteps this by building the env from the
requested class, but anything that trusts the file (``tools/render_video.py``,
``Demo.from_safetensors`` + ``metadata.get_env``) breaks. The folder a demo lives
in is the ground truth, so this script rewrites each file's ``env_name`` to match
its folder (tensors are untouched) and renames the two folders whose names match
no class.

Usage:
    python tools/fix_demo_env_names.py --dry-run
    python tools/fix_demo_env_names.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file

import roboeval.envs as envs_module
from roboeval.const import CACHE_PATH, DEMO_VERSION
from roboeval.demonstrations.const import SAFETENSORS_SUFFIX
from roboeval.utils.shared import find_class_in_module

DEFAULT_ROOT = CACHE_PATH / "roboeval_demos" / DEMO_VERSION / "BimanualPanda"

# Folders whose name does not match the current env class name.
FOLDER_RENAMES = {
    "CubeHandoverVertical": "VerticalCubeHandover",
    "LiftTrayDrag": "DragOverAndLiftTray",
}


def target_env_name(folder: Path) -> str:
    return FOLDER_RENAMES.get(folder.name, folder.name)


def fix_file(path: Path, env_name: str, dry_run: bool) -> bool:
    """Set ``environment_data.env_name`` in one file. Returns True if it needed changing."""
    with safe_open(path, framework="np") as f:
        metadata = f.metadata() or {}
        env_data = json.loads(metadata["environment_data"])
        if env_data["env_name"] == env_name:
            return False
        if dry_run:
            return True
        tensors = {key: f.get_tensor(key) for key in f.keys()}

    env_data["env_name"] = env_name
    metadata["environment_data"] = json.dumps(env_data)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, tmp_path, metadata=metadata)
    os.replace(tmp_path, path)  # atomic: never leaves a half-written demo behind
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Robot-level dataset folder")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Dataset folder not found: {args.root}")

    folders = sorted(p for p in args.root.iterdir() if p.is_dir())

    # Validate every target class up front so a typo can't corrupt anything.
    for folder in folders:
        name = target_env_name(folder)
        if find_class_in_module(envs_module, name) is None:
            raise SystemExit(f"No environment class named {name!r} for folder {folder}")

    total_changed = total_skipped = 0
    for folder in folders:
        env_name = target_env_name(folder)
        changed = skipped = 0
        for path in sorted(folder.rglob(f"*{SAFETENSORS_SUFFIX}")):
            if fix_file(path, env_name, args.dry_run):
                changed += 1
            else:
                skipped += 1
        total_changed += changed
        total_skipped += skipped
        if changed:
            print(f"{folder.name:46s} -> {env_name:32s} {changed:4d} to change, {skipped:4d} ok")

    for old, new in FOLDER_RENAMES.items():
        src, dst = args.root / old, args.root / new
        if not src.exists():
            continue
        if dst.exists():
            print(f"Skip rename {old} -> {new}: target already exists")
            continue
        print(f"Rename folder {old} -> {new}")
        if not args.dry_run:
            src.rename(dst)

    verb = "would change" if args.dry_run else "changed"
    print(f"\n{verb} {total_changed} files, {total_skipped} already correct")


if __name__ == "__main__":
    main()
