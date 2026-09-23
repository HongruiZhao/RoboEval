# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RoboEval is a MuJoCo-based benchmark for bimanual manipulation (Bimanual Franka Panda): 8 task families / 28 variants, human demo datasets, VR/keyboard teleop for data collection, and a fine-grained metrics system. Built on top of BiGym-style components via the `mojo` MJCF wrapper.

## Setup and commands

```bash
git submodule update --init --recursive     # thirdparty/mujoco_menagerie is a submodule (robot XMLs)
conda create -n roboeval python=3.10 && conda activate roboeval
pip install -e ".[examples]"                # add [vr] for Oculus, [dev] for pytest/pre-commit
python examples/1_data_replay.py            # smoke test: downloads demos to $ROBOEVAL_CACHE (default ~/nas/dataset/roboeval) on first run
```

- `pyproject.toml` is the authoritative dependency list (`setup.py` is legacy and pins mujoco 3.1.5). The released demos were recorded with mujoco 3.1.5, so with the pinned 3.3.3 every demo load logs `Demo replay could be unstable` — expected, not an install error. **Do not bump `mujoco`** casually. `numpy` is pinned to 1.26.x (pyquaternion).
- The `.gitmodules` URL is SSH; without GitHub SSH keys use `git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive`.
- On a headless box, `examples/1_data_replay.py` (`render_mode="human"`) needs Xvfb; its first run caches ~100 demos at ~30 s each (resumable if interrupted).
- `pytest` is listed under `[dev]` but there is no test suite in the repo; `examples/test_custom_robot.py` is the closest thing to a validation script (`python examples/test_custom_robot.py`).
- No linter/formatter config exists; the README asks for PEP 8.
- Headless: `Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99`. Use `render_mode=None` for speed.
- Clear the demo cache with `rm -rf ~/nas/dataset/roboeval/roboeval_demos` (or wherever `ROBOEVAL_CACHE` points) if demo loading misbehaves.

Data collection (Hydra; config in `roboeval/configs/data_collection.yaml`, overrides via `key=value`):
```bash
cd roboeval   # required: demo_recorder.py does `from base import InputMode` (relative to its own dir)
python data_collection/demo_recorder.py input_mode=Keyboard robot="Bimanual Panda" env="Lift Pot"
python data_collection/demo_recorder.py input_mode=Oculus robot="Bimanual Panda" env="Cube Handover"
```
`env` / `robot` values are the human-readable keys of `ENVIRONMENTS` / `ROBOTS` in `tools/shared/utils.py`, not class names.

Other tools: `python -m tools.demo_player.main` (GUI replay), `python -m tools.demo_recorder.main`, `python tools/usd_exporter/usd_exporter.py --demo-path X --output-path Y` (needs `pip install mujoco[usd]`), `tools/render_video.py`.

## Architecture

### Environment stack (`roboeval/roboeval_env.py`)
`RoboEvalEnv(gym.Env)` is the single base class. Construction order matters:
1. Loads `envs/xmls/world.xml` into a `Mojo` instance (physics dt 0.002s = 500 Hz; `control_frequency` 20–500 Hz sets how many physics substeps per `step()`).
2. Reads the task's `_PRESET_PATH` YAML (`envs/presets/*.yaml`) for `robot` start pose, `spawns` (randomization regions → `SpawnBoundary`, accessed via `get_spawn_boundary(name)`), and `props`.
3. Instantiates the robot (`robot_cls` or `DEFAULT_ROBOT = BimanualPanda`), `Preset` (props from YAML, resolved by class name in `roboeval.envs.props`), `Arena`.
4. Calls `_initialize_env()` — the task hook where props are created/fetched (`self._preset.get_props(BaseCabinet)`) and metrics are initialized.
5. Builds action/observation spaces, cameras, renderers, and `GenericUpperBodyIK` (`ik/base_ik.py`) used for end-effector action modes.

Task subclasses override: `_initialize_env`, `_on_reset`, `_on_step`, `_success`, `_fail`, `_get_task_info`, `_get_task_privileged_obs(_space)`. `success`/`fail`/`reward` are cached per step via `CallablesCache` — call the property, not `_success()` directly. `step(action=None)` runs physics only (useful to let objects settle); `step(..., fast=True)` skips observations (used by teleop).

### Tasks (`roboeval/envs/*.py`)
Each file holds a task family; each family has a base class plus `Position` / `Orientation` / `PositionAndOrientation` variants that only override `_on_reset` to randomize via spawn boundaries / random yaw. Task files: `lift_pot.py`, `lift_tray.py`, `manipulation.py` (StackTwoBlocks, CubeHandover), `stack_books.py`, `pack_objects.py`, `rotate_utility_objects.py`. Environment classes are resolved **by class name** at demo-load time (`find_class_in_module(roboeval.envs, name)` in `demonstrations/utils.py`), so renaming an env class breaks loading of existing demos. New tasks must also be registered in `ENVIRONMENTS` in `tools/shared/utils.py` to appear in teleop/GUI tools.

### Metrics (`roboeval/utils/metric_rollout.py`)
`MetricRolloutEval` is a mixin used as `class Task(RoboEvalEnv, ABC, MetricRolloutEval)`. Protocol: `_metric_init(...)` in `_initialize_env`/`_on_reset`, `_metric_step()` in `_on_step`, `_metric_stage(idx)` to mark subtask milestones, `_metric_finalize(success_flag=..., target_distance=..., pose_error=...)` returns the metrics dict which tasks expose through `_get_task_info()` → `info`. Tracks coordination (velocity/vertical sync), collisions, slips, jerk, path lengths, subtask progress. See the README "Detailed Metrics System" section for the metric key reference.

### Robots (`roboeval/robots/`)
`Robot` (`robot.py`) builds the model from a `RobotConfig` (`config.py`: arms per `HandSide`, gripper, floating base, actuators, cameras, model XML from `thirdparty/mujoco_menagerie`) plus a `RobotIKConfig`. Concrete robots live in `robots/configs/panda.py` (`BimanualPanda`, `SinglePanda`). Robots are also resolved by class name from demo metadata.

### Action modes (`roboeval/action_modes.py`)
`JointPositionActionMode(absolute, ee, floating_base, floating_dofs, block_until_reached)` is the only concrete mode. Action vector layout: `[floating-base dofs][arm joints or 6D EE pose per arm][one value per gripper]`. `ee=True` routes through IK; `absolute=False` means deltas. `floating_dofs=[]` with `floating_base=True` is the common configuration for the fixed-base Panda setup. Action-space bounds are scaled by the substep count, so they depend on `control_frequency`.

### Demonstrations (`roboeval/demonstrations/`)
- `Demo` / `LightweightDemo` (`demo.py`): list of `DemoStep`s saved as `.safetensors` with a `Metadata` header (env name, robot name, action-mode description, observation mode, camera config, package versions). Lightweight = no pixels.
- `DemoStore` (`demo_store.py`): cache under `<CACHE_PATH>/roboeval_demos/<DEMO_VERSION>/<robot>/<env>/<action_mode>/<obs_mode>[/<cameras>][/<freq>hz]/`. `CACHE_PATH` (`const.py`) is `$ROBOEVAL_CACHE`, defaulting to `~/nas/dataset/roboeval`. `get_demos(metadata, amount, frequency)` downloads the release zip from `DEMO_RELEASES` (`const.py`) on first use, then decimates 500 Hz demos to the requested frequency and re-renders observations by replaying them in a fresh env (`DemoConverter.create_demo_in_new_env`) — the first call for a new env/frequency combination is slow.
- `DemoConverter`: absolute↔delta, joint→EE conversions, decimation, clipping. `DemoPlayer`: `replay_in_env(demo, env, demo_frequency=...)` and `validate_in_env`.
- `Metadata.from_env(env)` is how you ask for demos matching an env configuration.

### Data collection (`roboeval/data_collection/`)
`demo_recorder.py` (Hydra entry) spawns the simulation in a subprocess with an `InputMode` (`keyboard_input.py`, `oculus_input.py`; VR needs PyOpenXR/GLIBC ≥ 2.32). Recordings go to `target_dir/<env_name>/`. Keyboard controls and the full Oculus/ADB setup are in `roboeval/data_collection/README.md`.

### Assets
`roboeval/envs/xmls/` (world, props, YCB, partnet) is package data; props map to XMLs via `ASSETS_PATH` in `envs/props/*.py`. `tools/ycb_*.py` download/convert YCB models and compute bounding boxes.
