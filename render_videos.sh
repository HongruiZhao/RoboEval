# Run from the repo root with the roboeval conda env active.
export MUJOCO_GL=egl PYTHONPATH=roboeval
D=${ROBOEVAL_CACHE:-$HOME/nas/dataset/roboeval}/roboeval_demos/1.0.0/BimanualPanda
A=JointPositionActionMode_floating_absolute_joint/lightweight
T=CubeHandover
# sed -n 2p: second file
python tools/render_video.py --demo-path "$(ls $D/${T}/$A/*.safetensors | sed -n 1p )"                       --output-path videos/$T
python tools/render_video.py --demo-path "$(ls $D/${T}Position/$A/*.safetensors | sed -n 100p )"               --output-path videos/$T
python tools/render_video.py --demo-path "$(ls $D/${T}Orientation/$A/*.safetensors | sed -n 1p )"            --output-path videos/$T
python tools/render_video.py --demo-path "$(ls $D/${T}PositionAndOrientation/$A/*.safetensors | sed -n 1p )" --output-path videos/$T