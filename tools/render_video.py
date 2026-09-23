#!/usr/bin/env python3

import sys
from pathlib import Path
import tempfile
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from demonstrations.demo import Demo
from demonstrations.demo import DemoStep
from demonstrations.demo_converter import DemoConverter

from roboeval.utils.observation_config import ObservationConfig, CameraConfig

import rich_click as click
from click_prompt import filepath_option
from rich.console import Console
from rich.progress import track

console = Console()

SUCCESS_COLOR = (0, 200, 0)
PENDING_COLOR = (128, 128, 128)


def draw_success(frame: np.ndarray, success: bool) -> np.ndarray:
    """Draw a "SUCCESS" label in the top right corner, green on success, gray otherwise."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, frame.shape[0] // 12))
    margin = frame.shape[0] // 40
    _, top, right, _ = draw.textbbox((0, 0), "SUCCESS", font=font)
    xy = (frame.shape[1] - margin - right, margin - top)
    color = SUCCESS_COLOR if success else PENDING_COLOR
    draw.text(xy, "SUCCESS", font=font, fill=color)
    return np.asarray(image)


@click.command()
@filepath_option("--demo-path", default="~/code/RobotOlympics/data/Bimanual Panda/0118e47678174c6ea83d0ef599d11d4a.safetensors", help="Recorded demo to load")
@filepath_option("--output-path", default="./videos", prompt=False, help="Default output folder to store the rendered videos")
@click.option("--cameras", default="head,external, left_wrist,right_wrist", help="Comma-separated camera names, tiled left to right")
@click.option("--resolution", default=256, help="Square resolution of each camera view")
def cli(demo_path, output_path, cameras, resolution):
    """
    Renderes a demo to a video
    """

    demo_path = Path(demo_path).expanduser()

    demo = Demo.from_safetensors(demo_path)

    frequency = 50
    camera_names = [c.strip() for c in cameras.split(",") if c.strip()]
    metadata = demo.metadata
    env = metadata.env_cls(
        action_mode=metadata.get_action_mode(),
        observation_config=ObservationConfig(
            cameras=[CameraConfig(name=c, resolution=(resolution, resolution)) for c in camera_names]
        ),
        render_mode=None,
        control_frequency=frequency,
        robot_cls=metadata.robot_cls,
    )
    robot_name = metadata.environment_data.robot_name
    env_name = metadata.environment_data.env_name
    demo_recorded_date = metadata.date
    output_path = Path(output_path).expanduser().absolute()

    if not output_path.exists():
        output_path.mkdir()

    if not output_path.is_dir():
        console.print("[red]Error[/red] output-path is not a folder")
        sys.exit(-1)

    output_video_path = output_path / f"{env_name}_{robot_name}_{demo_recorded_date}.mp4"

    console.print(f"[blue]Reading[/blue] demo from [gray]{demo_path}[/gray]")
    console.print(f"[blue]Writing[/blue] rendered video to [gray]{output_video_path}[/gray]")

    demo = DemoConverter.decimate(demo, frequency, robot=env.robot)

    # reset the env and replay the demo
    env.reset(seed=int(demo.seed))

    console.rule("Converting demo to video")

    try:
        rgb_frames = []
        for timestep in track(demo.timesteps, console=console, description="Rendering"):
            actual_timestep = DemoStep(
                *env.step(timestep.executed_action), timestep.executed_action
            )
            obs = actual_timestep.observation
            # Observations are CHW; tile the views side by side as one HWC frame
            frame = np.concatenate([np.moveaxis(obs[f"rgb_{c}"], 0, -1) for c in camera_names], axis=1)
            rgb_frames.append(draw_success(frame, actual_timestep.info["task_success"] > 0))
    except ValueError as e:
        console.print("[red]Error[/red] while rendering images: ", e)
    except KeyboardInterrupt:
        console.print("[orange]keyboard interrupt.[/orange]")
        rgb_frames.clear()
    finally:
        env.close()


    with tempfile.TemporaryDirectory() as rgb_image_output_folder:
        for i, rgb_image in track(enumerate(rgb_frames), console=console, description="Writing images"):
            Image.fromarray(rgb_image).save(f"{rgb_image_output_folder}/{i:04d}.png")
        cmd = f"ffmpeg  -r {frequency} -pattern_type glob -i '{rgb_image_output_folder}/*.png' -c:v libx264 -pix_fmt yuv420p {output_video_path}"
        subprocess.run(cmd, shell=True, check=True)
    console.rule("Done.")

if __name__ == "__main__":
    cli()

