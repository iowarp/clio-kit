"""A video of the scene's time steps, as the live view shows them.

``export_animation`` renders each step in the scene's own view (the one the person sees),
then encodes the frames: H.264 MP4 with ``ffmpeg`` when the ParaView machine has it, else
OGG Theora with VTK's own writer. The result says which, and why. A machine with neither
keeps the frames and says where. The scene's time step is put back afterwards.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

from . import manifest
from .errors import CommandError, _expand

#: Frames are encoded at an even size: H.264 in yuv420p needs one.
MAX_SIDE = 3840


def _even(value: int) -> int:
    value = max(64, min(int(value), MAX_SIDE))
    return value - value % 2


class SceneAnimation:
    """Mixed into the engine: render and encode the scene's time steps."""

    simple: Any
    view: Any
    root: Any

    # Provided by the engine.
    def require_scene(self) -> dict[str, Any]:
        raise NotImplementedError

    def timesteps(self) -> list[float]:
        raise NotImplementedError

    def _apply_time(self, scene: dict[str, Any]) -> None:
        raise NotImplementedError

    def cmd_export_animation(self, **args: Any) -> dict[str, Any]:
        frames = self.animating(**args)
        while True:
            try:
                next(frames)
            except StopIteration as done:
                return dict(done.value)

    def animating(
        self,
        path: str,
        width: int = 1280,
        height: int = 720,
        fps: float = 12.0,
        start: int | None = None,
        end: int | None = None,
        labels: list[str] | None = None,
    ) -> Generator[tuple[int, int], None, dict[str, Any]]:
        """export_animation a frame at a time: yields how many frames are rendered
        and how many there are, then encodes them and returns the video. Closing it
        early puts the scene's time back and removes the frames."""
        scene = self.require_scene()
        steps = self.timesteps()
        if len(steps) < 2:
            raise CommandError(
                "no_time_steps",
                "The scene has one time step, so there is nothing to animate. "
                "export_artifact writes a still figure.",
            )
        first = 0 if start is None else max(0, int(start))
        last = len(steps) - 1 if end is None else min(len(steps) - 1, int(end))
        if first >= last:
            raise CommandError(
                "bad_range", f"Steps {first} to {last} hold one frame or none."
            )
        if labels is not None and len(labels) != len(steps):
            raise CommandError(
                "bad_labels",
                f"labels has {len(labels)} entries; the scene has {len(steps)} time steps.",
            )
        target = Path(_expand(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        width, height = _even(width), _even(height)
        folder = manifest.scene_dir(self.root, scene["scene"]) / "exports"
        folder.mkdir(parents=True, exist_ok=True)
        frames = Path(tempfile.mkdtemp(prefix="frames-", dir=folder))
        simple = self.simple
        stamp = simple.Text(Text="") if labels is not None else None
        if stamp is not None:
            shown = simple.Show(stamp, self.view)
            shown.WindowLocation = "Upper Left Corner"
            shown.FontSize = 22
        animation = simple.GetAnimationScene()
        count = last - first + 1
        rendered = False
        try:
            for frame, index in enumerate(range(first, last + 1)):
                if stamp is not None and labels is not None:
                    stamp.Text = str(labels[index])
                animation.AnimationTime = steps[index]
                self.view.ViewTime = steps[index]
                simple.SaveScreenshot(
                    str(frames / f"f_{frame:05d}.png"),
                    self.view,
                    ImageResolution=[width, height],
                )
                yield frame + 1, count
            rendered = True
        finally:
            if stamp is not None:
                simple.Hide(stamp, self.view)
                simple.Delete(stamp)
            self._apply_time(scene)
            simple.Render(self.view)
            if not rendered:
                shutil.rmtree(frames, ignore_errors=True)
        encoded = self._encode(frames, target, fps)
        shutil.rmtree(frames, ignore_errors=True)
        return {
            **encoded,
            "frames": count,
            "fps": float(fps),
            "seconds": round(count / float(fps), 2),
            "width": width,
            "height": height,
            "steps": [first, last],
            "size_bytes": Path(encoded["path"]).stat().st_size,
        }

    def _encode(self, frames: Path, target: Path, fps: float) -> dict[str, Any]:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            out = target.with_suffix(".mp4")
            done = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    f"{float(fps):g}",
                    "-i",
                    str(frames / "f_%05d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "20",
                    "-movflags",
                    "+faststart",
                    str(out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0:
                raise CommandError(
                    "encode_failed",
                    f"ffmpeg could not encode the frames: {done.stderr[-600:]}",
                )
            return {"path": str(out), "format": "mp4", "encoder": "ffmpeg (H.264)"}
        try:
            from vtkmodules.vtkIOImage import vtkPNGReader
            from vtkmodules.vtkIOOggTheora import vtkOggTheoraWriter
        except ImportError as exc:
            kept = target.parent / f"{target.stem}_frames"
            shutil.copytree(frames, kept, dirs_exist_ok=True)
            raise CommandError(
                "no_video_encoder",
                f"This ParaView machine has neither ffmpeg nor VTK's OGG writer ({exc}); "
                f"the frames are in {kept}.",
            ) from exc
        out = target.with_suffix(".ogv")
        writer = vtkOggTheoraWriter()
        writer.SetFileName(str(out))
        writer.SetRate(max(1, round(float(fps))))
        reader = vtkPNGReader()
        files = sorted(frames.glob("f_*.png"))
        reader.SetFileName(str(files[0]))
        reader.Update()
        writer.SetInputConnection(reader.GetOutputPort())
        writer.Start()
        for file in files:
            reader.SetFileName(str(file))
            reader.Update()
            writer.Write()
        writer.End()
        return {
            "path": str(out),
            "format": "ogv",
            "encoder": "VTK OGG Theora (this machine has no ffmpeg, so not MP4)",
        }
