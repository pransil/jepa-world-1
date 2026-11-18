"""Thin GL wrapper over mujoco.Renderer. The only module in sim/ that needs an OpenGL context.

macOS uses CGL with no display needed. On Linux, set MUJOCO_GL=egl or osmesa. It is created only
when an episode is rendered, so physics-only code (the fast tier, CI) never touches GL.
"""

from __future__ import annotations

import mujoco
import numpy as np

CAMERA = "track"


class FrameRenderer:
    def __init__(self, model: mujoco.MjModel, size: int):
        self._r = mujoco.Renderer(model, height=size, width=size)

    def rgb(self, data: mujoco.MjData) -> np.ndarray:
        """(S, S, 3) uint8 from the tracking camera. data must be mj_forward-ed after moving the rig."""
        self._r.update_scene(data, camera=CAMERA)
        return self._r.render().copy()

    def segmentation(self, data: mujoco.MjData) -> np.ndarray:
        """(S, S) int16: the geom id visible at each pixel, or -1 for background."""
        self._r.enable_segmentation_rendering()
        try:
            self._r.update_scene(data, camera=CAMERA)
            out = self._r.render()
        finally:
            self._r.disable_segmentation_rendering()
        geom = out[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        return np.where(geom, out[..., 0], -1).astype(np.int16)

    def close(self) -> None:
        self._r.close()

    def __enter__(self) -> FrameRenderer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
