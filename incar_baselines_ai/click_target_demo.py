"""Point-and-click demo policy for the extension UI.

Shows a live camera view in a floating panel of the GUI. When the policy is executed it waits until
the user clicks a target in the image, marks it, and then outputs zero actions. It has no weights,
so it can be used to try out the extension UI without training anything: put a policy_config.json
with `"type": "click_target_demo"` in <workspace>/policies/<name>/checkpoints/best/model/ and select
it in the GUI (see the "Extension UI" page of the documentation).
"""
from dataclasses import dataclass, field
import threading
import time

import numpy as np
import torch
from incar.extensions.ai import BasePolicy, PolicyClientMarker, PolicyConfig, NormalizationMode
from incar.common import FeatureType
from incar.ui import UIContext, ImageClick, point


@PolicyConfig.register_subclass("click_target_demo")
@dataclass
class ClickTargetDemoConfig(PolicyConfig):
    # Observation key of the camera to show. Empty: the first visual input feature.
    camera: str = ""
    preview_fps: float = 10
    n_action_steps: int = 8

    normalization_mapping: dict[FeatureType, NormalizationMode] = field(
        default_factory=lambda: {
            FeatureType.VISUAL: NormalizationMode.VIDEO_ZERO_ONE,
            FeatureType.STATE: NormalizationMode.MIN_MAX,
            FeatureType.ACTION: NormalizationMode.MIN_MAX,
        }
    )

    @property
    def observation_relative_indices(self) -> list:
        return [0]

    @property
    def action_relative_indices(self) -> list:
        return list(range(self.n_action_steps))

    def get_default_optimizer(self):
        from incar.extensions.native.optimizers import AdamConfig
        return AdamConfig()

    def get_default_scheduler(self):
        return None

    def validate_features(self) -> None:
        pass

    def set_inference_params(self) -> None:
        pass

    def build_policy(self, stats):
        return PolicyClientMarker(self)  # nothing to train

    def build_policy_from_existing_model(self, model_path):
        return ClickTargetDemoPolicy(self)


class ClickTargetDemoPolicy(BasePolicy):
    def __init__(self, config: ClickTargetDemoConfig, dataset_stats=None):
        super().__init__(config, dataset_stats)
        self._frame: torch.Tensor | None = None
        self._frame_lock = threading.Lock()
        self.target: ImageClick | None = None
        self._preview_running = False

    def build_ui(self, ui: UIContext) -> None:
        panel = ui.panel("Dynamic UI Demo", width=480, height=460)
        self.status = panel.text("Press play, then click a target in the image", style="muted")
        self.view = panel.image(max_fps=self.config.preview_fps)
        with panel.row():
            panel.button("Reset target", on_click=self._reset)
            self.show_preview = panel.toggle("Toggle to enable/disable :)", value=True)

        # Stream camera frames while the policy is loaded, also when it is not executing
        self._preview_running = True
        threading.Thread(target=self._preview_loop, daemon=True, name="click-target-preview").start()

    def on_ui_closed(self) -> None:
        self._preview_running = False

    def _preview_loop(self):
        while self._preview_running:
            frame = self._latest_frame()
            if frame is not None and self.show_preview.value:
                self.view.set_image(frame)
            time.sleep(1.0 / self.config.preview_fps)

    def _reset(self):
        self.target = None
        self.view.overlays = []
        self.status.update(value="Click a target in the image", style="normal")

    def _latest_frame(self) -> torch.Tensor | None:
        with self._frame_lock:
            return self._frame

    def queue_observations(self, frame: dict[str, torch.Tensor]):
        key = self.config.camera or next(iter(self.config.image_features), None)
        if key is None:  # no visual features configured: use the first image-like observation
            key = next((k for k, v in frame.items() if isinstance(v, torch.Tensor) and v.dtype == torch.uint8 and v.ndim == 3), None)
        if key in frame:
            with self._frame_lock:
                self._frame = frame[key]

    def perform_inference(self) -> dict[str, np.ndarray]:
        if self.target is None:
            self.status.update(value="Dynamic UI! Wow! From a policy extension :)", style="warning")
            click = self.view.wait_for_click()
            self.target = click
            self.view.overlays = [point(click.x, click.y, color="lime", label="target")]
            self.status.update(value=f"Target at ({click.x:.0f}, {click.y:.0f})", style="normal")
        return {
            key: np.zeros((self.config.n_action_steps, feature.shape[0]), dtype=np.float32)
            for key, feature in self.config.action_features.items()
        }

    def close(self) -> None:
        self._preview_running = False

    def get_optim_params(self) -> dict:
        raise NotImplementedError("click_target_demo has nothing to train")

    def forward(self, batch):
        raise NotImplementedError("click_target_demo has nothing to train")

    def validate_batch(self, batch):
        raise NotImplementedError("click_target_demo has nothing to train")
