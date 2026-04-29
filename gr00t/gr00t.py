from collections import deque
from dataclasses import dataclass, field
import threading
from typing import Dict

import numpy as np
import torch
from incar.extensions.ai import BasePolicy, LRSchedulerConfig, OptimizerConfig, PolicyConfig, NormalizationMode
from incar.extensions.ai import PolicyClientMarker
from incar.common import FeatureType, ProcessHook

@PolicyConfig.register_subclass("gr00t")
@dataclass
class GROOTPolicyConfig(PolicyConfig):
    # You can add fields for your policy configuration here. The base config already defines some
    # fields, like `dt`, `n_obs_steps` and `n_action_steps`. You can override standard values here,
    # or during training, values can be overriden with flags, e.g. --policy.device = "cpu"
    dt: float = 0.1
    port: int = 5555
    prompt: str = ""

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
        return [0]
    
    # """
    # Here you can configure a default optimizer for your policy, which is used when no optimizer
    # is configured in the training config
    # """
    # def get_default_optimizer(self) -> OptimizerConfig:
    #     raise NotImplementedError("GR00T is trained in an external repo")

    # """
    # Here you can configure a default scheduler for your policy, which is used when no scheduler
    # is configured in the training config
    # """
    # def get_default_scheduler(self) -> LRSchedulerConfig | None:
    #     raise NotImplementedError("GR00T is trained in an external repo")

    def get_default_optimizer(self):
        from incar.extensions.native.optimizers import AdamConfig
        return AdamConfig(
            lr = 1e-4,
            betas = (0.95, 0.999),
            eps = 1e-8,
            weight_decay= 1e-6 
        )
    
    def get_default_scheduler(self):
        from incar.extensions.native.schedulers import DiffuserSchedulerConfig
        return DiffuserSchedulerConfig(
            name="cosine",
            num_warmup_steps=500
        )

    """
    This can be used to validate that the features specified in the config are compatible with the policy implementation. 
    For example, if a certain policy architecture requires a fixed number of visual features, this can be checked here 
    and an error raised if the config is not compatible.
    """
    def validate_features(self) -> None:
        pass

    """
    Used when certain parameters are different during inference from training. For example, if during
    training a random crop is used but during inference you want to use a center crop, you can set
    this here.
    """ 
    def set_inference_params(self) -> None:
        pass
    
    def build_policy(self, stats):
        return PolicyClientMarker(self) # Let's training know training is done externally.
    
    def build_policy_from_existing_model(self, model_path):
        return GROOTPolicy(self)


"""
This is an example to show the required interface for your policy. The implementation can highly differ
based on your method and implementation. For clear examples, please check the documentation and
baselines provided in the incar_baselines package: https://github.com/INCAR-Robotics/incar_baselines
""" 
class GROOTPolicy(BasePolicy):
    def __init__(
        self, 
        config: GROOTPolicyConfig, 
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None
    ):
        super().__init__(config, dataset_stats)

        try:
            from .nvidia.policy_client import PolicyClient
            self._policy = PolicyClient(host="localhost", port=5555)
        except ImportError:
            raise ImportError("GR00T policy client requires additional dependencies. Please install the 'gr00t' extra, e.g. pip install incar[gr00t]")

        if not self._policy.ping():
            raise RuntimeError("Could not Connect to policy server! Reload the scenario to try connect again. Esnure that the server is running")
        self._queues = {}
        self._queues_lock = threading.Lock()

    def get_optim_params(self) -> dict:
        raise NotImplementedError("GR00T is trained in an external repo")
    
    def queue_observations(self, frame: dict[str, torch.Tensor]):
        self._queues_lock.acquire()
        for key in frame:
            if key not in self._queues:
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)
            if len(self._queues[key]) == 0:
                while len(self._queues[key]) != self._queues[key].maxlen:
                    self._queues[key].append(frame[key])
            self._queues[key].append(frame[key])
        self._queues_lock.release()

    @torch.no_grad
    def perform_inference(self) -> torch.Tensor:
        self._queues_lock.acquire()
        obs = {k: list(self._queues[k]) for k in self._queues}
        self._queues_lock.release()

        # Preprocess each individual frame
        for i in range(self.config.n_obs_steps):
            frame = {k: obs[k][i] for k in obs.keys()}
            self.config.preprocessing.process(frame, ProcessHook.OBSERVATION)

        # Add batch dimension
        for key in obs:
            # print(f"Key: {key}, shape before: {obs[key]}")
            obs[key] = torch.stack(obs[key], dim=0) # Shape (ObsHorizon, D) or (ObsHorizon, H, W, C)
            # print(f"Key: {key}, shape after stack: {obs[key].shape}")
            if key not in self.config.input_features.keys():
                continue
            if key == "left.arm.ee.pose":
                obs[key] = obs[key][:, :6]
            if self.config.input_features[key].type is FeatureType.VISUAL:
                obs[key] = obs[key].permute(0,2,3,1).unsqueeze(0).type(torch.uint8).to(self.config.device)
            else:
                obs[key] = obs[key].unsqueeze(0).type(torch.float32).to(self.config.device)
            print(f"Key: {key}, shape after adding batch dimension: {obs[key].shape}")

        groot_obs = {
            "state": {},
            "video": {},
            "language": {
                "annotation.human.task_description": [[self.config.prompt]]
            }
        }
        for key in self.config.image_features.keys():
            groot_obs['video'][key] = obs[key].cpu().numpy() # Shape (Batch, ObsHorizon, H, W, C)
        for key in self.config.state_features.keys():
            groot_obs['state'][key] = obs[key].cpu().numpy() # Shape (Batch, ObsHorizon, D)

        action_dict, info = self._policy.get_action(groot_obs) # each key in self.action has Shape: (Batch, ObsHorizon, D)

        # Send back to CPU and ensure correct shape
        for key, value in action_dict.items():
            action_dict[key] = value.squeeze()
            if action_dict[key].ndim == 1: 
                action_dict[key] = np.expand_dims(action_dict[key], -1)

            print(f"Action key: {key}, shape: {action_dict[key].shape}")

        return action_dict
    
    @torch.no_grad
    def validate_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("GR00T is trained in an external repo")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        raise NotImplementedError("GR00T is trained in an external repo")

