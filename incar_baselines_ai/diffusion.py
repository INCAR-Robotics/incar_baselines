# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 INCAR Robotics AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

The majority of changes from Incar Robotics AB involve adapting the policy to interface with the Incar framework.
"""

from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Callable, Dict, Union

import numpy as np
from incar.extensions.ai import BasePolicy, PolicyConfig, NormalizationMode, action_tensor_to_dict
from incar.common import FeatureType, ProcessHook
import math
import copy
import einops
from einops import reduce
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


@PolicyConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(PolicyConfig):
    """Configuration class for DiffusionPolicy.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Diffusion model action prediction size as detailed in `DiffusionPolicy.select_action`.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `DiffusionPolicy.select_action` for more details.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
            within the image size. If None, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center crop in eval
            mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        use_separate_rgb_encoders_per_camera: Whether to use a separate RGB encoder for each camera view.
        down_dims: Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the diffusion modeling Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim: The Unet is conditioned on the diffusion timestep via a small non-linear
            network. This is the output dimension of that network, i.e., the embedding dimension.
        use_film_scale_modulation: FiLM (https://arxiv.org/abs/1709.07871) is used for the Unet conditioning.
            Bias modulation is used be default, while this parameter indicates whether to also use scale
            modulation.
        noise_scheduler_type: Name of the noise scheduler to use. Supported options: ["DDPM", "DDIM"].
        num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
        beta_schedule: Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
        beta_start: Beta value for the first forward-diffusion step.
        beta_end: Beta value for the last forward-diffusion step.
        prediction_type: The type of prediction that the diffusion modeling Unet makes. Choose from "epsilon"
            or "sample". These have equivalent outcomes from a latent variable modeling perspective, but
            "epsilon" has been shown to work better in many deep neural network settings.
        clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
            denoising step at inference time. WARNING: you will need to make sure your action-space is
            normalized to fit within this range.
        clip_sample_range: The magnitude of the clipping range as described above.
        num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
            spaced). If not provided, this defaults to be the same as `num_train_timesteps`.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for more information. Note, this defaults
            to False as the original Diffusion Policy implementation does the same.
    """

    # Inputs / output structure.
    # ----------------------------------------------------------------------------------------------
    # (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
    # |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
    # |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
    # |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
    # |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
    # ----------------------------------------------------------------------------------------------
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    obs_as_global_cond: bool = True

    normalization_mapping: dict[FeatureType, NormalizationMode] = field(
        default_factory=lambda: {
            FeatureType.VISUAL: NormalizationMode.VIDEO_ZERO_ONE,
            FeatureType.STATE: NormalizationMode.MIN_MAX,
            FeatureType.ACTION: NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    # vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    # pretrained_backbone_weights: str | None = None
    # use_group_norm: bool = True
    imagenet_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Unet.
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    # use_film_scale_modulation: bool = True
    cond_predict_scale = True
    # Noise scheduler.
    # noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    num_inference_timesteps: int = 10
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    # clip_sample_range: float = 1.0
    # set_alpha_to_one: bool = True
    steps_offset: int = 0

    # Loss computation
    do_mask_loss_for_padding: bool = False

    def __post_init__(self):
        super().__post_init__()
        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )

        # Check that the horizon size and U-Net downsampling is compatible.
        # U-Net downsamples by 2 with each stage.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor (which is determined "
                f"by `len(down_dims)`). Got {self.horizon=} and {self.down_dims=}"
            )
        
    def get_default_optimizer(self):
        from incar_baselines_ai.optimizers import AdamConfig
        return AdamConfig(
            lr = 1e-4,
            betas = (0.95, 0.999),
            eps = 1e-8,
            weight_decay= 1e-6 
        )
    
    def get_default_scheduler(self):
        from incar_baselines_ai.schedulers import DiffuserSchedulerConfig
        return DiffuserSchedulerConfig(
            name="cosine",
            num_warmup_steps=500
        )

    def validate_features(self) -> None:
        # TODO: Action feature shapes 1D
        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape.
        if len(self.image_features.items()) == 0: return
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )
            
    def set_inference_params(self) -> None:
        self.crop_is_random = False
    
    def build_policy(self, stats):
        return DiffusionPolicy(self, stats)
    
    def build_policy_from_existing_model(self, model_path):
        return DiffusionPolicy.load_from_safetensor(
            DiffusionPolicy(self),
            model_path,
            self.device
        )

    @property
    def observation_relative_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_relative_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))


class DiffusionPolicy(BasePolicy):
    def __init__(self, 
                 config: DiffusionConfig, 
                 dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None):
        super().__init__(config, dataset_stats)

        self.timestep = 0

        self.diffusion = DiffusionModel(config)
        self.to(config.device)

        self._queues = {}
        self._queues_lock = threading.Lock()

    def get_optim_params(self):
        return self.diffusion.parameters()
    
    def select_action(self, batch):
        return super().select_action(batch)

    def get_preprocessed_batch(self) -> dict[str, torch.Tensor]:
        # make shallow copy
        self._queues_lock.acquire()
        obs = {k: list(self._queues[k]) for k in self._queues}
        self._queues_lock.release()

        batch = dict()
        # pre-process each frame
        for i in range(self.config.n_obs_steps):
            frame = {k: obs[k][i] for k in obs.keys()}
            self.config.preprocessing.process(frame, ProcessHook.OBSERVATION)

            for key, value in self.config.input_features.items():
                if key not in frame.keys():
                    raise Exception(
                        f"Expected input feature {key} is not received by the system"
                    )
                if frame[key].shape != value.shape:
                    raise Exception(
                        f"Input feature {key} expects shape {value.shape} but the system received shape {frame[key].shape}"
                    )

                frame[key] = (frame[key]
                    .unsqueeze(0)
                    .type(torch.float32)
                    .to(self.config.device)
                )

            frame = self.normalize_inputs(frame)

            for key in self.config.input_features.keys():
                if not key in batch.keys():
                    batch[key] = [None]*self.config.n_obs_steps
                batch[key][i] = frame[key]
        
        # shape batch to required size
        for key in batch.keys():
            batch[key] = torch.stack(batch[key], dim=1)
        return batch

    @torch.no_grad
    def perform_inference(self) -> torch.Tensor:
        start = time.time()

        batch = self.get_preprocessed_batch()
        pp = time.time()

        actions = self.diffusion.predict_action(batch)
        predict = time.time()

        action_dict = action_tensor_to_dict(actions[0], self.config.action_features)
        action_dict = self.unnormalize_outputs(action_dict)
        for key, value in action_dict.items():
            action_dict[key] = value.to("cpu").numpy().squeeze()
            if action_dict[key].ndim == 1: 
                action_dict[key] = np.expand_dims(action_dict[key], -1)
        end = time.time()

        print(f"Inference times: pre = {(pp - start):.4f} s, pred = {(predict - pp):.4f} s, post = {(end - predict):.4f} s.", flush=True)
        return action_dict
    
    @torch.no_grad
    def validate_batch(self, batch) -> torch.Tensor:
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        
        action_mse = self.diffusion.compute_action_mse_loss(batch)
        return action_mse

    
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
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        self.timestep += 1

        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        
        loss = self.diffusion.compute_loss(batch)
        return loss, None
       


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        obs_encoder = MultiImageObsEncoder(config)
        # parse shapes
        # Sum up the shape of the action
        action_dim = 0
        for k in config.output_features.values():
            action_dim += k.shape[0]
        
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape(config)[0]

        # create diffusion model
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * config.n_obs_steps
        
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            cond_predict_scale=config.cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
            clip_sample=config.clip_sample,
            steps_offset=config.steps_offset,
        )
        
        self.horizon = config.horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = config.n_action_steps
        self.n_obs_steps = config.n_obs_steps
        self.obs_as_global_cond = config.obs_as_global_cond
        # self.kwargs = kwargs

        self.diffuser_num_train_steps = config.num_train_timesteps
        self.diffuser_num_inference_steps = config.num_inference_timesteps
    
    
    # ========= inference  ============
    def conditional_sample(self,
            global_cond=None
            ):
        model = self.model
        scheduler = self.noise_scheduler

        action = torch.randn(
            size=(global_cond.shape[0], self.horizon, self.action_dim),
            dtype=torch.float32,
            device=self.config.device,
        )
    
        # set step values
        scheduler.set_timesteps(self.diffuser_num_inference_steps)

        # Diffusion process
        for t in scheduler.timesteps:
            # Run model
            model_output = model(action, t, global_cond=global_cond)
            # Compute previous, refined, image: x_t -> x_t-1
            action = scheduler.step(
                model_output, t, action,
            ).prev_sample
        return action

    def encode_obs(self, obs_dict):
        batch_size = obs_dict[list(obs_dict.keys())[0]].shape[0]
        combined_obs_dict = {}
        for key, tensor in obs_dict.items():
            B, To, *rest = tensor.shape
            # Reshape to combine B and To into a single dimension
            combined_obs_dict[key] = tensor.reshape(B * To, *rest)
        ## Send the combined obs dict through the obs_encoder
        stacked_feature_vectors = self.obs_encoder(combined_obs_dict)
        ## Reshape
        global_cond = stacked_feature_vectors.view(batch_size, -1)
        return global_cond

    def predict_action(self, obs_dicts):
        global_cond = self.encode_obs(obs_dicts)
        # run sampling
        nsample = self.conditional_sample(
            global_cond=global_cond
        )
        return nsample
    
    def compute_action_mse_loss(self, batch):
        """
        Compute MSE between predicted actions and ground truth actions.
        Used for validation/evaluation to measure actual task performance.
        
        Args:
            batch: Dictionary with normalized observations and actions
            
        Returns:
            loss: Scalar MSE between predicted and ground truth actions
        """
        self.noise_scheduler.set_timesteps(self.diffuser_num_inference_steps)

        # Encode observations to get conditioning
        global_cond = self.encode_obs(batch)
        
        # Get ground truth actions (already normalized)
        action_tensor = torch.cat(
            [batch[key] for key in self.config.action_features], dim=-1
        )
        
        # Predict actions through full denoising process
        # This runs the full diffusion sampling (expensive but accurate)
        predicted_actions = self.conditional_sample(global_cond=global_cond)
        
        # Compute MSE loss
        # Compare only the action steps that will be executed (n_action_steps)
        # Note: predicted_actions shape is (batch_size, horizon, action_dim)
        #       action_tensor shape is (batch_size, horizon, action_dim)
        pred_action_steps = predicted_actions[:, :self.n_action_steps, :]
        gt_action_steps = action_tensor[:, :self.n_action_steps, :]
        
        # MSE loss
        loss = F.mse_loss(pred_action_steps, gt_action_steps)

        self.noise_scheduler.set_timesteps(self.diffuser_num_train_steps)

        return loss

    # to get the encodings with the actions:
    def conditional_sample_with_encodings(self,
            global_cond=None,
            # keyword arguments to scheduler.step
            **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        action = torch.randn(
            size=(global_cond.shape[0], self.horizon, self.action_dim),
            dtype=self.dtype,
            device=self.device,
        )
    
        # set step values
        scheduler.set_timesteps(self.diffuser_num_inference_steps)
        # Store encodings for each timestep
        encodings_list = []

        # Diffusion process
        for t in scheduler.timesteps:
            # Run model with encodings
            model_output, encodings = model.forward_with_encodings(action, t, global_cond=global_cond)
            # Store the encodings
            encodings_list.append(encodings)
            # Compute previous, refined action: x_t -> x_t-1
            action = scheduler.step(
                model_output, t, action,
            ).prev_sample

        return action, encodings_list

    def compute_loss(self, batch):
        # batch_size = batch['obs_dict'][list(batch['obs_dict'].keys())[0]].shape[0]
        batch_size = batch[self.config.feature_keys[0]].shape[0]
        global_cond = self.encode_obs(batch)
        
        # action_tensor = self.action_dict_to_action_tensor(batch['action_dict'])
        action_tensor = torch.cat(
            [batch[key] for key in self.config.action_features], dim=-1
        )
  
        # Generate the noise
        ## Sample noise
        noise = torch.randn(action_tensor.shape, device=action_tensor.device)

        ## Sample a random timesteps for each batch
        timesteps = torch.randint(
            0, 
            self.diffuser_num_train_steps, 
            (batch_size,), 
            device=action_tensor.device
        ).long()
        
        # Add noise to actions
        noisy_action = self.noise_scheduler.add_noise(
            action_tensor, 
            noise, 
            timesteps
        )

        # Predict the noise
        assert(noisy_action.shape == (batch_size, self.horizon, self.action_dim))
        assert(global_cond.shape == (batch_size, self.obs_feature_dim * self.n_obs_steps))
        pred_noise = self.model(
            noisy_action, 
            timesteps, 
            global_cond=global_cond
        )
        
        # Compute loss
        loss = F.mse_loss(pred_noise, noise, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss


class MultiImageObsEncoder(nn.Module):
    def __init__(self, config: DiffusionConfig):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        func = getattr(torchvision.models, "resnet18")
        resnet = func(weights=None)
        resnet.fc = torch.nn.Identity()
        rgb_model = resnet

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if not config.use_separate_rgb_encoder_per_camera:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        for key, feature in config.state_features.items():
            key_shape_map[key] = feature.shape
            low_dim_keys.append(key)

        for key, feature in config.image_features.items():
            key_shape_map[key] = feature.shape

            rgb_keys.append(key)
            # configure model for this key
            this_model = None
            if config.use_separate_rgb_encoder_per_camera:
                if isinstance(rgb_model, dict):
                    # have provided model for each key
                    this_model = rgb_model[key]
                else:
                    assert isinstance(rgb_model, nn.Module)
                    # have a copy of the rgb model
                    this_model = copy.deepcopy(rgb_model)
            
            if this_model is not None:
                if config.use_group_norm:
                    this_model = replace_submodules(
                        root_module=this_model,
                        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                        func=lambda x: nn.GroupNorm(
                            num_groups=x.num_features//16, 
                            num_channels=x.num_features)
                    )
                key_model_map[key] = this_model
            
            # configure randomizer
            this_randomizer = nn.Identity()

            # configure resize
            input_shape = feature.shape
            this_resizer = nn.Identity()
            if config.crop_shape is not None:
                h, w = config.crop_shape
                if config.crop_is_random:
                    this_resizer = torchvision.transforms.RandomCrop(
                        size=(h,w)
                    )
                else:
                    this_resizer = torchvision.transforms.CenterCrop(
                        size=(h,w)
                    )
            # configure normalizer
            this_normalizer = nn.Identity()
            if config.imagenet_norm:
                this_normalizer = torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
            key_transform_map[key] = this_transform
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = not config.use_separate_rgb_encoder_per_camera
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        self.image_saving_counter = 0

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]

                # folder = "debug_frames_inference"
                # if self.training:
                #     folder = "debug_frames_training"
                # print(f"saving img: {img.shape}")
                # unnormalized_img_one = img[0].cpu().numpy().transpose(1,2,0)*255
                # unnormalized_img_one = cv2.cvtColor(unnormalized_img_one, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f'/home/incar/incar_ws/{folder}/{key}_{self.image_saving_counter}_o{0}.png', unnormalized_img_one)
                # if img.shape[0] > 1:
                #     unnormalized_img_two = img[1].cpu().numpy().transpose(1,2,0)*255
                #     unnormalized_img_two = cv2.cvtColor(unnormalized_img_two, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite(f'/home/incar/incar_ws/{folder}/{key}_{self.image_saving_counter}_o{1}.png', unnormalized_img_two)
                # print("saved img")
                # self.image_saving_counter += 1

                img = self.key_transform_map[key](img)
                
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]

                
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
        
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    
    @torch.no_grad()
    def output_shape(self, config: DiffusionConfig):
        example_obs_dict = dict()
        batch_size = 1
        for key, value in config.input_features.items():
            shape = value.shape
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=torch.float32,
                device="cpu")
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape
    

class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv


    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


    #to get the encodings
    def forward_with_encodings(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None):
        """
        This method will return both the final output and intermediate encodings.
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # Encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        downsample_encodings = []  # Store downsample encodings
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            downsample_encodings.append(x)  # Store intermediate encoding
            x = downsample(x)

        mid_encodings = []  # Store mid-module encodings
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
            mid_encodings.append(x)

        upsample_encodings = []  # Store upsample encodings
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            upsample_encodings.append(x)  # Store intermediate encoding
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')

        return x, {
            "downsample_encodings": downsample_encodings,
            "mid_encodings": mid_encodings,
            "upsample_encodings": upsample_encodings
        }
    
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module