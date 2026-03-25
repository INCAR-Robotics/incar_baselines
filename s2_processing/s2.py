from dataclasses import dataclass, field
import fractions
import json
import os
from pathlib import Path
from typing import List, TYPE_CHECKING

import av
import numpy as np
from PIL import Image
import torch
import tqdm
if TYPE_CHECKING:
    import cv2

from incar.common import FeatureType, ProcessHook
from incar.utils import serialize_dict
from incar.extensions.processing_step import ProcessStep


DEMO_META_NAME = "meta.json"

@ProcessStep.register_subclass("depth_anything")
@dataclass
class DepthAnythingV3(ProcessStep):
    source_feature_name: str = ""
    target_feature_name: str = ""
    max_depth_value: int = 0

    model_id: str = "depth-anything/DA3-BASE"
    
    hooks: List[ProcessHook] = field(default_factory = lambda: [ProcessHook.DATASET, ProcessHook.OBSERVATION])

    def __post_init__(self):
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError:
            raise ImportError("Depth anything is not installed. Please install the S2 baseline dependencies with the following command: \n" \
            "pip install incar_baselines[s2]@git+https://github.com/INCAR-Robotics/incar_baselines")

        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.model = DepthAnything3.from_pretrained(self.model_id)
        self.model = self.model.to(device).eval()

    def process(self, root_path, config):        
        demos = [ f.name for f in os.scandir(root_path) if f.is_dir()]
        demos = sorted(demos, key=lambda x: int(x.split('_')[-1]))

        if config.features[self.source_feature_name].type != FeatureType.VISUAL:
            raise AssertionError(f"feature {self.source_feature_name} is not of type VISUAL, which is expected for grounded SAM")
        
        config.features[self.target_feature_name] = config.features[self.source_feature_name]

        max_global_depth = 0
        for demo in tqdm(demos, desc="Depth Anything first pass"):
            source_feature_path = Path(root_path) / demo / self.source_feature_name
            container = av.open(source_feature_path / "data.mp4")
            frames = []
            for frame in container.decode():
                frames.append(frame.to_ndarray(format='rgb24'))
            container.close()

            processed_frames = []
            for i, frame in enumerate(frames):
                depth = self.operation(frame)
                processed_frames.append(depth)

            max_frame_depth = np.max(np.array(processed_frames))
            max_global_depth = max(max_global_depth, max_frame_depth)

        self.max_depth_value = int(max_global_depth) + 1
        print(f"anydepth max depth: {self.max_depth_value}")
            
        for demo in tqdm(demos, desc="Depth Anything second pass"):
            source_feature_path = Path(root_path) / demo / self.source_feature_name
            target_feature_path = Path(root_path) / demo / self.target_feature_name

            container = av.open(source_feature_path / "data.mp4")
            frames = []
            for frame in container.decode():
                frames.append(frame.to_ndarray(format='rgb24'))
            container.close()

            processed_frames = []
            for i, frame in enumerate(frames):
                depth = self.operation(frame)
                processed_frames.append(depth)

            os.mkdir(target_feature_path)
            container = av.open(target_feature_path / "data.mp4", 'w')
            stream = container.add_stream('libx264', fractions.Fraction(config.video_fps))
            stream.height = frames[0].shape[0]
            stream.width = frames[0].shape[1]

            # Add all frames
            for i, frame in enumerate(processed_frames):
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pts = i
                video_frame.time_base = fractions.Fraction(1, config.video_fps)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            # Add ending frame
            for packet in stream.encode(None):
                container.mux(packet)
            container.close()

            data = np.array(processed_frames)
            with open(Path(root_path) / demo / DEMO_META_NAME, "r") as f:
                demo_meta = json.load(f)
            demo_meta["statistics"][self.target_feature_name] = serialize_dict({
                "min": np.min(data, axis=(0, 1, 2), keepdims=True),
                "max": np.max(data, axis=(0, 1, 2), keepdims=True),
                "mean": np.mean(data, axis=(0, 1, 2), keepdims=True),
                "std": np.std(data, axis=(0, 1, 2), keepdims=True),
                "count": np.array([len(data)]),
            })
            with open(Path(root_path) / demo / DEMO_META_NAME, "w") as f:
                f.write(json.dumps(demo_meta))

    def process_single_frame(self, frame):
        if frame[self.source_feature_name].shape[0] == 3:
            frame[self.target_feature_name] = torch.tensor(self.operation(np.array(frame[self.source_feature_name]).transpose(1, 2, 0))).permute(2,0,1)
        else:
            frame[self.target_feature_name] = torch.tensor(self.operation(np.array(frame[self.source_feature_name])))

    def operation(self, image: np.ndarray) -> np.ndarray:
        depth =  self.model.inference(image)
        # print("max depth: ", np.max(depth))
        # print("min depth: ", np.min(depth))
        if self.max_depth_value != 0:
            depth = depth * 255 / self.max_depth_value
            depth = depth.astype(np.uint8)
        else:
            depth = depth.astype(int)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        return depth


# @ProcessStep.register_subclass("depth_anything")
# @dataclass
# class DepthAnything(ProcessStep):
#     source_feature_name: str = ""
#     target_feature_name: str = ""
#     max_depth_value: int = 0

#     checkpoint_dir: str = ""
#     encoder: str = "vitl"

#     hooks: List[ProcessHook] = field(default_factory = lambda: [ProcessHook.DATASET, ProcessHook.OBSERVATION])

#     def __post_init__(self):
#         try:
#             from depth_anything_v2.dpt import DepthAnythingV2
#         except ImportError:
#             raise ImportError("Depth anything is not installed.")
        
#         model_configs = {
#             'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#             'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#             'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
#             'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
#         }

#         if self.encoder not in model_configs.keys():
#             raise ValueError(f"{self.encoder} is not a valid encoder. Valid models are {model_configs.keys()}")
        
#         device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

#         self.model = DepthAnythingV2(**model_configs[self.encoder])
#         self.model.load_state_dict(torch.load(self.checkpoint_dir / f"depth_anything_v2_{self.encoder}.pth", map_location='cpu'))
#         self.model = self.model.to(device).eval()

#     def process(self, root_path, config):        
#         demos = [ f.name for f in os.scandir(root_path) if f.is_dir()]
#         demos = sorted(demos, key=lambda x: int(x.split('_')[-1]))

#         if config.features[self.source_feature_name].type != FeatureType.VISUAL:
#             raise AssertionError(f"feature {self.source_feature_name} is not of type VISUAL, which is expected for grounded SAM")
        
#         config.features[self.target_feature_name] = config.features[self.source_feature_name]

#         max_global_depth = 0
#         for demo in tqdm(demos, desc="Depth Anything first pass"):
#             source_feature_path = Path(root_path) / demo / self.source_feature_name
#             container = av.open(source_feature_path / "data.mp4")
#             frames = []
#             for frame in container.decode():
#                 frames.append(frame.to_ndarray(format='rgb24'))
#             container.close()

#             processed_frames = []
#             for i, frame in enumerate(frames):
#                 depth = self.operation(frame)
#                 processed_frames.append(depth)

#             max_frame_depth = np.max(np.array(processed_frames))
#             max_global_depth = max(max_global_depth, max_frame_depth)

#         self.max_depth_value = int(max_global_depth) + 1
#         print(f"anydepth max depth: {self.max_depth_value}")
            
#         for demo in tqdm(demos, desc="Depth Anything second pass"):
#             source_feature_path = Path(root_path) / demo / self.source_feature_name
#             target_feature_path = Path(root_path) / demo / self.target_feature_name

#             container = av.open(source_feature_path / "data.mp4")
#             frames = []
#             for frame in container.decode():
#                 frames.append(frame.to_ndarray(format='rgb24'))
#             container.close()

#             processed_frames = []
#             for i, frame in enumerate(frames):
#                 depth = self.operation(frame)
#                 processed_frames.append(depth)

#             os.mkdir(target_feature_path)
#             container = av.open(target_feature_path / "data.mp4", 'w')
#             stream = container.add_stream('libx264', fractions.Fraction(config.video_fps))
#             stream.height = frames[0].shape[0]
#             stream.width = frames[0].shape[1]

#             # Add all frames
#             for i, frame in enumerate(processed_frames):
#                 video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
#                 video_frame.pts = i
#                 video_frame.time_base = fractions.Fraction(1, config.video_fps)
#                 for packet in stream.encode(video_frame):
#                     container.mux(packet)
#             # Add ending frame
#             for packet in stream.encode(None):
#                 container.mux(packet)
#             container.close()

#             data = np.array(processed_frames)
#             with open(Path(root_path) / demo / DEMO_META_NAME, "r") as f:
#                 demo_meta = json.load(f)
#             demo_meta["statistics"][self.target_feature_name] = serialize_dict({
#                 "min": np.min(data, axis=(0, 1, 2), keepdims=True),
#                 "max": np.max(data, axis=(0, 1, 2), keepdims=True),
#                 "mean": np.mean(data, axis=(0, 1, 2), keepdims=True),
#                 "std": np.std(data, axis=(0, 1, 2), keepdims=True),
#                 "count": np.array([len(data)]),
#             })
#             with open(Path(root_path) / demo / DEMO_META_NAME, "w") as f:
#                 f.write(json.dumps(demo_meta))

#     def process_single_frame(self, frame):
#         if frame[self.source_feature_name].shape[0] == 3:
#             frame[self.target_feature_name] = torch.tensor(self.operation(np.array(frame[self.source_feature_name]).transpose(1, 2, 0))).permute(2,0,1)
#         else:
#             frame[self.target_feature_name] = torch.tensor(self.operation(np.array(frame[self.source_feature_name])))

#     def operation(self, image: np.ndarray) -> np.ndarray:
#         depth =  self.model.infer_image(image)
#         # print("max depth: ", np.max(depth))
#         # print("min depth: ", np.min(depth))
#         if self.max_depth_value != 0:
#             depth = depth * 255 / self.max_depth_value
#             depth = depth.astype(np.uint8)
#         else:
#             depth = depth.astype(int)
#         depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
#         return depth


@ProcessStep.register_subclass("grounded_SAM")
@dataclass
class GroundedSAM(ProcessStep):
    source_feature_name: str = ""
    target_feature_name: str = ""

    prompts: List[str] = field(default_factory=list())

    run_during_inference: bool = True

    mask_replacement_color: List[int] = field(default_factory=lambda: [255, 255, 255])
    inverse_mask_replacement_color: List[int] = field(default_factory=lambda: [0, 0, 0])
    
    replace_mask_color: bool = True
    replace_inverse_mask_color: bool = True
    mask_dilation_pixels: int = 5

    sam2_model_id: str = ""
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-tiny"

    hooks: List[ProcessHook] = field(default_factory = lambda: [ProcessHook.DATASET, ProcessHook.OBSERVATION])

    def _to_pil_rgb(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        image = np.asarray(image)

        # CHW -> HWC if needed
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)

        image = np.ascontiguousarray(image)

        # dtype fix (handles your earlier float32 case too)
        if image.dtype != np.uint8:
            mx = float(image.max()) if image.size else 0.0
            if mx <= 1.0:
                image = (image * 255.0).round()
            image = np.clip(image, 0, 255).astype(np.uint8)

        return Image.fromarray(image, mode="RGB")

    def __post_init__(self):
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError("SAM2 is not installed. Please install the S2 baseline dependencies with the following command: \n" \
            "pip install incar_baselines[s2]@git+https://github.com/INCAR-Robotics/incar_baselines")
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is not installed, which is required for the Grounded SAM processing step. Please install it with 'pip install opencv-python'")
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.sam2_predictor = SAM2ImagePredictor.from_pretrained(self.sam2_model_id)

        self.processor = AutoProcessor.from_pretrained(self.grounding_dino_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_dino_model_id).to("cuda")

        self.prompt = ""
        for p in self.prompts:
            self.prompt += p.lower() + "."

        self.kernel = np.ones((self.mask_dilation_pixels, self.mask_dilation_pixels), np.uint8)

    def process(self, root_path, config):
        if config.features[self.source_feature_name].type != FeatureType.VISUAL:
            raise AssertionError(f"feature {self.source_feature_name} is not of type VISUAL, which is expected for grounded SAM")
        
        config.features[self.target_feature_name] = config.features[self.source_feature_name]

        demos = [ f.name for f in os.scandir(root_path) if f.is_dir()]
        demos = sorted(demos, key=lambda x: int(x.split('_')[-1]))

        for demo in tqdm(demos, desc="Segmenting using Grounded SAM 2"):
            source_feature_path = Path(root_path) / demo / self.source_feature_name
            target_feature_path = Path(root_path) / demo / self.target_feature_name

            os.makedirs(target_feature_path, exist_ok=True)

            container = av.open(source_feature_path / "data.mp4")
            frames = []
            for frame in container.decode():
                frames.append(frame.to_ndarray(format='rgb24'))
            container.close()

            processed_frames = []
            for i, frame in enumerate(frames):
                processed_frames.append(self.operation(frame))

            container = av.open(target_feature_path / "data.mp4", 'w')
            stream = container.add_stream('libx264', fractions.Fraction(config.video_fps))
            stream.height = frames[0].shape[0]
            stream.width = frames[0].shape[1]

            # Add all frames
            for i, frame in enumerate(processed_frames):
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pts = i
                video_frame.time_base = fractions.Fraction(1, config.video_fps)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            # Add ending frame
            for packet in stream.encode(None):
                container.mux(packet)
            container.close()

            data = np.stack(processed_frames)
            with open(Path(root_path) / demo / DEMO_META_NAME, "r") as f:
                demo_meta = json.load(f)
            demo_meta["statistics"][self.target_feature_name] = serialize_dict({
                "min": np.min(data, axis=(0, 1, 2), keepdims=True),
                "max": np.max(data, axis=(0, 1, 2), keepdims=True),
                "mean": np.mean(data, axis=(0, 1, 2), keepdims=True),
                "std": np.std(data, axis=(0, 1, 2), keepdims=True),
                "count": np.array([len(data)]),
            })
            with open(Path(root_path) / demo / DEMO_META_NAME, "w") as f:
                f.write(json.dumps(demo_meta))

    def operation(self, image: np.ndarray) -> np.ndarray:
        # print(image.shape)
        # print(image)
        image = self._to_pil_rgb(image)
        # image = Image.fromarray(image)

        self.sam2_predictor.set_image(image)
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        grounding_results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        input_boxes = grounding_results[0]["boxes"].cpu().numpy()
        if input_boxes.size == 0:
            return self.nothing_found_frame(image)
        # TODO: filter boxes

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        if masks.shape[0] == 0:
            return self.nothing_found_frame(image)

        combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            combined_mask += (mask * 255).astype(np.uint8)
        combined_mask = cv2.dilate(combined_mask.astype(np.uint8), self.kernel, iterations=1)

        image = np.array(image, dtype=np.uint8)
        if self.replace_inverse_mask_color:
            image[(combined_mask == 0)] = self.inverse_mask_replacement_color
        if self.replace_mask_color:
            image[(cv2.bitwise_not(combined_mask) == 0)] = self.mask_replacement_color
        return image

    def nothing_found_frame(self, image):
        image = np.array(image, dtype=np.uint8)
        if self.replace_inverse_mask_color:
            image[:, :, 0] = self.inverse_mask_replacement_color[0]
            image[:, :, 1] = self.inverse_mask_replacement_color[1]
            image[:, :, 2] = self.inverse_mask_replacement_color[2]
        return image
    
    def process_single_frame(self, frame):
        src = frame[self.source_feature_name]

        if isinstance(src, torch.Tensor):
            src = src.detach().cpu().numpy()
        elif isinstance(src, Image.Image):
            src = np.array(src)

        # src is now a numpy array; make sure it's HWC for operation
        if src.ndim == 3 and src.shape[0] == 3 and src.shape[-1] != 3:
            src = src.transpose(1, 2, 0)

        out = self.operation(src)                # returns numpy uint8 HWC (or convert inside)
        frame[self.target_feature_name] = torch.from_numpy(out).permute(2, 0, 1)

    # def process_single_frame(self, frame: dict[str, Union[np.ndarray, torch.Tensor]]):
    #     if frame[self.source_feature_name].shape[0] == 3:
    #         frame[self.target_feature_name] = torch.tensor(self.operation(np.array(frame[self.source_feature_name]).transpose(1, 2, 0))).permute(2,0,1)
    #     else:
    #         frame[self.target_feature_name] = torch.tensor(self.operation(np.array(frame[self.source_feature_name])))

@ProcessStep.register_subclass("combine_single_channels")
@dataclass
class CombineSingleChannels(ProcessStep):
    first_feature: str = ""
    second_feature: str = ""
    third_feature: str = ""

    target_feature_name: str = ""

    hooks: List[ProcessHook] = field(default_factory = lambda: [ProcessHook.DATASET, ProcessHook.OBSERVATION])

    def __post_init__(self):
        self.has_third_feature = self.third_feature is not None and self.third_feature != ""
    
    def process(self, root_path, config):
        if config.features[self.first_feature] != config.features[self.second_feature]:
            raise Exception("Trying to combine single channels of features with different shapes")
        if self.has_third_feature and (config.features[self.first_feature] != config.features[self.third_feature]):
            raise Exception("Trying to combine single channels of features with different shapes")

        config.features[self.target_feature_name] = config.features[self.first_feature]

        demos = [ f.name for f in os.scandir(root_path) if f.is_dir()]
        demos = sorted(demos, key=lambda x: int(x.split('_')[-1]))

        for demo in tqdm(demos, desc="Combining single channel features"):
            first_feature_path = Path(root_path) / demo / self.first_feature
            second_feature_path = Path(root_path) / demo / self.second_feature
            if self.has_third_feature:
                third_feature_path = Path(root_path) / demo / self.third_feature
            target_feature_path = Path(root_path) / demo / self.target_feature_name

            os.makedirs(target_feature_path, exist_ok=True)

            container = av.open(first_feature_path / "data.mp4")
            first_frames = []
            for frame in container.decode():
                first_frames.append(frame.to_ndarray(format='rgb24'))
            container.close()

            container = av.open(second_feature_path / "data.mp4")
            second_frames = []
            for frame in container.decode():
                second_frames.append(frame.to_ndarray(format='rgb24'))
            container.close()

            if self.has_third_feature:
                container = av.open(third_feature_path / "data.mp4")
                third_frames = []
                for frame in container.decode():
                    third_frames.append(frame.to_ndarray(format='rgb24'))
                container.close()

            processed_frames = []
            for i in range(len(first_frames)):
                if self.has_third_feature:
                    processed_frames.append(self.operation(first_frames[i], second_frames[i], third_frames[i]))
                else:
                    processed_frames.append(self.operation(first_frames[i], second_frames[i], None))

            container = av.open(target_feature_path / "data.mp4", 'w')
            stream = container.add_stream('libx264', fractions.Fraction(config.video_fps))
            stream.height = first_frames[0].shape[0]
            stream.width = first_frames[0].shape[1]

            # Add all frames
            for i, frame in enumerate(processed_frames):
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                video_frame.pts = i
                video_frame.time_base = fractions.Fraction(1, config.video_fps)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            # Add ending frame
            for packet in stream.encode(None):
                container.mux(packet)
            container.close()

            data = np.stack(processed_frames)
            with open(Path(root_path) / demo / DEMO_META_NAME, "r") as f:
                demo_meta = json.load(f)
            demo_meta["statistics"][self.target_feature_name] = serialize_dict({
                "min": np.min(data, axis=(0, 1, 2), keepdims=True),
                "max": np.max(data, axis=(0, 1, 2), keepdims=True),
                "mean": np.mean(data, axis=(0, 1, 2), keepdims=True),
                "std": np.std(data, axis=(0, 1, 2), keepdims=True),
                "count": np.array([len(data)]),
            })
            with open(Path(root_path) / demo / DEMO_META_NAME, "w") as f:
                f.write(json.dumps(demo_meta))
    
    def process_single_frame(self, frame):
        if frame[self.first_feature].shape[0] == 3:
            third_feature = None
            if self.has_third_feature:
                third_feature = np.array(frame[self.third_feature]).transpose(1, 2, 0)

            frame[self.target_feature_name] = torch.tensor(
                self.operation(
                    np.array(frame[self.first_feature]).transpose(1, 2, 0),
                    np.array(frame[self.second_feature]).transpose(1, 2, 0),
                    third_feature,
                )
            ).permute(2,0,1)
        else:
            third_feature = None
            if self.has_third_feature:
                third_feature = np.array(frame[self.third_feature])
            frame[self.target_feature_name] = torch.tensor(
                self.operation(
                    np.array(frame[self.first_feature]),
                    np.array(frame[self.second_feature]),
                    third_feature,
                )
            )
    
    def operation(self, first, second, third):
        first[:,:,0] = second[:,:,0]
        if third is not None:
            first[:,:,1] = third[:,:,0]
        return first


