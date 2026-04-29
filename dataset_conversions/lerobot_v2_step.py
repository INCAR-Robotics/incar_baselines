from dataclasses import dataclass, field
import json
import os
import shutil
from pathlib import Path
from typing import List

import h5py
import numpy as np

from tqdm import tqdm

from incar.common import FeatureType, ProcessHook
from incar.config.dataset_config import DatasetConfig
from incar.extensions.processing_step import ProcessStep

DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


@ProcessStep.register_subclass("lerobot_v2_conversion")
@dataclass
class LeRobotV2Conversion(ProcessStep):
    task: str = ""
    hooks: List[ProcessHook] = field(default_factory=lambda: [ProcessHook.DATASET])

    def __post_init__(self):
        try:
            import pyarrow
            import pyarrow.parquet as pq
            import jsonlines
        except ImportError:
            raise ImportError("LeRobot v2 conversion step requires additional dependencies. Please install the 'gr00t' extra, e.g. pip install incar[gr00t]")
        
    def _new_feature_name(self, feature, original_name: str) -> str:
        if feature.type == FeatureType.VISUAL:
            return "observation.images." + original_name
        if feature.type == FeatureType.STATE:
            return "observation.state." + original_name
        if feature.type == FeatureType.ACTION:
            return "action"
        return None

    def _convert_demo(self, demo_folder: str, demo_idx: int, incar_root: Path, lerobot_root: Path, config: DatasetConfig):
        import pyarrow
        import pyarrow.parquet as pq
        import jsonlines
        
        with open(incar_root / demo_folder / "meta.json", "r") as f:
            incar_meta = json.load(f)

        statistics = incar_meta.get("statistics", {})
        statistics["observation.state"] = []
        statistics["action"] = []
        for feature in config.features.keys():
            if config.features[feature].type == FeatureType.STATE:
                statistics["observation.state"].append(statistics[feature])
            if config.features[feature].type == FeatureType.ACTION:
                statistics["action"].append(statistics[feature])

        statistics["observation.state"] = {
            "mean": np.concatenate([s["mean"] for s in statistics["observation.state"]], axis=-1).tolist(),
            "std": np.concatenate([s["std"] for s in statistics["observation.state"]], axis=-1).tolist(),
            "min": np.concatenate([s["min"] for s in statistics["observation.state"]], axis=-1).tolist(),
            "max": np.concatenate([s["max"] for s in statistics["observation.state"]], axis=-1).tolist(),
            "count": statistics["observation.state"][0]["count"],
        }
        statistics["action"] = {
            "mean": np.concatenate([s["mean"] for s in statistics["action"]], axis=-1).tolist(),
            "std": np.concatenate([s["std"] for s in statistics["action"]], axis=-1).tolist(),
            "min": np.concatenate([s["min"] for s in statistics["action"]], axis=-1).tolist(),
            "max": np.concatenate([s["max"] for s in statistics["action"]], axis=-1).tolist(),
            "count": statistics["action"][0]["count"],
        }
        with jsonlines.open(lerobot_root / "meta" / "episodes_stats.jsonl", mode="a") as writer:
            writer.write({"episode_index": demo_idx, "stats": statistics})

        demo_data = {}
        for feature in config.features.keys():
            feature_path = incar_root / demo_folder / feature

            if config.features[feature].type == FeatureType.TEXT:
                continue

            if config.features[feature].type == FeatureType.VISUAL:
                new_name = self._new_feature_name(config.features[feature], feature)
                dest = lerobot_root / VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=demo_idx // 1000, video_key=new_name, episode_index=demo_idx
                )
                os.makedirs(dest.parent, exist_ok=True)
                shutil.copy(feature_path / "data.mp4", dest)
            else:
                with h5py.File(feature_path / "data.h5", "r") as f:
                    demo_data[feature] = f["data"][:]

        demo_data["observation.state"] = []
        demo_data["action"] = []
        for feature in config.features.keys():
            if config.features[feature].type == FeatureType.STATE:
                demo_data["observation.state"].append(demo_data.pop(feature))
            if config.features[feature].type == FeatureType.ACTION:
                demo_data["action"].append(demo_data.pop(feature))
        demo_data["observation.state"] = list(np.concatenate(demo_data["observation.state"], axis=-1))
        demo_data["action"] = list(np.concatenate(demo_data["action"], axis=-1))
        demo_data["timestamp"] = np.arange(len(demo_data["action"])) * config.data_timestep
        demo_data["task_index"] = np.array([0] * len(demo_data["action"]))

        parquet_path = lerobot_root / DATA_PATH_TEMPLATE.format(
            episode_chunk=demo_idx // 1000, episode_index=demo_idx
        )
        os.makedirs(parquet_path.parent, exist_ok=True)
        pq.write_table(pyarrow.Table.from_pydict(demo_data), parquet_path)

        with jsonlines.open(lerobot_root / "meta" / "episodes.jsonl", mode="a") as writer:
            self._total_frames += len(demo_data["action"])
            writer.write({
                "episode_index": demo_idx,
                "tasks": [self.task],
                "length": len(demo_data["action"]),
            })

    def process_dataset(self, root_path: str, config: DatasetConfig) -> None:
        import jsonlines

        incar_root = Path(root_path)
        lerobot_root = incar_root.parent.parent / "datasets_lerobot" / incar_root.name
        lerobot_root.mkdir(parents=True, exist_ok=True)

        subfolders = sorted(
            [f.name for f in os.scandir(incar_root) if f.is_dir()],
            key=lambda x: int(x.split("_")[-1]),
        )

        visual_features = [f for f in config.features.values() if f.type == FeatureType.VISUAL]
        info_dict = {
            "codebase_version": "v2.1",
            "robot_type": "",
            "chunks_size": 1000,
            "total_chunks": 1,
            "total_tasks": 1,
            "data_path": DATA_PATH_TEMPLATE,
            "video_path": VIDEO_PATH_TEMPLATE,
            "fps": 1 / config.data_timestep,
            "total_episodes": len(subfolders),
            "total_videos": len(subfolders) * len(visual_features),
            "splits": {"train": f"0:{len(subfolders)}"},
            "features": {},
        }

        state_size = 0
        action_size = 0
        for name, feature in config.features.items():
            if feature.type == FeatureType.VISUAL:
                info_dict["features"][self._new_feature_name(feature, name)] = {
                    "dtype": "video",
                    "shape": feature.shape,
                }
            elif feature.type == FeatureType.STATE:
                state_size += feature.shape[0]
            elif feature.type == FeatureType.ACTION:
                action_size += feature.shape[0]
        info_dict["features"]["observation.state"] = {"dtype": "float32", "shape": (state_size,)}
        info_dict["features"]["action"] = {"dtype": "float32", "shape": (action_size,)}

        os.makedirs(lerobot_root / "meta", exist_ok=True)
        with jsonlines.open(lerobot_root / "meta" / "tasks.jsonl", mode="w") as writer:
            writer.write({"task_index": 0, "task": self.task})

        self._total_frames = 0

        if len(subfolders) > 1000:
            print(
                f"[WARNING] LeRobot chunking is 1000 demos by default. "
                f"This dataset has {len(subfolders)} demos — chunking is not handled."
            )
        for idx, folder in enumerate(tqdm(subfolders, desc="Converting to LeRobot v2")):
            self._convert_demo(folder, idx, incar_root, lerobot_root, config)

        info_dict["total_frames"] = self._total_frames
        with open(lerobot_root / "meta" / "info.json", "w") as f:
            json.dump(info_dict, f)
