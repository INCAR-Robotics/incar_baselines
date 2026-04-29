import argparse
import json
import os
import shutil
import h5py
import jsonlines
import pyarrow
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

from incar.config.dataset_config import DatasetConfig
from incar.common import PolicyFeature, FeatureType
import numpy as np

EPISODES_PATH = "meta/episodes.jsonl"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"

DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

class DatasetConverter:
    def new_feature_name(self, feature: PolicyFeature, original_name: str) -> str:
        if feature.type == FeatureType.VISUAL:
            return "observation.images." + original_name
        if feature.type == FeatureType.STATE:
            return "observation.state." + original_name
        if feature.type == FeatureType.ACTION:
            return "action"
        else:
            return None
        
    def convert_demo(self, demo_folder: str, demo_idx: int):
        incar_meta_path = Path(self.incar_root) / demo_folder / "meta.json"
        with open(incar_meta_path, 'r') as f:
            incar_meta = json.load(f)
        
        # TODO: Merge statistics! Statistics are a dict with as key the feature names, and value the statistics.
        # They should be merged similarly to the demo_data
        statistics = incar_meta.get("statistics", {})
        statistics["observation.state"] = []
        statistics["action"] = []
        for feature in self.config.features.keys():
            if self.config.features[feature].type == FeatureType.STATE:
                statistics["observation.state"].append(statistics[feature])
            if self.config.features[feature].type == FeatureType.ACTION:
                statistics["action"].append(statistics[feature])

        statistics["observation.state"] = {
            "mean": np.concatenate([stat["mean"] for stat in statistics["observation.state"]], axis=-1).tolist(),
            "std": np.concatenate([stat["std"] for stat in statistics["observation.state"]], axis=-1).tolist(), 
            "min": np.concatenate([stat["min"] for stat in statistics["observation.state"]], axis=-1).tolist(),
            "max": np.concatenate([stat["max"] for stat in statistics["observation.state"]], axis=-1).tolist(),
            "count": statistics["observation.state"][0]["count"]
        }
        statistics["action"] = {
            "mean": np.concatenate([stat["mean"] for stat in statistics["action"]], axis=-1).tolist(),
            "std": np.concatenate([stat["std"] for stat in statistics["action"]], axis=-1).tolist(),
            "min": np.concatenate([stat["min"] for stat in statistics["action"]], axis=-1).tolist(),
            "max": np.concatenate([stat["max"] for stat in statistics["action"]], axis=-1).tolist(),
            "count": statistics["action"][0]["count"]
        }
        with jsonlines.open(self.lerobot_root / "meta" / "episodes_stats.jsonl", mode='a') as writer:
            writer.write({
                "episode_index": demo_idx,
                "stats": statistics
            })

        demo_data = {}

        for feature in self.config.features.keys():
            feature_path = Path(self.incar_root) / demo_folder / feature

            if self.config.features[feature].type == FeatureType.TEXT:
                continue

            if self.config.features[feature].type == FeatureType.VISUAL:
                new_feature_name = self.new_feature_name(self.config.features[feature], feature)
                os.makedirs((self.lerobot_root / VIDEO_PATH_TEMPLATE.format(episode_chunk=demo_idx//1000, video_key=new_feature_name, episode_index=demo_idx)).parent, exist_ok=True)
                shutil.copy(feature_path / "data.mp4", self.lerobot_root / VIDEO_PATH_TEMPLATE.format(episode_chunk=demo_idx//1000, video_key=new_feature_name, episode_index=demo_idx))

            else:
                with h5py.File(feature_path / "data.h5", "r") as f:
                    data = f["data"][:]
                demo_data[feature] = data

        demo_data["observation.state"] = []
        demo_data["action"] = []
        for feature in self.config.features.keys():
            if self.config.features[feature].type == FeatureType.STATE:
                demo_data["observation.state"].append(demo_data.pop(feature))
                print(f"Appended state feature {feature} to observation.state")
            if self.config.features[feature].type == FeatureType.ACTION:
                print(f"Appended action feature {feature} to action")
                demo_data["action"].append(demo_data.pop(feature))
        demo_data["observation.state"] = list(np.concatenate(demo_data["observation.state"], axis=-1))
        demo_data["action"] = list(np.concatenate(demo_data["action"], axis=-1))
        demo_data["timestamp"] = np.arange(len(demo_data["action"])) * self.config.data_timestep
        demo_data["task_index"] = np.array([0] * len(demo_data["action"]))

        table: pyarrow.Table = pyarrow.Table.from_pydict(demo_data)
        os.makedirs((self.lerobot_root / DATA_PATH_TEMPLATE.format(episode_chunk=demo_idx//1000, episode_index=demo_idx)).parent, exist_ok=True)
        pq.write_table(table, self.lerobot_root / DATA_PATH_TEMPLATE.format(episode_chunk=demo_idx//1000, episode_index=demo_idx))
        
        with jsonlines.open(self.lerobot_root / "meta" / "episodes.jsonl", mode='a') as writer:
            self.total_frames += len(demo_data["action"])
            writer.write({
                "episode_index": demo_idx,
                "tasks": [self.task],
                "length": len(demo_data["action"])
            })

    def run(self, args):
        self.incar_root: Path = Path(args.workspace).expanduser().resolve() / "datasets_processed" / args.name
        self.lerobot_root: Path = Path(args.workspace).expanduser().resolve() / "datasets_lerobot" / args.name
        self.lerobot_root.mkdir(parents=True, exist_ok=True)

        self.config: DatasetConfig = DatasetConfig.from_file(self.incar_root)

        subfolders = [ f.name for f in os.scandir(self.incar_root) if f.is_dir()]
        subfolders = sorted(subfolders, key=lambda x: int(x.split('_')[-1]))
        print("incar root: ", self.incar_root)
        print("num subfolders (demos): ", len(subfolders))

        info_dict = {
            "codebase_version": "v2.1",
            "robot_type": "",
            "chunks_size": 1000,
            "total_chunks": 1,
            "total_tasks": 1,
            "data_path": DATA_PATH_TEMPLATE,
            "video_path": VIDEO_PATH_TEMPLATE,
            "fps": 1/self.config.data_timestep,
            "total_episodes": len(subfolders),
            "total_tasks": 1,
            "total_videos": len(subfolders) * len([f for f in self.config.features.values() if f.type == FeatureType.VISUAL]),
            "splits": {
                "train": f"0:{len(subfolders)}",
            },
            "features": {}
        }
        state_size = 0
        action_size = 0
        for feature in self.config.features.keys():
            if self.config.features[feature].type == FeatureType.VISUAL:
                info_dict['features'][self.new_feature_name(self.config.features[feature], feature)] = {
                    "dtype": "video",
                    "shape": self.config.features[feature].shape
                }
            elif self.config.features[feature].type == FeatureType.STATE:
                state_size += self.config.features[feature].shape[0]
            elif self.config.features[feature].type == FeatureType.ACTION:
                action_size += self.config.features[feature].shape[0]
        info_dict['features']["observation.state"] = {
            "dtype": "float32",
            "shape": (state_size,)
        }
        info_dict['features']["action"] = {
            "dtype": "float32",
            "shape": (action_size,)
        }

        os.makedirs(self.lerobot_root / "meta", exist_ok=True)
        with jsonlines.open(self.lerobot_root / "meta" / "tasks.jsonl", mode='w') as writer:
            writer.write({
                "task_index": 0,
                "task": args.task
            })

        self.total_frames = 0
        self.task = args.task

        if len(subfolders) > 1000:
            print(f"[WARNING] LeRobot dataset has as default chunking 1000 demos. This converter however does handle chunking. \
                  The dataset to be converted has {len(subfolders)} demos")
        for idx, folder in enumerate(tqdm(subfolders)):
            self.convert_demo(folder, idx)
        
        info_dict["total_frames"] = self.total_frames
        with open(self.lerobot_root / "meta" / "info.json", 'w') as f:
            json.dump(info_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()
    converter = DatasetConverter()
    converter.run(args)