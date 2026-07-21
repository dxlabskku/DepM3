from pathlib import Path
from typing import Union, Optional

import torch
from torch.utils import data
import numpy as np
import pandas as pd
import pickle
import random


FIXED_FRAMES = 44
VISUAL_DIM = 136
ACOUSTIC_DIM = 25
MAX_VIDEOS_PER_USER = 294


class MUD3(data.Dataset):
    """MUD3 dataset - user-level depression detection from multiple videos.

    Each user has multiple videos of varying lengths. Each video is
    padded/truncated to FIXED_FRAMES (44) frames.
    Original 161-dim features are split into visual (136) and acoustic (25).
    Per-sample output: visual (num_videos, 44, 136), acoustic (num_videos, 44, 25).
    """
    def __init__(
        self, root: Union[str, Path], fold: str="train",
        gender: str="both", transform=None, target_transform=None, aug=False
    ):
        self.root = root if isinstance(root, Path) else Path(root)
        self.fold = fold
        self.gender = gender
        self.transform = transform
        self.target_transform = target_transform
        self.aug = aug

        self.visual_features = []
        self.acoustic_features = []
        self.labels = []

        label_df = pd.read_csv(self.root / "labels.csv", index_col=0)

        with open(self.root / "dep_feat.pkl", "rb") as f:
            dep_feat = pickle.load(f)
        with open(self.root / "nondep_feat.pkl", "rb") as f:
            nondep_feat = pickle.load(f)

        name_to_features = {}
        for name, user_videos in zip(dep_feat["name"], dep_feat["features"]):
            name_to_features[name] = user_videos
        for name, user_videos in zip(nondep_feat["name"], nondep_feat["features"]):
            name_to_features[name] = user_videos

        for _, row in label_df.iterrows():
            s_name = row["names"]
            s_label = int(row["labels"])
            s_split = row["split"]

            if s_split != self.fold:
                continue

            if s_name not in name_to_features:
                continue

            user_videos = name_to_features[s_name]
            visual, acoustic = self._align_videos(user_videos)
            self.visual_features.append(visual)
            self.acoustic_features.append(acoustic)
            self.labels.append(s_label)

            if self.aug and self.fold == "train":
                num_videos = visual.shape[0]
                for _ in range(5):
                    n_vids = int(random.random() * num_videos)
                    if n_vids < 2:
                        continue
                    v_start = random.randint(0, num_videos - n_vids)
                    self.labels.append(s_label)
                    self.visual_features.append(visual[v_start:v_start + n_vids])
                    self.acoustic_features.append(acoustic[v_start:v_start + n_vids])

        print(f"[MUD3-{self.fold}] ALL:{len(self.labels)}, "
              f"Positive:{np.sum(self.labels)}, "
              f"Negative:{len(self.labels) - np.sum(self.labels)}")

    @staticmethod
    def _align_videos(user_videos):
        """Pad/truncate each video to FIXED_FRAMES, split into visual/acoustic.

        Returns:
            visual:   (num_videos, FIXED_FRAMES, 136)
            acoustic: (num_videos, FIXED_FRAMES, 25)
        """
        aligned = []
        for video in user_videos:
            T = video.shape[0]
            if T >= FIXED_FRAMES:
                aligned.append(video[:FIXED_FRAMES])
            else:
                pad = np.zeros((FIXED_FRAMES - T, video.shape[1]), dtype=video.dtype)
                aligned.append(np.concatenate([video, pad], axis=0))
        stacked = np.stack(aligned, axis=0).astype(np.float32)
        visual = stacked[:, :, :VISUAL_DIM]
        acoustic = stacked[:, :, VISUAL_DIM:]
        return visual, acoustic

    def __getitem__(self, i: int):
        visual = self.visual_features[i]
        acoustic = self.acoustic_features[i]
        label = self.labels[i]
        if visual.shape[0] > MAX_VIDEOS_PER_USER:
            idx = np.sort(np.random.choice(visual.shape[0], MAX_VIDEOS_PER_USER, replace=False))
            visual = visual[idx]
            acoustic = acoustic[idx]
        return visual, acoustic, label

    def __len__(self):
        return len(self.labels)


def _collate_fn(batch):
    visuals, acoustics, labels = zip(*batch)

    max_num_videos = max(v.shape[0] for v in visuals)

    padded_visuals = []
    padded_acoustics = []
    padding_masks = []

    for v, a in zip(visuals, acoustics):
        n = v.shape[0]
        mask = np.ones(max_num_videos, dtype=np.int64)
        mask[n:] = 0
        padding_masks.append(mask)

        if n < max_num_videos:
            v_pad = np.zeros((max_num_videos - n, FIXED_FRAMES, VISUAL_DIM), dtype=np.float32)
            a_pad = np.zeros((max_num_videos - n, FIXED_FRAMES, ACOUSTIC_DIM), dtype=np.float32)
            v = np.concatenate([v, v_pad], axis=0)
            a = np.concatenate([a, a_pad], axis=0)

        padded_visuals.append(v)
        padded_acoustics.append(a)

    padded_visuals = torch.from_numpy(np.stack(padded_visuals, axis=0))
    padded_acoustics = torch.from_numpy(np.stack(padded_acoustics, axis=0))
    padding_masks = torch.from_numpy(np.stack(padding_masks, axis=0))
    labels = torch.tensor(labels)

    return padded_visuals, padded_acoustics, labels, padding_masks


def get_mud3_dataloader(
    root: Union[str, Path], fold: str="train", batch_size: int=8,
    gender: str="both",
    transform=None, target_transform=None, aug=True,
    drop_last: bool=False,
):
    """Get dataloader for MUD3 dataset.

    Args:
        root (Union[str, Path]): path to the MUD3 dataset directory
            containing dep_feat.pkl, nondep_feat.pkl, and labels.csv.
        fold (str, optional): train / val / test. Defaults to "train".
        batch_size (int, optional): Defaults to 8.
        gender (str, optional): Not used for MUD3, kept for API compatibility.
        transform (optional): Defaults to None.
        target_transform (optional): Defaults to None.
        aug (bool, optional): Whether to use data augmentation. Defaults to True.
        drop_last (bool, optional): Drop the last incomplete batch. Defaults to False.

    Returns:
        the dataloader. Each batch returns:
            visual:       (batch, max_num_videos, 44, 136)
            acoustic:     (batch, max_num_videos, 44, 25)
            labels:       (batch,)
            padding_mask: (batch, max_num_videos) - 1 for real videos, 0 for padding
    """
    dataset = MUD3(root, fold, gender, transform, target_transform, aug)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=_collate_fn,
        shuffle=(fold == "train"),
        drop_last=drop_last,
    )
    return dataloader


if __name__ == "__main__":
    root = "./data/MUD3"
    train_loader = get_mud3_dataloader(root, "train")
    print(f"train_loader: {len(train_loader.dataset)} samples")
    val_loader = get_mud3_dataloader(root, "val")
    print(f"val_loader: {len(val_loader.dataset)} samples")
    test_loader = get_mud3_dataloader(root, "test")
    print(f"test_loader: {len(test_loader.dataset)} samples")

    b = next(iter(train_loader))
    print(f"visual={b[0].shape}, acoustic={b[1].shape}, labels={b[2].shape}, mask={b[3].shape}")
    b = next(iter(val_loader))
    print(f"visual={b[0].shape}, acoustic={b[1].shape}, labels={b[2].shape}, mask={b[3].shape}")
    b = next(iter(test_loader))
    print(f"visual={b[0].shape}, acoustic={b[1].shape}, labels={b[2].shape}, mask={b[3].shape}")
