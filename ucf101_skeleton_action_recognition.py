"""
UCF101 skeleton‑based action recognition.

This script implements a small pipeline for classifying human actions
from 2D skeleton sequences derived from the UCF101 video dataset.  The
code is designed as a teaching example for how to build a deep
learning model that operates on skeleton data and can be extended to
other datasets.  It includes two architectures: a simple baseline
model that flattens the skeleton sequence and feeds it through a
multi‑layer perceptron (MLP), and a more expressive model that uses
an LSTM to capture temporal dynamics before classification.  Both
models are implemented using PyTorch.

The UCF101 skeleton annotations distributed by OpenMMLab are stored in
a pickled dictionary with two fields, ``split`` and ``annotations``.
Each skeleton annotation entry contains metadata about the video
(including the number of frames and the label) and an array of
keypoint coordinates of shape ``[M x T x V x C]``, where ``M`` is
the number of persons, ``T`` is the number of frames, ``V`` is the
number of keypoints and ``C`` is the coordinate dimension (2 for
2D skeletons)【374822187822292†L220-L247】.  This script expects the pickled
file to be downloaded separately (for example from
`https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl`) and
placed on disk.

Usage
-----
To train and evaluate the models, run the script from the command
line, specifying the path to the skeleton annotation file and
optionally a subset of classes:

.. code-block:: bash

    python ucf101_skeleton_action_recognition.py \
        --annotation-path ./ucf101_2d.pkl \
        --train-split train \
        --val-split val \
        --selected-classes 12 20 34 55 70 \
        --epochs 20 --batch-size 8 --model lstm

Because the full UCF101 dataset is large and training from scratch
requires significant computational resources, the code allows you to
select a subset of classes and limit the number of videos per class.
This makes it feasible to experiment on modest hardware.

Notes
-----
* This script does not perform any intensive pre‑processing on the
  skeletons.  Depending on your application you may wish to
  normalise the keypoints (e.g. centre them on the torso, scale them
  relative to subject height) or augment the data.
* When using all 101 classes, be aware that the distribution of
  examples across classes is imbalanced.  Stratified sampling or
  class weights may improve performance.
* For a baseline comparison, the original UCF101 paper reported
  a bag‑of‑visual‑words model achieving 44.5 % accuracy on the
  dataset【367233494771988†L18-L27】.  Modern 3D convolutional networks such as
  I3D can reach around 98 % accuracy when pre‑trained on Kinetics
  and fine‑tuned on UCF101【215100582023531†L20-L27】.

"""

import argparse
import os
import pickle
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class UCF101SkeletonDataset(Dataset):
    """PyTorch dataset for UCF101 skeleton sequences.

    Each item in the dataset is a tuple ``(sequence, label)`` where
    ``sequence`` is a tensor of shape ``[T, V * C]`` representing the
    concatenated x and y coordinates for all keypoints at each frame.
    The label is an integer class index.
    """

    def __init__(
        self,
        annotations: List[dict],
        split_ids: List[str],
        selected_classes: Optional[List[int]] = None,
        max_samples_per_class: Optional[int] = None,
        num_frames: int = 64,
        random_seed: int = 42,
    ):
        """Initialise the dataset.

        Parameters
        ----------
        annotations : list of dict
            Parsed ``annotations`` field from the pickled skeleton file.
        split_ids : list of str
            Identifiers of videos belonging to this split (e.g. train).
        selected_classes : list of int, optional
            If provided, only include annotations whose labels are in this
            list.  Useful for working with a subset of classes.
        max_samples_per_class : int, optional
            Limit the number of videos per class.  If ``None``, include all
            videos.
        num_frames : int
            Number of frames to sample from each video.  Videos shorter
            than this will be padded; longer videos will be uniformly
            downsampled.
        random_seed : int
            Random seed for reproducible sampling.
        """
        super().__init__()
        self.num_frames = num_frames
        self.random = random.Random(random_seed)

        # Build mapping from video identifier to annotation.
        video_to_ann = {ann["frame_dir"]: ann for ann in annotations}
        entries: List[Tuple[np.ndarray, int]] = []
        per_class_counts: dict[int, int] = {}
        for vid in split_ids:
            ann = video_to_ann.get(vid)
            if ann is None:
                continue
            label = ann["label"]
            if selected_classes is not None and label not in selected_classes:
                continue
            # Enforce maximum number of samples per class if requested.
            if max_samples_per_class is not None:
                count = per_class_counts.get(label, 0)
                if count >= max_samples_per_class:
                    continue
                per_class_counts[label] = count + 1
            # ann["keypoint"] shape: [M, T, V, C].  Use first person.
            keypoints = ann["keypoint"][0]  # shape [T, V, C]
            entries.append((keypoints, label))
        self.entries = entries

        # Map labels to contiguous indices if using a subset of classes.
        if selected_classes is not None:
            self.label_map = {orig: idx for idx, orig in enumerate(sorted(selected_classes))}
        else:
            unique_labels = sorted({label for _, label in entries})
            self.label_map = {label: label for label in unique_labels}

    def __len__(self) -> int:
        return len(self.entries)

    def _temporal_sample(self, keypoints: np.ndarray) -> np.ndarray:
        """Sample a fixed number of frames from a skeleton sequence.

        Parameters
        ----------
        keypoints : ndarray of shape [T, V, C]
            Original sequence of keypoint coordinates.

        Returns
        -------
        ndarray of shape [self.num_frames, V, C]
            Uniformly resampled sequence padded or downsampled to
            ``self.num_frames`` frames.
        """
        T, V, C = keypoints.shape
        if T >= self.num_frames:
            # Uniformly sample indices over the temporal dimension.
            indices = np.linspace(0, T - 1, num=self.num_frames, dtype=np.int32)
            sampled = keypoints[indices]
        else:
            # Pad by repeating the last frame.
            pad_len = self.num_frames - T
            pad = np.repeat(keypoints[-1][np.newaxis, :, :], pad_len, axis=0)
            sampled = np.concatenate([keypoints, pad], axis=0)
        return sampled

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        keypoints, label = self.entries[idx]
        seq = self._temporal_sample(keypoints)  # [T, V, C]
        # Flatten V and C into a single dimension per frame.
        seq = seq.reshape(self.num_frames, -1)  # [T, V*C]
        # Normalise each sequence separately.
        mean = np.mean(seq, axis=0, keepdims=True)
        std = np.std(seq, axis=0, keepdims=True) + 1e-6
        norm_seq = (seq - mean) / std
        tensor_seq = torch.from_numpy(norm_seq.astype(np.float32))
        return tensor_seq, self.label_map[label]


class BaselineMLP(nn.Module):
    """Simple baseline MLP for skeleton classification.

    This model ignores the temporal order of the skeleton sequence by
    flattening all frames into a single vector.  It passes this vector
    through two hidden layers and outputs class scores.
    """

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, T, feature_dim]
        batch_size, T, feature_dim = x.shape
        x_flat = x.reshape(batch_size, T * feature_dim)
        return self.net(x_flat)


class LSTMClassifier(nn.Module):
    """LSTM‑based classifier for skeleton sequences.

    Processes the sequence with an LSTM and uses the final hidden
    state for classification.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, T, input_dim]
        outputs, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # [batch_size, hidden_dim]
        return self.fc(last_hidden)



def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function to batch skeleton sequences.

    Since all sequences are resampled to the same length, we can simply
    stack them along the first dimension.
    """
    sequences, labels = zip(*batch)
    stacked = torch.stack(sequences, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return stacked, labels_tensor



def load_skeleton_annotations(annotation_path: str) -> Tuple[dict, List[dict]]:
    """Load skeleton annotations from a pickled file.

    Returns a tuple of (split, annotations) where ``split`` is a
    mapping from split names to lists of video identifiers, and
    ``annotations`` is a list of annotation dictionaries.
    """
    with open(annotation_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    split = data["split"]
    annotations = data["annotations"]
    return split, annotations



def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * sequences.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        outputs = model(sequences)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a skeleton‑based action recognition model on UCF101.")
    parser.add_argument("--annotation-path", type=str, required=True, help="Path to the UCF101 skeleton annotation .pkl file.")
    parser.add_argument("--train-split", type=str, default="train", help="Name of the split to use for training.")
    parser.add_argument("--val-split", type=str, default="val", help="Name of the split to use for validation.")
    parser.add_argument("--selected-classes", type=int, nargs="*", help="List of class indices to include.")
    parser.add_argument("--max-samples-per-class", type=int, default=None, help="Maximum number of videos per class to include.")
    parser.add_argument("--num-frames", type=int, default=64, help="Number of frames to sample from each video.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--model", type=str, choices=["mlp", "lstm"], default="mlp", help="Model architecture to use.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for MLP or LSTM.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers (used only for LSTM model).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on.")
    args = parser.parse_args()

    # Load dataset.
    split, annotations = load_skeleton_annotations(args.annotation_path)
    train_ids = split.get(args.train_split)
    val_ids = split.get(args.val_split)
    if train_ids is None or val_ids is None:
        raise ValueError(f"Specified train or val split not found in annotation file: {args.train_split}, {args.val_split}")

    # Create dataset and data loaders.
    train_dataset = UCF101SkeletonDataset(
        annotations=annotations,
        split_ids=train_ids,
        selected_classes=args.selected_classes,
        max_samples_per_class=args.max_samples_per_class,
        num_frames=args.num_frames,
    )
    val_dataset = UCF101SkeletonDataset(
        annotations=annotations,
        split_ids=val_ids,
        selected_classes=args.selected_classes,
        max_samples_per_class=args.max_samples_per_class,
        num_frames=args.num_frames,
    )

    # Determine input dimension and number of classes.
    sample_seq, _ = train_dataset[0]
    input_dim = sample_seq.shape[1]
    num_classes = len(train_dataset.label_map)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Instantiate model.
    if args.model == "mlp":
        model = BaselineMLP(input_dim * args.num_frames, num_classes, hidden_dim=args.hidden_dim)
    else:
        model = LSTMClassifier(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_classes=num_classes)

    device = torch.device(args.device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop.
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    main()
