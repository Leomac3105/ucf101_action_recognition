# Este archivo implementa modelos de aprendizaje profundo para reconocimiento de acciones
# a partir de secuencias de esqueletos del conjunto de datos UCF101.
# Los comentarios y cadenas de ayuda están escritos en español.

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
    """Dataset de PyTorch para secuencias de esqueletos UCF101.

    Cada elemento del conjunto de datos es una tupla ``(secuencia, etiqueta)`` donde
    ``secuencia`` es un tensor de forma ``[T, V * C]`` que representa las coordenadas
    x e y concatenadas de todos los puntos clave en cada fotograma.
    La etiqueta es un índice de clase entero.
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
        """Inicializa el dataset.

        Parámetros
        ----------
        annotations : lista de ``dict``
            Campo ``annotations`` analizado del archivo de esqueletos en pickle.
        split_ids : lista de ``str``
            Identificadores de videos pertenecientes a esta partición (por ejemplo, ``train``).
        selected_classes : lista de ``int``, opcional
            Si se proporciona, solo se incluyen anotaciones cuyos labels están en esta lista.
            Útil para trabajar con un subconjunto de clases.
        max_samples_per_class : ``int``, opcional
            Limita el número de videos por clase.  Si es ``None``, se incluyen todos los videos.
        num_frames : ``int``
            Número de frames a muestrear de cada video.  Los videos más cortos se rellenan y los más largos se muestrean uniformemente.
        random_seed : ``int``
            Semilla aleatoria para muestreo reproducible.
        """
        super().__init__()
        self.num_frames = num_frames
        self.random = random.Random(random_seed)

        # Construir un mapeo del identificador de video a la anotación.
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
            # Aplicar el número máximo de muestras por clase si se solicita.
                if max_samples_per_class is not None:
                    count = per_class_counts.get(label, 0)
                    if count >= max_samples_per_class:
                        continue
                    per_class_counts[label] = count + 1
            # ann["keypoint"] forma: [M, T, V, C]. Usar la primera persona.
            keypoints = ann["keypoint"][0]  # shape [T, V, C]
            entries.append((keypoints, label))
        self.entries = entries

        # Mapear etiquetas a índices contiguos si se usa un subconjunto de clases.
        if selected_classes is not None:
            self.label_map = {orig: idx for idx, orig in enumerate(sorted(selected_classes))}
        else:
            unique_labels = sorted({label for _, label in entries})
            self.label_map = {label: label for label in unique_labels}

    def __len__(self) -> int:
        return len(self.entries)

    def _temporal_sample(self, keypoints: np.ndarray) -> np.ndarray:
        """Muestrea un número fijo de frames de una secuencia de esqueleto.

        Parámetros
        ----------
        keypoints : ndarray de forma [T, V, C]
            Secuencia original de coordenadas de puntos clave.

        Retorna
        -------
        ndarray de forma [self.num_frames, V, C]
            Secuencia re-muestreada uniformemente rellena o reducida a
            ``self.num_frames`` frames.
        """
        T, V, C = keypoints.shape
        if T >= self.num_frames:
            # Muestrear índices uniformemente sobre la dimensión temporal.
            indices = np.linspace(0, T - 1, num=self.num_frames, dtype=np.int32)
            sampled = keypoints[indices]
        else:
            # Rellenar repitiendo el último frame.
            pad_len = self.num_frames - T
            pad = np.repeat(keypoints[-1][np.newaxis, :, :], pad_len, axis=0)
            sampled = np.concatenate([keypoints, pad], axis=0)
        return sampled

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        keypoints, label = self.entries[idx]
        seq = self._temporal_sample(keypoints)  # [T, V, C]
        # Aplanar V y C en una sola dimensión por fotograma.
        seq = seq.reshape(self.num_frames, -1)  # [T, V*C]
        # Normalizar cada secuencia por separado.
        mean = np.mean(seq, axis=0, keepdims=True)
        std = np.std(seq, axis=0, keepdims=True) + 1e-6
        norm_seq = (seq - mean) / std
        tensor_seq = torch.from_numpy(norm_seq.astype(np.float32))
        return tensor_seq, self.label_map[label]


class BaselineMLP(nn.Module):
    """Perceptrón multicapa (MLP) simple como línea base para la clasificación de esqueletos.

    Este modelo ignora el orden temporal de la secuencia de esqueletos al aplanar
    todos los fotogramas en un único vector.  Pasa este vector a través de dos capas ocultas
    y produce puntuaciones de clase.
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
    """Clasificador basado en LSTM para secuencias de esqueletos.

    Procesa la secuencia con un LSTM y utiliza el estado oculto final
    para realizar la clasificación.
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
        # x: tensor de forma [tamaño_de_lote, T, dimensión_de_entrada]
        outputs, (h_n, _) = self.lstm(x)
        # Tomar el último estado oculto de la pila de capas LSTM
        last_hidden = h_n[-1]  # [tamaño_de_lote, dimensión_oculta]
        return self.fc(last_hidden)


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Función de agrupamiento (collate) para secuencias de esqueletos.

    Como todas las secuencias se re-muestrean a la misma longitud, simplemente
    se apilan a lo largo de la primera dimensión para formar un lote.
    """
    sequences, labels = zip(*batch)
    stacked = torch.stack(sequences, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return stacked, labels_tensor


def load_skeleton_annotations(annotation_path: str) -> Tuple[dict, List[dict]]:
    """Carga las anotaciones de esqueletos desde un archivo pickle.

    Devuelve una tupla ``(split, annotations)`` donde ``split`` es un
    diccionario que mapea nombres de partición a listas de identificadores de video,
    y ``annotations`` es una lista de diccionarios de anotación.
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
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de reconocimiento de acciones basado en esqueletos en UCF101."
    )
    parser.add_argument(
        "--annotation-path", type=str, required=True,
        help="Ruta al archivo de anotaciones de esqueletos (.pkl) de UCF101."
    )
    parser.add_argument(
        "--train-split", type=str, default="train",
        help="Nombre de la partición a utilizar para entrenamiento."
    )
    parser.add_argument(
        "--val-split", type=str, default="val",
        help="Nombre de la partición a utilizar para validación."
    )
    parser.add_argument(
        "--selected-classes", type=int, nargs="*",
        help="Lista de índices de clase a incluir (opcional)."
    )
    parser.add_argument(
        "--max-samples-per-class", type=int, default=None,
        help="Número máximo de videos por clase a incluir (opcional)."
    )
    parser.add_argument(
        "--num-frames", type=int, default=64,
        help="Número de fotogramas a muestrear de cada video."
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Tamaño de lote para el entrenamiento."
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Número de épocas de entrenamiento."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="Tasa de aprendizaje."
    )
    parser.add_argument(
        "--model", type=str, choices=["mlp", "lstm"], default="mlp",
        help="Arquitectura del modelo a utilizar (mlp o lstm)."
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Dimensión oculta para el MLP o el LSTM."
    )
    parser.add_argument(
        "--num-layers", type=int, default=2,
        help="Número de capas LSTM (solo para el modelo LSTM)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Dispositivo en el que se entrenará el modelo."
    )
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
