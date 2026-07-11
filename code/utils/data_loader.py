import os
import numpy as np
import torch
import psutil
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def process_chunk(chunk: np.ndarray) -> List[str]:
    return [str(x).strip() for x in chunk]


def parallel_load_data(file_path: str, n_jobs: int = None) -> Tuple[np.ndarray, np.ndarray]:
    if n_jobs is None:
        n_jobs = max(1, psutil.cpu_count(logical=False) - 1)

    print(f"Loading data using {n_jobs} processes...")

    with np.load(file_path, mmap_mode='r') as data:
        features = data['features']
        labels = data['labels']
        total_samples = len(features)

        chunk_size = max(1, total_samples // n_jobs)
        chunks = []

        for i in range(0, total_samples, chunk_size):
            end = min(i + chunk_size, total_samples)
            chunks.append(features[i:end])

        processed_features = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            with tqdm(total=len(chunks), desc="Processing data chunks") as pbar:
                for chunk_result in executor.map(process_chunk, chunks):
                    processed_features.extend(chunk_result)
                    pbar.update(1)

        labels = labels.astype(np.int64, copy=True)

    return np.array(processed_features), labels


def estimate_memory_usage(n_samples: int, feature_dim: int) -> float:
    bytes_per_feature = 8
    bytes_per_label = 8
    total_bytes = n_samples * (feature_dim * bytes_per_feature + bytes_per_label)
    return total_bytes / (1024**3)


def get_optimal_chunk_size(total_samples: int, feature_dim: int) -> int:
    available_memory = psutil.virtual_memory().available
    memory_per_sample = feature_dim * 8
    optimal_chunk_size = int(available_memory * 0.2 / memory_per_sample)
    return min(optimal_chunk_size, total_samples)


def data_import(data_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    print(f"Loading data from {data_path}...")
    features, labels = parallel_load_data(data_path)
    num_labels = len(np.unique(labels))
    print(f"Loaded {len(features)} samples with {num_labels} classes")
    return features, labels, num_labels
