import torch
import numpy as np
import gc, os, sys
import torch.nn.functional as F
from collections import deque, Counter, defaultdict
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from custom_embedding_module import FeatureExtractor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from custom_fine_tuning import get_tunable_model, ModelConfig, TaskCategory
from model_hub import CustomTokenizer, CustomModelForClassification
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import torch.multiprocessing as mp

class DataWrapper(Dataset):
    def __init__(self, input_data: List[str], tokenizer_obj, seq_length: int = 128):
        self.input_data = input_data
        self.tokenizer_obj = tokenizer_obj
        self.seq_length = seq_length

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        text = self.input_data[index]
        tokenized = self.tokenizer_obj(
            str(text),
            truncation=True,
            padding='max_length',
            max_length=self.seq_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
        }

class AbstractSamplingStrategy(ABC):
    def __init__(self, model_obj, tokenizer_obj, **config):
        self.model_obj = model_obj
        self.tokenizer_obj = tokenizer_obj
        self.computation_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_obj = self.model_obj.to(self.computation_device)
        self._candidate_indices = None
        self._priority_queue = None

    def get_inference_results(self, input_data: List[str]) -> np.ndarray:
        self.model_obj.eval()
        results = []
        
        data_batch = DataWrapper(input_data, self.tokenizer_obj)
        loader = DataLoader(data_batch, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for data in loader:
                ids = data['input_ids'].to(self.computation_device)
                mask = data['attention_mask'].to(self.computation_device)
                
                output = self.model_obj(
                    input_ids=ids,
                    attention_mask=mask
                )
                
                probabilities = torch.softmax(output.logits, dim=-1)
                results.extend(probabilities.cpu().numpy())
                
        return np.array(results)

    @abstractmethod
    def rank_by_value(self, unlabeled_data: List[str], **config) -> List[int]:
        pass

    def setup_sampling(self, unlabeled_data: List[str], unlabeled_indices: List[int], **config) -> None:
        print("Initializing sampling strategy...")
        self._candidate_indices = unlabeled_indices.copy()
        priority_ranking = self.rank_by_value(unlabeled_data, **config)
        self._priority_queue = [self._candidate_indices[i] for i in priority_ranking]
        print(f"Sampling setup completed with {len(self._priority_queue)} candidates available")

    def get_next_samples(self, batch_size: int) -> List[int]:
        if len(self._priority_queue) < batch_size:
            print("Warning: Available samples less than requested batch size")
            batch_size = len(self._priority_queue)

        selected = self._priority_queue[:batch_size]
        self._priority_queue = self._priority_queue[batch_size:]
        return selected

    def get_available_count(self) -> int:
        return len(self._priority_queue) if self._priority_queue is not None else 0


class AdaptiveSamplingStrategy(AbstractSamplingStrategy):
    def __init__(self, model_obj, tokenizer_obj=None, **config):
        if tokenizer_obj is None:
            class PlaceholderTokenizer:
                def __call__(self, *args, **kwargs):
                    return {'input_ids': torch.tensor([]), 'attention_mask': torch.tensor([])}
            tokenizer_obj = PlaceholderTokenizer()
            
        super().__init__(model_obj, tokenizer_obj, **config)
        
        self.feature_network = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15)
        ).to(self.computation_device)
        
        self.prediction_head = torch.nn.Linear(128, 8).to(self.computation_device)
        
        self.buffer_capacity = config.get('buffer_capacity', 1000)
        self.memory_capacity = config.get('memory_capacity', 500)
        self.margin_threshold = config.get('margin_threshold', 0.15)
        self.distance_min = config.get('distance_min', 0.02)
        self.uncertainty_weight = config.get('uncertainty_weight', 0.45)
        self.balance_weight = config.get('balance_weight', 0.25)
        self.diversity_weight = config.get('diversity_weight', 0.3)
        self.smoothing_factor = config.get('smoothing_factor', 1.0)
        self.processing_size = config.get('processing_size', 32)
        
        self.boundary_samples = deque(maxlen=self.buffer_capacity)
        self.rare_class_samples = deque(maxlen=self.memory_capacity)
        self.feature_normalizer = StandardScaler()
        self.neighbor_finder = None
        self.label_distribution = Counter()

    def _standardize_features(self, feature_vectors):
        if isinstance(feature_vectors, np.ndarray) and feature_vectors.dtype == np.float32:
            return feature_vectors
            
        clean_vectors = []
        for vector in feature_vectors:
            if isinstance(vector, str):
                try:
                    vector = vector.replace('\n', ' ').strip('[]')
                    numeric_vector = np.fromstring(vector, sep=' ')
                    clean_vectors.append(numeric_vector)
                except Exception as e:
                    print(f"Warning: Feature preprocessing error: {str(e)}")
                    try:
                        values = [float(x) for x in vector.split() if x.strip()]
                        clean_vectors.append(np.array(values))
                    except:
                        raise ValueError(f"Cannot parse feature vector: {vector[:100]}...")
            else:
                clean_vectors.append(vector)
        
        return np.array(clean_vectors, dtype=np.float32)

    def get_inference_results(self, feature_vectors: np.ndarray) -> np.ndarray:
        try:
            standardized_vectors = self._standardize_features(feature_vectors)
            
            self.model_obj.eval()
            self.feature_network.eval()
            self.prediction_head.eval()
            inference_results = []
            
            for i in range(0, len(standardized_vectors), self.processing_size):
                batch_vectors = standardized_vectors[i:i + self.processing_size]
                tensor_batch = torch.FloatTensor(batch_vectors).to(self.computation_device)
                
                with torch.no_grad():
                    latent_representation = self.feature_network(tensor_batch)
                    logits = self.prediction_head(latent_representation)
                    probabilities = torch.softmax(logits, dim=-1)
                    inference_results.extend(probabilities.cpu().numpy())
            
            return np.array(inference_results)
            
        except Exception as e:
            print(f"Inference error: {str(e)}")
            print(f"Model type: {type(self.model_obj)}")
            print(f"Feature shape: {standardized_vectors.shape}")
            print(f"Batch tensor shape: {tensor_batch.shape if 'tensor_batch' in locals() else 'Not created'}")
            if 'latent_representation' in locals():
                print(f"Latent representation shape: {latent_representation.shape}")
            raise

    def _compute_diversity_metric(self, feature_vector: np.ndarray) -> float:
        try:
            if not self.rare_class_samples:
                return 1.0
                
            reference_vectors = self._feature_vectors[list(self.rare_class_samples)]
            
            distances = np.linalg.norm(reference_vectors - feature_vector, axis=1)
            return float(np.min(distances))
            
        except Exception as e:
            print(f"Diversity metric calculation error: {str(e)}")
            return 1.0

    def setup_sampling(self, feature_vectors: np.ndarray, candidate_indices: List[int], **config):
        print("Setting up adaptive sampling strategy...")
        
        try:
            standardized_vectors = self._standardize_features(feature_vectors)
            print(f"Feature vectors shape: {standardized_vectors.shape}")
            
            normalized_vectors = self.feature_normalizer.fit_transform(standardized_vectors)
            print(f"Normalized vectors shape: {normalized_vectors.shape}")
            
            self.neighbor_finder = NearestNeighbors(
                n_neighbors=min(50, len(candidate_indices)),
                algorithm='ball_tree'
            ).fit(normalized_vectors)
            
            self._feature_vectors = normalized_vectors
            self._candidate_indices = candidate_indices
            
            print("Identifying boundary regions...")
            self._identify_boundary_regions(standardized_vectors)
            
            print("Computing priority ranking...")
            priority_ranking = self.rank_by_value(standardized_vectors)
            self._priority_queue = [self._candidate_indices[i] for i in priority_ranking]
            
            print(f"Sampling setup completed with {len(candidate_indices)} candidates available")
            
        except Exception as e:
            print(f"Sampling setup error: {str(e)}")
            print(f"Feature shape: {feature_vectors.shape if feature_vectors is not None else 'None'}")
            raise

    def _identify_boundary_regions(self, feature_vectors: np.ndarray) -> None:
        try:
            inference_results = self.get_inference_results(feature_vectors)
            sorted_probabilities = np.sort(inference_results, axis=1)
            confidence_margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]  # margin between top-2
            potential_boundaries = np.where(confidence_margins < self.margin_threshold)[0]
            
            if len(potential_boundaries) > 0:
                boundary_candidates = self._feature_vectors[potential_boundaries]
                clustering = DBSCAN(eps=0.5, min_samples=5)
                cluster_assignments = clustering.fit_predict(boundary_candidates)
                outlier_points = potential_boundaries[cluster_assignments == -1]
                self.boundary_samples.extend([
                    self._candidate_indices[i] for i in outlier_points
                ])
                
                print(f"Identified {len(outlier_points)} boundary samples")
            
        except Exception as e:
            print(f"Boundary detection error: {str(e)}")
            raise

    def rank_by_value(self, feature_vectors: np.ndarray, **config) -> List[int]:
        try:
            standardized_vectors = self._standardize_features(feature_vectors)
            
            inference_results = self.get_inference_results(standardized_vectors)
            
            sample_scores = []
            for i, (prediction, feature) in enumerate(zip(inference_results, standardized_vectors)):
                uncertainty_score = 1 - np.max(prediction)
                
                predicted_class = np.argmax(prediction)
                class_frequency = self.label_distribution[predicted_class]
                balance_score = self.smoothing_factor / (class_frequency + self.smoothing_factor)
                diversity_score = self._compute_diversity_metric(feature)
                combined_score = (self.uncertainty_weight * uncertainty_score + 
                        self.balance_weight * balance_score + 
                        self.diversity_weight * diversity_score)
                sample_scores.append(combined_score)
            
            return np.argsort(-np.array(sample_scores)).tolist()
            
        except Exception as e:
            print(f"Ranking calculation error: {str(e)}")
            raise

    def update_sample_tracking(self, labeled_indices: List[int], assigned_labels: List[int]) -> None:
        try:
            for label in assigned_labels:
                self.label_distribution[label] += 1
            
            distribution_median = np.median(list(self.label_distribution.values()))
            underrepresented_samples = [
                idx for idx, label in zip(labeled_indices, assigned_labels)
                if self.label_distribution[label] < distribution_median
            ]
            
            self.rare_class_samples.extend(underrepresented_samples)
            self.boundary_samples = deque(
                [x for x in self.boundary_samples if x not in labeled_indices],
                maxlen=self.buffer_capacity
            )
            
        except Exception as e:
            print(f"Sample tracking update error: {str(e)}")
            raise

    def get_next_samples(self, batch_size: int) -> List[int]:
        try:
            if len(self._priority_queue) < batch_size:
                print(f"Warning: Only {len(self._priority_queue)} samples available")
                batch_size = len(self._priority_queue)
            
            selected_samples = self._priority_queue[:batch_size]
            self._priority_queue = self._priority_queue[batch_size:]
            
            return selected_samples
            
        except Exception as e:
            print(f"Sample selection error: {str(e)}")
            raise

    def get_available_count(self) -> int:
        return len(self._priority_queue) if self._priority_queue is not None else 0


def create_sampling_strategy(strategy_name: str, model_obj, tokenizer_obj=None, **config) -> AbstractSamplingStrategy:
    available_strategies = {
        'adaptive': AdaptiveSamplingStrategy,
    }
    
    if strategy_name not in available_strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available options: {list(available_strategies.keys())}")
    
    try:
        if strategy_name == 'adaptive':
            return available_strategies[strategy_name](model_obj, **config)
        else:
            if tokenizer_obj is None:
                raise ValueError(f"Strategy {strategy_name} requires a tokenizer")
            
            return available_strategies[strategy_name](model_obj, tokenizer_obj, **config)
        
    except Exception as e:
        print(f"Strategy creation error ({strategy_name}): {str(e)}")
        raise