import torch
import numpy as np
import gc, psutil, os
import torch.nn.functional as F
from collections import deque, Counter, defaultdict
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import threading
import torch.multiprocessing as mp
import time

class CustomDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            str(text),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }

class BaseStrategy(ABC):
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self._unlabeled_indices = None
        self._remaining_indices = None

    def get_predictions(self, data: List[str]) -> np.ndarray:
        self.model.eval()
        predictions = []
        
        dataset = CustomDataset(data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.softmax(outputs.logits, dim=-1)
                predictions.extend(probs.cpu().numpy())
                
        return np.array(predictions)

    @abstractmethod
    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        """Compute value ranking for unlabeled samples"""
        pass

    def initialize_sampling(self, unlabeled_data: List[str], unlabeled_indices: List[int], **kwargs) -> None:
        """Initialize the sampling process"""
        print("Initializing sampling process...")
        self._unlabeled_indices = unlabeled_indices.copy()
        value_ranking = self.compute_value_ranking(unlabeled_data, **kwargs)
        self._remaining_indices = [self._unlabeled_indices[i] for i in value_ranking]
        print(f"Sampling initialized with {len(self._remaining_indices)} samples")

    def select_next_batch(self, query_size: int) -> List[int]:
        """Select the indices of the next batch of samples to label"""
        if len(self._remaining_indices) < query_size:
            print("Warning: Not enough samples remaining")
            query_size = len(self._remaining_indices)

        selected_indices = self._remaining_indices[:query_size]
        self._remaining_indices = self._remaining_indices[query_size:]
        return selected_indices

    def get_remaining_samples_count(self) -> int:
        """Get the number of remaining unlabeled samples"""
        return len(self._remaining_indices) if self._remaining_indices is not None else 0

class FeatureBasedStrategy(ABC):
    def __init__(self, model, **kwargs):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self._unlabeled_indices = None
        self._remaining_indices = None

    def get_predictions(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        predictions = []
        
        # Convert features to tensors and process in batches
        batch_size = 32
        n_samples = len(features)
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_features = torch.FloatTensor(
                    features[i:i + batch_size]
                ).to(self.device)
                
                outputs = self.model(batch_features)
                probs = torch.softmax(outputs.logits, dim=-1)
                predictions.extend(probs.cpu().numpy())
                
        return np.array(predictions)

    @abstractmethod
    def compute_value_ranking(self, features: np.ndarray, **kwargs) -> List[int]:
        """Compute value ranking for unlabeled samples"""
        pass

    def initialize_sampling(self, features: np.ndarray, unlabeled_indices: List[int], **kwargs) -> None:
        """Initialize the sampling process"""
        print("Initializing sampling process...")
        self._unlabeled_indices = unlabeled_indices.copy()
        value_ranking = self.compute_value_ranking(features, **kwargs)
        self._remaining_indices = [self._unlabeled_indices[i] for i in value_ranking]
        print(f"Sampling initialized with {len(self._remaining_indices)} samples")

class BoundaryAwareSampling(BaseStrategy):
    def __init__(self, model, tokenizer=None, **kwargs):
        if tokenizer is None:
            class DummyTokenizer:
                def __call__(self, *args, **kwargs):
                    return {'input_ids': torch.tensor([]), 'attention_mask': torch.tensor([])}
            tokenizer = DummyTokenizer()
            
        super().__init__(model, tokenizer, **kwargs)
        
        # Add feature projection layer to convert 89-dim to 128-dim
        self.feature_projector = torch.nn.Sequential(
            torch.nn.Linear(89, 128),  # Project from 89-dim to 128-dim
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)  # Add dropout to improve robustness
        ).to(self.device)

        # Add new classification head
        self.classifier_head = torch.nn.Linear(128, 12).to(self.device)  # Assuming 12 classes

        # Other initialization parameters
        self.buffer_size = kwargs.get('buffer_size', 1000)
        self.memory_size = kwargs.get('memory_size', 500)
        self.tau = kwargs.get('tau', 0.1)
        self.eps = kwargs.get('eps', 0.01)
        self.alpha = kwargs.get('alpha', 0.4)
        self.beta = kwargs.get('beta', 0.3)
        self.gamma = kwargs.get('gamma', 0.3)
        self.temperature = kwargs.get('temperature', 1.0)
        self.batch_size = kwargs.get('batch_size', 32)
        
        self.boundary_buffer = deque(maxlen=self.buffer_size)
        self.minority_memory = deque(maxlen=self.memory_size)
        self.scaler = StandardScaler()
        self.nn_searcher = None
        self.class_counts = Counter()

    def _preprocess_features(self, features):
        """Preprocess feature data, ensuring correct format"""
        if isinstance(features, np.ndarray) and features.dtype == np.float32:
            return features
            
        processed_features = []
        for feature in features:
            if isinstance(feature, str):
                # Process string-type features
                try:
                    # Remove extra characters and split the string
                    feature = feature.replace('\n', ' ').strip('[]')
                    feature_values = np.fromstring(feature, sep=' ')
                    processed_features.append(feature_values)
                except Exception as e:
                    print(f"Warning: Error processing feature: {str(e)}")
                    # If processing fails, try alternative parsing method
                    try:
                        values = [float(x) for x in feature.split() if x.strip()]
                        processed_features.append(np.array(values))
                    except:
                        raise ValueError(f"Unable to parse feature: {feature[:100]}...")
            else:
                # If already numeric type, add directly
                processed_features.append(feature)
        
        return np.array(processed_features, dtype=np.float32)

    def get_predictions(self, features: np.ndarray) -> np.ndarray:
        """Get model predictions"""
        try:
            processed_features = self._preprocess_features(features)
            
            self.model.eval()
            self.feature_projector.eval()
            self.classifier_head.eval()
            predictions = []
            
            for i in range(0, len(processed_features), self.batch_size):
                batch = processed_features[i:i + self.batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                with torch.no_grad():
                    # Project feature dimensions
                    projected_features = self.feature_projector(batch_tensor)
                    # Use new classification head
                    logits = self.classifier_head(projected_features)
                    probs = torch.softmax(logits, dim=-1)
                    predictions.extend(probs.cpu().numpy())
            
            return np.array(predictions)
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            print(f"Model type: {type(self.model)}")
            print(f"Feature shape: {processed_features.shape}")
            print(f"Batch tensor shape: {batch_tensor.shape if 'batch_tensor' in locals() else 'Not created'}")
            if 'projected_features' in locals():
                print(f"Projected features shape: {projected_features.shape}")
            raise

    def _calculate_diversity_score(self, feature: np.ndarray) -> float:
        """Compute diversity score"""
        try:
            if not self.minority_memory:
                return 1.0
                
            # Get features from memory bank
            memory_features = self._features[list(self.minority_memory)]

            # Compute distances to memory bank samples
            distances = np.linalg.norm(memory_features - feature, axis=1)
            return float(np.min(distances))
            
        except Exception as e:
            print(f"Error calculating diversity score: {str(e)}")
            return 1.0

    def initialize_sampling(self, features: np.ndarray, unlabeled_indices: List[int], **kwargs):
        """Initialize the sampling process"""
        print("Initializing boundary-aware sampling...")
        
        try:
            processed_features = self._preprocess_features(features)
            print(f"Processed features shape: {processed_features.shape}")
            
            normalized_features = self.scaler.fit_transform(processed_features)
            print(f"Normalized features shape: {normalized_features.shape}")
            
            self.nn_searcher = NearestNeighbors(
                n_neighbors=min(50, len(unlabeled_indices)),
                algorithm='ball_tree'
            ).fit(normalized_features)
            
            self._features = normalized_features
            self._unlabeled_indices = unlabeled_indices
            
            print("Detecting boundary samples...")
            self._detect_boundary_samples(processed_features)
            
            print("Computing value ranking...")
            value_ranking = self.compute_value_ranking(processed_features)
            self._remaining_indices = [self._unlabeled_indices[i] for i in value_ranking]
            
            print(f"Sampling initialized with {len(unlabeled_indices)} samples")
            
        except Exception as e:
            print(f"Error during sampling initialization: {str(e)}")
            print(f"Feature shape: {features.shape if features is not None else 'None'}")
            raise

    def _detect_boundary_samples(self, features: np.ndarray) -> None:
        """Detect boundary samples using mixed features"""
        try:
            predictions = self.get_predictions(features)

            # Compute entropy of prediction probabilities
            probs = np.sort(predictions, axis=1)
            confidence_gaps = probs[:, -1] - probs[:, -2]  # Top-2 probability gap

            # Identify low-confidence samples
            boundary_candidates = np.where(confidence_gaps < self.tau)[0]

            if len(boundary_candidates) > 0:
                # Get candidate sample features
                candidate_features = self._features[boundary_candidates]

                # Cluster analysis
                clusterer = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = clusterer.fit_predict(candidate_features)

                # Find outliers as boundary samples
                boundary_points = boundary_candidates[cluster_labels == -1]

                # Update boundary buffer
                self.boundary_buffer.extend([
                    self._unlabeled_indices[i] for i in boundary_points
                ])
                
                print(f"Detected {len(boundary_points)} boundary samples")
            
        except Exception as e:
            print(f"Error during boundary detection: {str(e)}")
            raise

    def compute_value_ranking(self, features: np.ndarray, **kwargs) -> List[int]:
        """Compute sample value ranking"""
        try:
            # Preprocess features
            processed_features = self._preprocess_features(features)

            # Get prediction probabilities
            predictions = self.get_predictions(processed_features)
            
            scores = []
            for i, (pred, feature) in enumerate(zip(predictions, processed_features)):
                # Uncertainty score
                uncertainty_score = 1 - np.max(pred)

                # Class weight score
                pred_class = np.argmax(pred)
                class_count = self.class_counts[pred_class]
                class_weight = self.temperature / (class_count + self.temperature)

                # Diversity score
                diversity_score = self._calculate_diversity_score(feature)

                # Combined score
                score = (self.alpha * uncertainty_score + 
                        self.beta * class_weight + 
                        self.gamma * diversity_score)
                scores.append(score)
            
            return np.argsort(-np.array(scores)).tolist()
            
        except Exception as e:
            print(f"Error during value ranking computation: {str(e)}")
            raise

    def update_memory(self, labeled_indices: List[int], labels: List[int]) -> None:
        """Update memory bank"""
        try:
            # Update class statistics
            for label in labels:
                self.class_counts[label] += 1

            # Find minority class samples
            median_count = np.median(list(self.class_counts.values()))
            minority_samples = [
                idx for idx, label in zip(labeled_indices, labels)
                if self.class_counts[label] < median_count
            ]

            # Update minority class memory bank
            self.minority_memory.extend(minority_samples)

            # Remove labeled samples from boundary buffer
            self.boundary_buffer = deque(
                [x for x in self.boundary_buffer if x not in labeled_indices],
                maxlen=self.buffer_size
            )
            
        except Exception as e:
            print(f"Error updating memory: {str(e)}")
            raise

    def select_next_batch(self, query_size: int) -> List[int]:
        """Select the next batch of samples to label"""
        try:
            if len(self._remaining_indices) < query_size:
                print(f"Warning: Only {len(self._remaining_indices)} samples remaining")
                query_size = len(self._remaining_indices)
            
            selected_indices = self._remaining_indices[:query_size]
            self._remaining_indices = self._remaining_indices[query_size:]
            
            return selected_indices
            
        except Exception as e:
            print(f"Error selecting next batch: {str(e)}")
            raise

    def get_remaining_samples_count(self) -> int:
        """Get the number of remaining unlabeled samples"""
        return len(self._remaining_indices) if self._remaining_indices is not None else 0

class ImbalancedAwareSampling(BaseStrategy):
    """
    Active learning strategy designed for imbalanced datasets
    """

    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)

        self.source_memory = set()
        self.class_counts = {}
        self.labeled_features = np.array([])
        self.labeled_classes = np.array([], dtype=np.int64)

        self.balance_threshold = kwargs.get('balance_threshold', 0.3)
        self.is_balancing = False
        self.batch_size = kwargs.get('batch_size', 32)

        self.predictions_cache = None

    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        """
        Compute sample value ranking based on the pre-trained model's predictions
        """
        if self.predictions_cache is None or len(self.predictions_cache) != len(unlabeled_data):
            self.predictions_cache = self.get_predictions(unlabeled_data)

        pred_classes = np.argmax(self.predictions_cache, axis=1)
        pred_probs = np.max(self.predictions_cache, axis=1)
        
        # Sort by prediction probability and class
        indices = []
        unique_classes = np.unique(pred_classes)
        
        for cls in unique_classes:
            class_indices = np.where(pred_classes == cls)[0]
            sorted_indices = class_indices[np.argsort(-pred_probs[class_indices])]
            for idx in sorted_indices:
                if cls not in self.source_memory:
                    indices.append(int(idx))
                    self.source_memory.add(cls)
                    if len(indices) >= len(unlabeled_data):
                        break
            if len(indices) >= len(unlabeled_data):
                break
        return indices

    def update_labeled_info(self, features: np.ndarray, labels: List[int]) -> None:
        """Update labeled sample information"""
        self.labeled_features = np.concatenate([self.labeled_features, features]) if self.labeled_features.size else features
        self.labeled_classes = np.concatenate([self.labeled_classes, labels])
        for label in labels:
            self.class_counts[label] = self.class_counts.get(label, 0) + 1

    def _check_balance_switch(self, m: int, b: int) -> bool:
        """
        Check whether to switch to balancing mode
        """
        if not self.class_counts:
            return False

        mean_samples = np.mean(list(self.class_counts.values()))
        under_represented = [c for c, count in self.class_counts.items() if count <= mean_samples]
        if not under_represented:
            return False

        samples_needed = len(under_represented) * (mean_samples - np.mean([self.class_counts[c] for c in under_represented]))
        return b - m <= samples_needed

    def _calculate_distance_to_class(self, feature: np.ndarray, class_idx: int) -> float:
        """Compute distance from feature to a specified class"""
        class_features = self.labeled_features[self.labeled_classes == class_idx]
        if class_features.size == 0:
            return float('inf')
        return np.min(np.linalg.norm(class_features - feature, axis=1))

    def compute_balancing_value(self, features: np.ndarray) -> List[int]:
        """
        Compute balancing value ranking, prioritizing samples from under-represented classes
        """
        if not self.class_counts or self.labeled_features.size == 0:
            return np.random.permutation(len(features)).tolist()

        mean_samples = np.mean(list(self.class_counts.values()))
        under_represented = [c for c, count in self.class_counts.items() if count <= mean_samples]
        if not under_represented:
            return np.random.permutation(len(features)).tolist()

        target_class = sorted(under_represented, key=lambda c: self.class_counts[c])[0]
        over_represented = [c for c, count in self.class_counts.items() if count > mean_samples]

        balance_scores = []
        for feature in features:
            target_dist = self._calculate_distance_to_class(feature, target_class)
            over_dist = min([self._calculate_distance_to_class(feature, over_class) for over_class in over_represented], default=float('inf'))
            score = over_dist / target_dist if target_dist else float('inf')
            balance_scores.append(score)

        return np.argsort(-np.array(balance_scores)).tolist()

    def initialize_sampling(self, unlabeled_data: List[str], unlabeled_indices: List[int], **kwargs) -> None:
        """Initialize the sampling process"""
        print("Initializing imbalanced-aware sampling...")
        self._unlabeled_indices = unlabeled_indices.copy()
        self.is_balancing = False
        value_ranking = self.compute_value_ranking(unlabeled_data, **kwargs)
        self._remaining_indices = [self._unlabeled_indices[i] for i in value_ranking]
        print(f"Sampling initialized with {len(self._remaining_indices)} samples")

    def select_next_batch(self, query_size: int) -> List[int]:
        """Select the next batch of samples to label, with support for switching to balancing mode"""
        if len(self._remaining_indices) < query_size:
            query_size = len(self._remaining_indices)

        labeled_count = sum(self.class_counts.values()) if self.class_counts else 0
        if not self.is_balancing and self._check_balance_switch(labeled_count, labeled_count + query_size):
            print("Switching to balancing mode...")
            self.is_balancing = True
            features = [self._remaining_indices[i] for i in range(len(self._remaining_indices))]
            balance_ranking = self.compute_balancing_value(features)
            self._remaining_indices = [self._remaining_indices[i] for i in balance_ranking]

        selected_indices = self._remaining_indices[:query_size]
        self._remaining_indices = self._remaining_indices[query_size:]
        return selected_indices


# class ImbalancedAwareSampling(BaseStrategy):
#     """
#     Active learning strategy designed for imbalanced datasets

#     Based on paper: 'Active Learning for Imbalanced Datasets' (Aggarwal et al., WACV 2020)

#     This strategy combines two main innovations:
#     1. Modified acquisition function for diversified sample selection based on source-domain pre-trained model predictions
#     2. Introduction of a balancing step to reduce labeled dataset imbalance
#     """
    
#     def __init__(self, model, tokenizer, **kwargs):
#         super().__init__(model, tokenizer, **kwargs)
        
#         # Source class memory for diversified selection
#         self.source_memory = set()

#         # Labeled class counts and means
#         self.class_counts = {}
#         self.labeled_features = np.array([])
#         self.labeled_classes = np.array([], dtype=np.int64)
        
#         # Balancing parameters
#         self.balance_threshold = kwargs.get('balance_threshold', 0.3)
#         self.is_balancing = False
#         self.batch_size = kwargs.get('batch_size', 32)
        
#         # Cache last prediction results to avoid redundant computation
#         self.predictions_cache = None
    
#     def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
#         """
#         Compute sample value ranking, implementing the modified acquisition function from the paper
#
#         Uses deterministic predictions from the pre-trained model with diversified selection
#         """
#         print("Computing value ranking...")
        
#         # Get model predictions
#         if self.predictions_cache is None or len(self.predictions_cache) != len(unlabeled_data):
#             predictions = self.get_predictions(unlabeled_data)
#             self.predictions_cache = predictions
#         else:
#             predictions = self.predictions_cache
        
#         # Get predicted classes and highest probabilities
#         pred_classes = np.argmax(predictions, axis=1)
#         pred_probs = np.max(predictions, axis=1)
        
#         # Group by predicted class
#         unique_classes = np.unique(pred_classes)
#         selected_indices = []
        
#         # Diversified sampling: select samples from each class in order of decreasing certainty
#         available_classes = set(unique_classes)
        
#         while available_classes and len(selected_indices) < len(unlabeled_data):
#             for cls in list(available_classes):
#                 # Find all samples of the current class
#                 class_indices = np.where(pred_classes == cls)[0]
#                 if len(class_indices) == 0:
#                     available_classes.remove(cls)
#                     continue
                
#                 # Sort by prediction probability in descending order
#                 sorted_indices = class_indices[np.argsort(-pred_probs[class_indices])]
                
#                 # Select sample with highest probability
#                 for idx in sorted_indices:
#                     if cls not in self.source_memory:
#                         selected_indices.append(int(idx))
#                         self.source_memory.add(cls)
#                         break
            
#             # If all classes processed but not enough samples, reset source memory
#             if len(selected_indices) < len(unlabeled_data) and not available_classes:
#                 available_classes = set(unique_classes)
#                 self.source_memory.clear()
        
#         return selected_indices
    
#     def update_labeled_info(self, features: np.ndarray, labels: List[int]) -> None:
#         """Update labeled sample information"""
#         # Update labeled sample features and classes
#         if self.labeled_features.size == 0:
#             self.labeled_features = np.array(features)
#             self.labeled_classes = np.array(labels, dtype=np.int64)
#         else:
#             self.labeled_features = np.concatenate([self.labeled_features, features])
#             self.labeled_classes = np.concatenate([self.labeled_classes, labels])
        
#         # Update class counts
#         for label in labels:
#             self.class_counts[label] = self.class_counts.get(label, 0) + 1
    
#     def _check_balance_switch(self, m: int, b: int) -> bool:
#         """
#         Check whether to switch to balancing mode, based on the rules from the paper
#
#         Args:
#             m: Number of currently labeled samples
#             b: Total labeling budget
#
#         Returns:
#             Whether to switch to balancing mode
#         """
#         if not self.class_counts:
#             return False
        
#         # Compute class mean
#         mean_samples = np.mean(list(self.class_counts.values()))
        
#         # Identify over-represented and under-represented classes
#         over_represented = [c for c, count in self.class_counts.items() if count > mean_samples]
#         under_represented = [c for c, count in self.class_counts.items() if count <= mean_samples]
        
#         if not under_represented:
#             return False
        
#         # Compute mean sample count for under-represented and over-represented classes
#         mean_under = np.mean([self.class_counts[c] for c in under_represented])
#         mean_over = np.mean([self.class_counts[c] for c in over_represented])
        
#         # Check if remaining budget is sufficient to balance the dataset
#         samples_needed = len(under_represented) * (mean_over - mean_under)
#         remaining_budget = b - m
        
#         return remaining_budget <= samples_needed
    
#     def _calculate_distance_to_class(self, feature: np.ndarray, class_idx: int) -> float:
#         """Compute distance from feature to a specified class"""
#         class_mask = (self.labeled_classes == class_idx)
#         if not np.any(class_mask):
#             return float('inf')
        
#         class_features = self.labeled_features[class_mask]
#         distances = np.linalg.norm(class_features - feature, axis=1)
#         return float(np.min(distances))
    
#     def compute_balancing_value(self, features: np.ndarray) -> List[int]:
#         """
#         Implement the balancing step from the paper, computing balancing value ranking
#
#         Prioritize samples from under-represented classes while avoiding over-represented classes
#         """
#         if not self.class_counts or self.labeled_features.size == 0:
#             # If no labeled data, return random ordering
#             return np.random.permutation(len(features)).tolist()
        
#         # Compute class mean
#         mean_samples = np.mean(list(self.class_counts.values()))
        
#         # Find under-represented classes
#         under_represented = [(c, count) for c, count in self.class_counts.items() 
#                             if count <= mean_samples]
#         if not under_represented:
#             # If no under-represented classes, return random ordering
#             return np.random.permutation(len(features)).tolist()
        
#         # Sort under-represented classes by sample count in ascending order
#         under_represented.sort(key=lambda x: x[1])
#         target_class = under_represented[0][0]  # Class with fewest samples
        
#         # Compute over-represented classes
#         over_represented = [c for c, count in self.class_counts.items() 
#                            if count > mean_samples]
        
#         # Compute balancing score for each sample
#         balance_scores = []
#         for i, feature in enumerate(features):
#             # Distance to target class
#             target_dist = self._calculate_distance_to_class(feature, target_class)
            
#             # Minimum distance to over-represented classes
#             over_dist = float('inf')
#             for over_class in over_represented:
#                 dist = self._calculate_distance_to_class(feature, over_class)
#                 over_dist = min(over_dist, dist)
            
#             # Balancing score = distance to over-represented class / distance to target class
#             # Higher score means more likely to belong to the target class
#             if target_dist == 0:
#                 score = float('inf')
#             elif over_dist == float('inf'):
#                 score = 1.0 / target_dist
#             else:
#                 score = over_dist / target_dist
                
#             balance_scores.append(score)
        
#         # Sort by balancing score in descending order
#         return np.argsort(-np.array(balance_scores)).tolist()
    
#     def initialize_sampling(self, unlabeled_data: List[str], unlabeled_indices: List[int], **kwargs) -> None:
#         """Initialize the sampling process"""
#         print("Initializing imbalanced-aware sampling...")
#         self._unlabeled_indices = unlabeled_indices.copy()
        
#         # Initial phase uses modified acquisition function
#         self.is_balancing = False
#         value_ranking = self.compute_value_ranking(unlabeled_data, **kwargs)
#         self._remaining_indices = [self._unlabeled_indices[i] for i in value_ranking]
        
#         print(f"Sampling initialized with {len(self._remaining_indices)} samples")
    
#     def select_next_batch(self, query_size: int) -> List[int]:
#         """
#         Select the next batch of samples to label, with support for switching to balancing mode
#
#         Args:
#             query_size: Number of samples to select
#
#         Returns:
#             List of selected sample indices
#         """
#         if len(self._remaining_indices) < query_size:
#             print(f"Warning: Only {len(self._remaining_indices)} samples remaining")
#             query_size = len(self._remaining_indices)
        
#         # Check whether to switch to balancing mode
#         labeled_count = len(self.class_counts) > 0 and sum(self.class_counts.values()) or 0
#         if not self.is_balancing and self._check_balance_switch(labeled_count, labeled_count + query_size):
#             print("Switching to balancing mode...")
#             self.is_balancing = True
            
#             # Recompute sample ranking using balancing function
#             features = [self._remaining_indices[i] for i in range(len(self._remaining_indices))]
#             balance_ranking = self.compute_balancing_value(features)
#             self._remaining_indices = [self._remaining_indices[i] for i in balance_ranking]
        
#         # Select samples
#         selected_indices = self._remaining_indices[:query_size]
#         self._remaining_indices = self._remaining_indices[query_size:]
        
#         return selected_indices

class BATLSampling(BaseStrategy):
    """
    Batch acquisition strategy based on triplet loss (Batch Acquisition with Triplet Loss)

    Paper: Active Learning on Pre-trained Language Model with Task-Independent Triplet Loss

    This strategy leverages pre-trained model knowledge and task-related features, using triplet loss
    to ensure the selected batch of samples is both informative and diverse.
    """
    
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        # Get hyperparameters from kwargs
        self.margin = kwargs.get('margin', 0.3)
        self.lambda_scale = kwargs.get('lambda_scale', 1.0)
        self.batch_size = kwargs.get('batch_size', 32)
        self.embedding_dim = model.config.hidden_size  # Get model hidden size

        # Create projection layer for concatenating sentence representations and task-related features
        self.projection_layer = torch.nn.Sequential(
            torch.nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU()
        ).to(self.device)
        
        # Initialize cache
        self.embeddings_cache = None
        self.predictions_cache = None
    
    def get_sentence_representation(self, texts):
        """Get sentence representations from the pre-trained language model"""
        self.model.eval()
        sentence_embeddings = []
        
        dataset = CustomDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting sentence representations"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get hidden states
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Use [CLS] token or average pooling from the last layer as sentence representation
                # Here we use the [CLS] representation from the last layer
                last_hidden_states = outputs.hidden_states[-1]
                cls_embeddings = last_hidden_states[:, 0, :]  # [batch_size, hidden_size]
                
                sentence_embeddings.extend(cls_embeddings.cpu().numpy())
                
                # Clear cache
                del last_hidden_states, cls_embeddings, outputs
                torch.cuda.empty_cache()
        
        return np.array(sentence_embeddings)
    
    def get_task_features(self, texts):
        """Get task-related features"""
        self.model.eval()
        task_features = []
        
        dataset = CustomDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting task features"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get features before the classification layer
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get features before the final classifier input
                if hasattr(outputs, 'hidden_states'):
                    # If model output includes hidden states
                    task_feature = outputs.hidden_states[-1][:, 0, :]  # Use [CLS]
                else:
                    # Otherwise, get features from the layer before logits
                    # This depends on the specific model architecture and may need adjustment
                    task_feature = self.model.classifier.in_features
                
                task_features.extend(task_feature.cpu().numpy())
                
                # Clear cache
                del task_feature, outputs
                torch.cuda.empty_cache()
        
        return np.array(task_features)
    
    def compute_target_loss(self, predictions):
        """Compute target task loss"""
        # Use the max prediction probability as certainty; its inverse as the uncertainty score
        return 1 - np.max(predictions, axis=1)
    
    def select_triplets(self, embeddings, predictions):
        """Select samples for triplet loss"""
        n_samples = len(embeddings)
        triplets = []
        pred_classes = np.argmax(predictions, axis=1)
        
        # Find hard positives and semi-hard negatives for each sample
        for i in range(n_samples):
            anchor = embeddings[i]
            anchor_class = pred_classes[i]
            
            # Find samples of the same class as the anchor
            same_class_indices = np.where(pred_classes == anchor_class)[0]
            same_class_indices = same_class_indices[same_class_indices != i]  # Exclude self

            # Find samples of different classes from the anchor
            diff_class_indices = np.where(pred_classes != anchor_class)[0]
            
            if len(same_class_indices) == 0 or len(diff_class_indices) == 0:
                continue
            
            # Compute distances
            pos_distances = np.linalg.norm(embeddings[same_class_indices] - anchor, axis=1)
            neg_distances = np.linalg.norm(embeddings[diff_class_indices] - anchor, axis=1)
            
            # Select hardest positive sample (farthest same-class sample)
            hard_pos_idx = same_class_indices[np.argmax(pos_distances)]

            # Select semi-hard negative sample
            # Semi-hard negatives are those farther than the positive but within the margin
            semi_hard_neg_mask = (neg_distances > pos_distances.max()) & (neg_distances < pos_distances.max() + self.margin)
            
            if np.any(semi_hard_neg_mask):
                # Semi-hard negatives found
                semi_hard_neg_indices = diff_class_indices[semi_hard_neg_mask]
                hard_neg_idx = semi_hard_neg_indices[np.argmin(neg_distances[semi_hard_neg_mask])]
            else:
                # No semi-hard negatives, select the nearest negative
                hard_neg_idx = diff_class_indices[np.argmin(neg_distances)]
            
            # Add triplet (anchor, positive, negative)
            triplets.append((i, hard_pos_idx, hard_neg_idx))
        
        return triplets
    
    def compute_triplet_loss(self, embeddings, triplets):
        """Compute triplet loss"""
        if not triplets:
            return np.zeros(len(embeddings))
        
        # Initialize loss to 0 for each sample
        sample_losses = np.zeros(len(embeddings))

        # Compute loss for each triplet
        for anchor_idx, pos_idx, neg_idx in triplets:
            anchor = embeddings[anchor_idx]
            positive = embeddings[pos_idx]
            negative = embeddings[neg_idx]
            
            # Compute Euclidean distance
            pos_dist = np.linalg.norm(anchor - positive)
            neg_dist = np.linalg.norm(anchor - negative)
            
            # Compute triplet loss
            loss = max(0, self.margin + pos_dist - neg_dist)
            
            # Accumulate loss to corresponding samples
            sample_losses[anchor_idx] += loss
        
        return sample_losses
    
    def compute_value_ranking(self, unlabeled_data, **kwargs):
        """Compute sample value ranking"""
        print("Computing sample value ranking using BATL strategy...")

        # 1. Get sentence representations from the pre-trained language model
        sentence_embeddings = self.get_sentence_representation(unlabeled_data)

        # 2. Get model prediction probability distributions
        predictions = self.get_predictions(unlabeled_data)

        # 3. Get task-related features (optional)
        # Note: This step may need adjustment depending on model architecture
        # Here we simplify by using sentence representations as task-related features

        # 4. Concatenate features
        combined_embeddings = sentence_embeddings

        # 5. Compute target task loss
        target_losses = self.compute_target_loss(predictions)

        # 6. Select triplets
        triplets = self.select_triplets(combined_embeddings, predictions)

        # 7. Compute triplet loss
        triplet_losses = self.compute_triplet_loss(combined_embeddings, triplets)

        # 8. Compute final loss
        final_losses = target_losses + self.lambda_scale * triplet_losses

        # Return indices sorted by loss in descending order
        return np.argsort(-final_losses).tolist()
    
    def initialize_sampling(self, unlabeled_data, unlabeled_indices, **kwargs):
        """Initialize the sampling process"""
        print("Initializing batch acquisition with triplet loss sampling...")
        self._unlabeled_indices = unlabeled_indices.copy()

        # Compute and cache feature representations
        if self.embeddings_cache is None:
            self.embeddings_cache = self.get_sentence_representation(unlabeled_data)
        
        # Compute and cache prediction results
        if self.predictions_cache is None:
            self.predictions_cache = self.get_predictions(unlabeled_data)
        
        # Compute value ranking
        value_ranking = self.compute_value_ranking(unlabeled_data, **kwargs)
        self._remaining_indices = [self._unlabeled_indices[i] for i in value_ranking]
        
        print(f"Sampling initialized with {len(self._remaining_indices)} samples")


class QBCSampling(BaseStrategy):
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.n_committees = kwargs.get('n_committees', 3)
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.batch_size = kwargs.get('batch_size', 32)
        self.model_path = kwargs.get('model_path')  # Get model path from kwargs
        
        if not self.model_path:
            raise ValueError("model_path must be provided for QBCSampling")
        
        print(f"Initializing QBC with {self.n_committees} committees...")
        print(f"Using model path: {self.model_path}")
        print(f"Dropout rate: {self.dropout_rate}")
        print(f"Batch size: {self.batch_size}")
        
        # Create committee members
        self.committees = []
        for i in range(self.n_committees):
            try:
                print(f"\nCreating committee member {i+1}/{self.n_committees}...")
                
                # Create a new base model instance
                committee_base = AlbertForSequenceClassification.from_pretrained(
                    self.model_path,
                    num_labels=model.config.num_labels,
                    local_files_only=True
                )
                print(f"Base model created for committee {i+1}")
                
                # Create new LoRA configuration
                lora_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=model.peft_config['default'].r,
                    lora_alpha=model.peft_config['default'].lora_alpha,
                    lora_dropout=self.dropout_rate,  # Use different dropout
                    target_modules=model.peft_config['default'].target_modules,
                    bias=model.peft_config['default'].bias,
                    modules_to_save=model.peft_config['default'].modules_to_save
                )
                print(f"LoRA config created for committee {i+1}")
                
                # Apply LoRA
                committee_model = get_peft_model(committee_base, lora_config)
                print(f"LoRA applied to committee {i+1}")
                
                # Get and load PEFT model weights
                peft_state_dict = {}
                for key, value in model.state_dict().items():
                    if 'lora' in key or 'modules_to_save' in key:
                        peft_state_dict[key] = value.clone()
                
                # Load PEFT weights
                missing_keys, unexpected_keys = committee_model.load_state_dict(
                    peft_state_dict, strict=False
                )
                print(f"Weights loaded for committee {i+1}")
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
                
                # Set dropout
                for name, module in committee_model.named_modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = self.dropout_rate
                print(f"Dropout set to {self.dropout_rate} for committee {i+1}")
                
                committee_model.train()  # Keep in training mode to enable dropout
                self.committees.append(committee_model.to(self.device))
                
                print(f"Committee member {i+1} created successfully")
                
                # Clean up memory
                del committee_base
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Error creating committee member {i+1}: {str(e)}")
                print(f"Error details: {e.__class__.__name__}")
                raise

    def _get_committee_predictions(self, texts: List[str]) -> List[np.ndarray]:
        """Get predictions from all committee members"""
        committee_predictions = []
        
        for i, committee in enumerate(self.committees):
            try:
                predictions = []
                dataset = CustomDataset(texts, self.tokenizer)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=self.batch_size, 
                    shuffle=False,
                    num_workers=0
                )
                
                committee.eval()  # Evaluation mode
                with torch.no_grad():
                    for batch in dataloader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        
                        outputs = committee(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        
                        probs = torch.softmax(outputs.logits, dim=-1)
                        predictions.extend(probs.cpu().numpy())
                
                committee_predictions.append(np.array(predictions))
                print(f"Committee member {i+1}/{len(self.committees)} predictions completed")
                
                # Clean up memory
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in predictions for committee {i+1}: {str(e)}")
                raise
            
        return committee_predictions

    def compute_vote_entropy(self, committee_predictions: List[np.ndarray]) -> np.ndarray:
        """Compute vote entropy"""
        try:
            n_samples = committee_predictions[0].shape[0]
            n_classes = committee_predictions[0].shape[1]
            vote_entropy = np.zeros(n_samples)
            
            # Compute vote entropy for each sample
            for i in range(n_samples):
                # Get predicted class from each committee member
                votes = np.array([np.argmax(pred[i]) for pred in committee_predictions])
                # Count votes for each class
                vote_counts = np.bincount(votes, minlength=n_classes)
                # Compute vote proportions
                vote_probs = vote_counts / len(self.committees)
                # Compute entropy (add small epsilon to avoid log(0))
                entropy = -np.sum(vote_probs * np.log2(vote_probs + 1e-10))
                vote_entropy[i] = entropy
            
            return vote_entropy
            
        except Exception as e:
            print(f"Error computing vote entropy: {str(e)}")
            raise

    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        """Compute sample value ranking"""
        try:
            print("\nStarting QBC value ranking computation...")
            print(f"Processing {len(unlabeled_data)} unlabeled samples")

            # Get committee predictions
            print("Getting committee predictions...")
            committee_predictions = self._get_committee_predictions(unlabeled_data)

            # Compute vote entropy
            print("Computing vote entropy...")
            vote_entropy = self.compute_vote_entropy(committee_predictions)

            # Clear prediction results to free memory
            del committee_predictions
            gc.collect()
            
            print("Value ranking computation completed")
            
            # Return indices sorted by entropy in descending order
            return np.argsort(-vote_entropy).tolist()
            
        except Exception as e:
            print(f"Error in QBC value ranking computation: {str(e)}")
            raise

    def initialize_sampling(self, unlabeled_data: List[str], unlabeled_indices: List[int], **kwargs) -> None:
        """Initialize the sampling process"""
        print("\nInitializing QBC sampling...")
        try:
            self._unlabeled_indices = unlabeled_indices.copy()
            print(f"Computing value ranking for {len(unlabeled_data)} samples...")
            
            value_ranking = self.compute_value_ranking(unlabeled_data, **kwargs)
            self._remaining_indices = [self._unlabeled_indices[i] for i in value_ranking]
            
            print(f"QBC sampling initialized with {len(self._remaining_indices)} samples")
            
        except Exception as e:
            print(f"Error during QBC sampling initialization: {str(e)}")
            raise

    def select_next_batch(self, query_size: int) -> List[int]:
        """Select the next batch of samples to label"""
        try:
            if not self._remaining_indices:
                print("Warning: No remaining samples to select from")
                return []
                
            if len(self._remaining_indices) < query_size:
                print(f"Warning: Only {len(self._remaining_indices)} samples remaining")
                query_size = len(self._remaining_indices)
            
            selected_indices = self._remaining_indices[:query_size]
            self._remaining_indices = self._remaining_indices[query_size:]
            
            print(f"Selected {len(selected_indices)} samples for labeling")
            print(f"Remaining samples: {len(self._remaining_indices)}")
            
            return selected_indices
            
        except Exception as e:
            print(f"Error selecting next batch: {str(e)}")
            raise

    def __del__(self):
        """Clean up resources"""
        try:
            for committee in self.committees:
                del committee
            self.committees = []
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

class DiversitySampling(BaseStrategy):
    def __init__(self, model, tokenizer, sentence_model_path='./base-model/paraphrase-MiniLM-L6-v2', **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.sentence_transformer = SentenceTransformer(sentence_model_path)
        self.batch_size = kwargs.get('batch_size', 10000)  # Set batch processing size
    
    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        print("Computing embeddings...")
        # Get text embeddings
        embeddings = self.sentence_transformer.encode(unlabeled_data, batch_size=self.batch_size, show_progress_bar=True)

        print("Computing diversity scores...")
        n_samples = len(embeddings)
        diversity_scores = np.zeros(n_samples)

        # Compute similarity and diversity scores in batches
        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            batch_embeddings = embeddings[i:end_idx]
            
            # Compute similarity of current batch with all samples
            batch_similarities = cosine_similarity(batch_embeddings, embeddings)

            # Compute average similarity for each sample
            batch_diversity = 1 / (np.mean(batch_similarities, axis=1) + 1e-10)
            diversity_scores[i:end_idx] = batch_diversity
            
            # Free memory
            del batch_similarities
            gc.collect()
            
            if i % 10000 == 0:
                print(f"Processed {i}/{n_samples} samples")
        
        return np.argsort(-diversity_scores).tolist()

class DensitySampling(BaseStrategy):
    def __init__(self, model, tokenizer, sentence_model_path='./base-model/paraphrase-MiniLM-L6-v2', **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.sentence_transformer = SentenceTransformer(sentence_model_path)
        self.eps = kwargs.get('eps', 0.5)
        self.min_samples = kwargs.get('min_samples', 5)
        self.batch_size = kwargs.get('batch_size', 1000)
    
    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        print("Computing embeddings...")
        # Get text embeddings
        embeddings = self.sentence_transformer.encode(
            unlabeled_data, 
            batch_size=self.batch_size,
            show_progress_bar=True
        )
        
        print("Computing density scores...")
        n_samples = len(embeddings)
        density_scores = np.zeros(n_samples)
        
        try:
            # Compute density scores in batches
            for i in range(0, n_samples, self.batch_size):
                if i % 10000 == 0:
                    print(f"Processing batch starting at {i}/{n_samples}")
                
                end_idx = min(i + self.batch_size, n_samples)
                batch_embeddings = embeddings[i:end_idx]
                
                # Compute distances of current batch to all samples
                batch_density_scores = np.zeros(end_idx - i)
                
                # Compute distances to other samples in batches
                for j in range(0, n_samples, self.batch_size):
                    j_end = min(j + self.batch_size, n_samples)
                    other_embeddings = embeddings[j:j_end]
                    
                    # Compute distances of current batch to other samples
                    batch_distances = cdist(batch_embeddings, other_embeddings)
                    
                    # Count samples with distance less than eps
                    batch_density_scores += np.sum(batch_distances < self.eps, axis=1)
                    
                    # Clean up memory
                    del batch_distances
                    gc.collect()
                
                density_scores[i:end_idx] = batch_density_scores
                
                # Clean up memory
                del batch_density_scores
                gc.collect()
                
            print("Density computation completed")
            
            # Normalize density scores
            if np.max(density_scores) > np.min(density_scores):
                density_scores = (density_scores - np.min(density_scores)) / (np.max(density_scores) - np.min(density_scores))
            
            # Return indices sorted by density score in descending order
            return np.argsort(-density_scores).tolist()
            
        except Exception as e:
            print(f"Error during density computation: {str(e)}")
            print(f"Error details: {e.__class__.__name__}: {str(e)}")
            raise

class GraphDensitySampling(BaseStrategy):
    def __init__(self, model, tokenizer, sentence_model_path='./base-model/paraphrase-MiniLM-L6-v2', **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.sentence_transformer = SentenceTransformer(sentence_model_path)
        self.k_neighbors = kwargs.get('k_neighbors', 5)
        self.batch_size = kwargs.get('batch_size', 1000)
    
    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        print("Computing embeddings...")
        embeddings = self.sentence_transformer.encode(
            unlabeled_data,
            batch_size=self.batch_size,
            show_progress_bar=True
        )
        
        print("Computing graph density scores...")
        n_samples = len(embeddings)
        graph_density_scores = np.zeros(n_samples)
        
        try:
            # Compute graph density scores in batches
            for i in range(0, n_samples, self.batch_size):
                if i % 1000 == 0:
                    print(f"Processing batch {i}/{n_samples}")
                
                end_idx = min(i + self.batch_size, n_samples)
                batch_embeddings = embeddings[i:end_idx]
                batch_scores = np.zeros(end_idx - i)
                
                # Compute similarity and nearest neighbors in batches
                for j in range(0, n_samples, self.batch_size):
                    j_end = min(j + self.batch_size, n_samples)
                    other_embeddings = embeddings[j:j_end]
                    
                    # Compute similarity of current batch with other samples
                    batch_similarities = self._compute_cosine_similarity(
                        batch_embeddings, 
                        other_embeddings
                    )
                    
                    # Update top-k nearest neighbors for each sample
                    for idx in range(len(batch_embeddings)):
                        # Get similarity of current sample with other samples
                        similarities = batch_similarities[idx]
                        
                        # Find top-k most similar samples
                        top_k_indices = np.argpartition(similarities, -min(self.k_neighbors, len(similarities)))[-self.k_neighbors:]
                        
                        # Compute average similarity
                        batch_scores[idx] += np.mean(similarities[top_k_indices])
                    
                    # Clean up memory
                    del batch_similarities
                    gc.collect()
                
                # Divide batch scores by the number of processed batches
                batch_scores /= ((n_samples + self.batch_size - 1) // self.batch_size)
                graph_density_scores[i:end_idx] = batch_scores
                
                # Clean up memory
                del batch_scores
                gc.collect()
            
            print("Graph density computation completed")
            
            # Normalize scores
            if np.max(graph_density_scores) > np.min(graph_density_scores):
                graph_density_scores = (graph_density_scores - np.min(graph_density_scores)) / (
                    np.max(graph_density_scores) - np.min(graph_density_scores)
                )
            
            return np.argsort(-graph_density_scores).tolist()
            
        except Exception as e:
            print(f"Error during graph density computation: {str(e)}")
            raise
    
    def _compute_cosine_similarity(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of vectors

        Args:
            A: Array of shape (n_samples_A, n_features)
            B: Array of shape (n_samples_B, n_features)

        Returns:
            Similarity matrix of shape (n_samples_A, n_samples_B)
        """
        # Compute L2 norms
        norm_A = np.linalg.norm(A, axis=1)
        norm_B = np.linalg.norm(B, axis=1)
        
        # Normalize vectors
        A_normalized = A / norm_A[:, np.newaxis]
        B_normalized = B / norm_B[:, np.newaxis]
        
        # Compute dot product
        similarity = np.dot(A_normalized, B_normalized.T)
        
        return similarity
    
    def initialize_sampling(self, unlabeled_data: List[str], unlabeled_indices: List[int], **kwargs) -> None:
        """Initialize the sampling process"""
        print("Initializing graph density-based sampling...")
        try:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._unlabeled_indices = unlabeled_indices.copy()
            value_ranking = self.compute_value_ranking(unlabeled_data, **kwargs)
            self._remaining_indices = [self._unlabeled_indices[i] for i in value_ranking]
            
            print(f"Sampling initialized with {len(self._remaining_indices)} samples")
            
        except Exception as e:
            print(f"Error during sampling initialization: {str(e)}")
            raise

class UncertaintySampling(BaseStrategy):
    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        predictions = self.get_predictions(unlabeled_data)
        uncertainties = -np.max(predictions, axis=1)
        return np.argsort(uncertainties).tolist()
    
class EntropyBasedSampling(BaseStrategy):
    """
    Entropy-based Sampling

    Selects samples with the highest entropy in their prediction probability distribution,
    indicating the model's predictions are more uniform and thus more uncertain.
    """
    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        predictions = self.get_predictions(unlabeled_data)

        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        predictions = np.clip(predictions, epsilon, 1.0)

        # Compute entropy for each sample (-sum(p_i * log(p_i)))
        entropies = -np.sum(predictions * np.log(predictions), axis=1)

        # Return indices sorted by entropy in descending order (higher entropy = higher uncertainty)
        return np.argsort(-entropies).tolist()


class LeastConfidenceSampling(BaseStrategy):
    """
    Least Confidence Sampling

    Selects samples with the lowest maximum prediction probability, which are the most uncertain.
    """
    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        predictions = self.get_predictions(unlabeled_data)

        # Get the maximum prediction probability for each sample
        confidences = np.max(predictions, axis=1)

        # Return indices sorted by confidence in ascending order (lower confidence = higher uncertainty)
        return np.argsort(confidences).tolist()


class MarginSampling(BaseStrategy):
    """
    Margin Sampling

    Selects samples with the smallest gap between the highest and second-highest prediction
    probabilities. These samples are near the decision boundary and the model is more uncertain.
    """
    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        predictions = self.get_predictions(unlabeled_data)

        # Find the highest and second-highest prediction probabilities for each sample
        sorted_predictions = np.sort(predictions, axis=1)
        # Compute the gap between the highest and second-highest probabilities
        margins = sorted_predictions[:, -1] - sorted_predictions[:, -2]

        # Return indices sorted by margin in ascending order (smaller margin = higher uncertainty)
        return np.argsort(margins).tolist()

class CoreSetSampling(BaseStrategy):
    """
    Optimized Core-Set sampling strategy

    Uses parallel computation and batch processing to accelerate distance calculations
    """
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.sentence_transformer = None
        self.batch_size = kwargs.get('batch_size', 128)
        self.distance_metric = kwargs.get('distance_metric', 'euclidean')
        self.n_jobs = kwargs.get('n_jobs', -1)  # Number of parallel jobs, -1 means use all CPUs
        self.use_gpu = kwargs.get('use_gpu', True) and torch.cuda.is_available()
        self.chunk_size = kwargs.get('chunk_size', 10000)  # Chunk size for batch processing

        # Store compute resource information
        self.num_cpus = os.cpu_count()
        self.has_gpu = torch.cuda.is_available()
        
        # If sentence_transformer_model param is provided, use the provided model
        self.sentence_transformer_model = kwargs.get('sentence_transformer_model', './base-model/paraphrase-MiniLM-L6-v2')
        
        # Store labeled sample embeddings
        self.labeled_embeddings = None
        
        # Cache computed results
        self.embedding_cache = {}
        
        print(f"Initialized CoreSetSampling with {self.num_cpus} CPUs, GPU: {self.has_gpu}, "
              f"batch_size: {self.batch_size}, chunk_size: {self.chunk_size}")
    
    def _initialize_sentence_transformer(self):
        """Initialize sentence transformer (if not already initialized)"""
        if self.sentence_transformer is None:
            from sentence_transformers import SentenceTransformer
            try:
                self.sentence_transformer = SentenceTransformer(self.sentence_transformer_model)
                if self.use_gpu:
                    self.sentence_transformer = self.sentence_transformer.to(torch.device('cuda'))
                print(f"Initialized sentence transformer model: {self.sentence_transformer_model}")
            except Exception as e:
                print(f"Error initializing sentence transformer: {str(e)}")
                raise
    
    def _get_embeddings(self, texts):
        """Get text embeddings, using cache for performance"""
        self._initialize_sentence_transformer()
        
        # Create unique identifier for caching
        text_hash = hash(tuple(texts))
        if text_hash in self.embedding_cache:
            print("Using cached embeddings")
            return self.embedding_cache[text_hash]
        
        print(f"Computing embeddings for {len(texts)} texts...")
        
        # Use sentence transformer batch processing
        embeddings = self.sentence_transformer.encode(
            texts, 
            batch_size=self.batch_size, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Cache results
        self.embedding_cache[text_hash] = embeddings
        
        return embeddings
    
    def _compute_distances_parallel(self, X, Y=None, metric='euclidean'):
        """
        Compute distance matrix in parallel

        Args:
            X: Array of shape (n_samples_X, n_features)
            Y: Array of shape (n_samples_Y, n_features), defaults to X
            metric: Distance metric, defaults to euclidean

        Returns:
            Distance matrix of shape (n_samples_X, n_samples_Y)
        """
        if Y is None:
            Y = X
        
        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]
        
        # Select computation method
        if self.use_gpu and X.shape[0] * Y.shape[0] <= 100000000:  # Limit GPU memory usage
            return self._compute_distances_gpu(X, Y, metric)
        else:
            return self._compute_distances_cpu_parallel(X, Y, metric)
    
    def _compute_distances_gpu(self, X, Y, metric='euclidean'):
        """Compute distances using GPU acceleration"""
        # print("Computing distances with GPU...")
        start_time = time.time()
        
        # Transfer data to GPU
        X_tensor = torch.tensor(X, device='cuda', dtype=torch.float32)
        Y_tensor = torch.tensor(Y, device='cuda', dtype=torch.float32)
        
        # Compute squared Euclidean distance
        if metric == 'euclidean':
            # Use chunked computation to avoid GPU memory overflow
            distances = torch.zeros((X.shape[0], Y.shape[0]), device='cuda')
            
            for i in range(0, X.shape[0], self.chunk_size):
                end_i = min(i + self.chunk_size, X.shape[0])
                X_chunk = X_tensor[i:end_i]
                
                for j in range(0, Y.shape[0], self.chunk_size):
                    end_j = min(j + self.chunk_size, Y.shape[0])
                    Y_chunk = Y_tensor[j:end_j]
                    
                    # Compute chunk distance
                    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
                    X_squared = torch.sum(X_chunk ** 2, dim=1, keepdim=True)
                    Y_squared = torch.sum(Y_chunk ** 2, dim=1, keepdim=True).t()
                    XY = torch.mm(X_chunk, Y_chunk.t())
                    dist_chunk = X_squared + Y_squared - 2 * XY
                    
                    # Ensure no negative values (numerical errors may cause tiny negatives)
                    dist_chunk = torch.clamp(dist_chunk, min=0.0)
                    
                    # Take square root to get Euclidean distance
                    distances[i:end_i, j:end_j] = torch.sqrt(dist_chunk)
                    
                    # Clean up temporary variables to save memory
                    del X_squared, Y_squared, XY, dist_chunk
                    torch.cuda.empty_cache()
            
            # Transfer results back to CPU
            distances_np = distances.cpu().numpy()
            
            # Free GPU memory
            del X_tensor, Y_tensor, distances
            torch.cuda.empty_cache()
            
        elif metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            # Normalize vectors
            X_norm = F.normalize(X_tensor, p=2, dim=1)
            Y_norm = F.normalize(Y_tensor, p=2, dim=1)
            
            # Compute cosine similarity matrix in chunks
            similarities = torch.zeros((X.shape[0], Y.shape[0]), device='cuda')
            
            for i in range(0, X.shape[0], self.chunk_size):
                end_i = min(i + self.chunk_size, X.shape[0])
                X_chunk = X_norm[i:end_i]
                
                for j in range(0, Y.shape[0], self.chunk_size):
                    end_j = min(j + self.chunk_size, Y.shape[0])
                    Y_chunk = Y_norm[j:end_j]
                    
                    # Compute cosine similarity
                    similarities[i:end_i, j:end_j] = torch.mm(X_chunk, Y_chunk.t())
            
            # Cosine distance = 1 - cosine similarity
            distances = 1 - similarities
            distances_np = distances.cpu().numpy()
            
            # Free GPU memory
            del X_tensor, Y_tensor, X_norm, Y_norm, similarities, distances
            torch.cuda.empty_cache()
        
        else:
            # For other distance metrics, fall back to CPU computation
            return self._compute_distances_cpu_parallel(X, Y, metric)
        
        # print(f"GPU distance computation completed in {time.time() - start_time:.2f} seconds")
        return distances_np
    
    def _compute_distances_cpu_parallel(self, X, Y, metric='euclidean'):
        """Compute distance matrix using multi-process parallelism"""
        from concurrent.futures import ProcessPoolExecutor
        from scipy.spatial.distance import cdist
        
        print(f"Computing distances with parallel CPU processing ({self.n_jobs} jobs)...")
        start_time = time.time()
        
        # Determine actual number of jobs to use
        n_jobs = self.n_jobs if self.n_jobs > 0 else self.num_cpus
        
        # Determine optimal chunk size based on CPU count and data size
        chunk_size = min(max(1, X.shape[0] // (n_jobs * 4)), self.chunk_size)
        n_chunks = (X.shape[0] + chunk_size - 1) // chunk_size
        
        print(f"Using {n_jobs} CPU cores with {n_chunks} chunks of size {chunk_size}")
        
        # Create result array
        distances = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)
        
        # Define processing function for a single chunk
        def process_chunk(start_idx, end_idx):
            return start_idx, end_idx, cdist(X[start_idx:end_idx], Y, metric=metric)
        
        # Compute distances in parallel using process pool
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(process_chunk, i * chunk_size, min((i + 1) * chunk_size, X.shape[0])): i
                for i in range(n_chunks)
            }
            
            # Process results
            for i, future in enumerate(tqdm(future_to_chunk, desc="Computing distances")):
                start_idx, end_idx, chunk_distances = future.result()
                distances[start_idx:end_idx] = chunk_distances
        
        print(f"Parallel CPU distance computation completed in {time.time() - start_time:.2f} seconds")
        return distances
    
    def _greedy_k_center(self, unlabeled_embeddings, labeled_embeddings, n_samples):
        """
        Select samples using greedy k-center algorithm, optimized version

        Args:
            unlabeled_embeddings: Embeddings of unlabeled samples
            labeled_embeddings: Embeddings of labeled samples (if any)
            n_samples: Number of samples to select

        Returns:
            List of selected indices
        """
        print(f"Running greedy k-center algorithm to select {n_samples} samples...")
        start_time = time.time()
        
        selected_indices = []
        
        # If no labeled samples, randomly select one as starting point
        if labeled_embeddings is None or len(labeled_embeddings) == 0:
            random_idx = np.random.randint(0, len(unlabeled_embeddings))
            selected_indices.append(random_idx)
            n_samples -= 1
        
        if n_samples <= 0:
            return selected_indices
        
        # Compute distances between unlabeled and labeled samples
        if labeled_embeddings is not None and len(labeled_embeddings) > 0:
            print("Computing distances between unlabeled and labeled samples...")
            # Compute distance from each unlabeled sample to nearest labeled sample
            distances_to_labeled = self._compute_distances_parallel(
                unlabeled_embeddings, 
                labeled_embeddings,
                metric=self.distance_metric
            )
            min_distances = np.min(distances_to_labeled, axis=1)
            
            # Free memory
            del distances_to_labeled
            gc.collect()
        else:
            # If no labeled samples, use the first selected sample as reference
            first_sample = unlabeled_embeddings[selected_indices[0]].reshape(1, -1)
            distances_to_first = self._compute_distances_parallel(
                unlabeled_embeddings, 
                first_sample,
                metric=self.distance_metric
            )
            min_distances = distances_to_first.reshape(-1)
            
            # Set first sample's distance to infinity to avoid reselection
            min_distances[selected_indices[0]] = float('inf')
            
            # Free memory
            del distances_to_first
            gc.collect()
        
        # Use mask to track selectable samples
        available_mask = np.ones(len(unlabeled_embeddings), dtype=bool)
        for idx in selected_indices:
            available_mask[idx] = False
        
        # Greedily select remaining samples
        print(f"Selecting {n_samples} samples...")
        for i in tqdm(range(n_samples), desc="Selecting samples"):
            if not np.any(available_mask):
                print("No more available samples to select")
                break
                
            # Only consider distances of available samples
            masked_distances = min_distances.copy()
            masked_distances[~available_mask] = -np.inf
            
            # Select the point farthest from already selected samples
            next_idx = np.argmax(masked_distances)
            selected_indices.append(next_idx)
            available_mask[next_idx] = False
            
            # Update minimum distances (only when needed)
            if i < n_samples - 1:  # No update needed for the last sample
                next_sample = unlabeled_embeddings[next_idx].reshape(1, -1)
                distances_to_next = self._compute_distances_parallel(
                    unlabeled_embeddings, 
                    next_sample,
                    metric=self.distance_metric
                ).reshape(-1)
                
                # Update minimum distance from each unlabeled sample to selected set
                min_distances = np.minimum(min_distances, distances_to_next)
                
                # Free memory
                del distances_to_next
                gc.collect()
            
            # Periodically print progress and memory usage
            if (i + 1) % 1000 == 0 or i == 0:
                memory_info = psutil.Process().memory_info()
                print(f"Selected {i + 1}/{n_samples} samples. "
                      f"Memory usage: {memory_info.rss / (1024 ** 3):.2f} GB")
        
        # Clean up memory
        del min_distances, available_mask
        gc.collect()
        
        print(f"Greedy k-center completed in {time.time() - start_time:.2f} seconds")
        return selected_indices
    
    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        """Compute sample value ranking"""
        print("Computing value ranking with optimized Core-Set sampling...")

        # Get unlabeled sample embeddings
        unlabeled_embeddings = self._get_embeddings(unlabeled_data)
        
        # If labeled samples exist, get their embeddings
        labeled_data = kwargs.get('labeled_data', None)
        if labeled_data is not None and len(labeled_data) > 0:
            print(f"Computing embeddings for {len(labeled_data)} labeled samples...")
            self.labeled_embeddings = self._get_embeddings(labeled_data)
        
        # Limit number of samples processed for efficiency
        max_samples = min(len(unlabeled_data), 10000)  # Process at most 10000 samples
        if len(unlabeled_data) > max_samples:
            print(f"Warning: Limiting to {max_samples} samples for efficiency")
            value_ranking_full = np.zeros(len(unlabeled_data), dtype=int)
            
            # Randomly select subset for processing
            subset_indices = np.random.choice(len(unlabeled_data), max_samples, replace=False)
            subset_embeddings = unlabeled_embeddings[subset_indices]
            
            # Run greedy k-center algorithm on subset
            subset_ranking = self._greedy_k_center(
                subset_embeddings, 
                self.labeled_embeddings, 
                max_samples
            )
            
            # Map subset ranking back to original indices
            value_ranking_full[:max_samples] = subset_indices[subset_ranking]
            
            # Randomly order remaining samples
            remaining_indices = np.setdiff1d(np.arange(len(unlabeled_data)), subset_indices)
            np.random.shuffle(remaining_indices)
            value_ranking_full[max_samples:] = remaining_indices
            
            return value_ranking_full.tolist()
        else:
            # Run greedy k-center algorithm on full dataset
            selected_indices = self._greedy_k_center(
                unlabeled_embeddings, 
                self.labeled_embeddings, 
                len(unlabeled_data)
            )
            
            return selected_indices
    
    def initialize_sampling(self, unlabeled_data: List[str], unlabeled_indices: List[int], **kwargs) -> None:
        """Initialize the sampling process"""
        print("Initializing optimized Core-Set sampling...")
        self._unlabeled_indices = unlabeled_indices.copy()

        # Get labeled sample data (if available)
        labeled_data = kwargs.get('labeled_data', None)
        
        try:
            # Compute sample value ranking
            value_ranking = self.compute_value_ranking(unlabeled_data, labeled_data=labeled_data)
            self._remaining_indices = [self._unlabeled_indices[i] for i in value_ranking]
            
            print(f"Core-Set sampling initialized with {len(self._remaining_indices)} samples")
        except Exception as e:
            print(f"Error during Core-Set initialization: {str(e)}")
            # If error occurs, fall back to random ordering
            print("Falling back to random sampling")
            np.random.shuffle(self._unlabeled_indices)
            self._remaining_indices = self._unlabeled_indices.copy()
    
    def update_labeled_embeddings(self, new_labeled_data: List[str]) -> None:
        """Update labeled sample embeddings"""
        if not new_labeled_data:
            return
            
        print(f"Updating labeled embeddings with {len(new_labeled_data)} new samples...")
        new_embeddings = self._get_embeddings(new_labeled_data)
        
        if self.labeled_embeddings is None:
            self.labeled_embeddings = new_embeddings
        else:
            self.labeled_embeddings = np.vstack([self.labeled_embeddings, new_embeddings])
            
        print(f"Updated labeled embeddings, new shape: {self.labeled_embeddings.shape}")
    
    def __del__(self):
        """Clean up resources"""
        # Clear cache
        self.embedding_cache.clear()

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class RandomSampling(BaseStrategy):
    def compute_value_ranking(self, unlabeled_data: List[str], **kwargs) -> List[int]:
        return np.random.permutation(len(unlabeled_data)).tolist()

class BoundaryNoMemorySampling(BoundaryAwareSampling):
    """
    Ablation study 1: Boundary-aware sampling without dynamic sample memory

    Compared to the original BoundaryAwareSampling, this class removes the minority class
    memory bank (minority_memory) to evaluate the impact of the memory mechanism on performance.
    """
    def __init__(self, model, tokenizer=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        print("Initializing BoundaryNoMemorySampling - Ablation study without memory mechanism")
    
    def _calculate_diversity_score(self, feature: np.ndarray) -> float:
        """
        Modified diversity score computation without memory bank

        Since the memory bank is removed, this method always returns a fixed value,
        effectively disabling the memory bank's contribution to diversity scoring
        """
        # No memory bank used, always return fixed value
        return 1.0
    
    def update_memory(self, labeled_indices: List[int], labels: List[int]) -> None:
        """
        Update class statistics without updating memory bank
        """
        try:
            # Only update class statistics
            for label in labels:
                self.class_counts[label] += 1
            
            # Remove labeled samples from boundary buffer
            self.boundary_buffer = deque(
                [x for x in self.boundary_buffer if x not in labeled_indices],
                maxlen=self.buffer_size
            )
            
        except Exception as e:
            print(f"Error updating class counts: {str(e)}")
            raise

class BoundaryNoAdversarialSampling(BoundaryAwareSampling):
    """
    Ablation study 2: Boundary-aware sampling without adversarial training

    Compared to the original BoundaryAwareSampling, this class removes adversarial perturbation
    generation and usage to evaluate the impact of adversarial training on performance.
    """
    def __init__(self, model, tokenizer=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        # Set adversarial perturbation size to 0, effectively disabling adversarial training
        self.eps = 0.0
        print("Initializing BoundaryNoAdversarialSampling - Ablation study without adversarial training")
    
    def _detect_boundary_samples(self, features: np.ndarray) -> None:
        """
        Detect boundary samples using non-adversarial method
        """
        try:
            predictions = self.get_predictions(features)

            # Compute entropy of prediction probabilities without adversarial perturbation
            probs = np.sort(predictions, axis=1)
            confidence_gaps = probs[:, -1] - probs[:, -2]  # Top-2 probability gap

            # Identify low-confidence samples
            boundary_candidates = np.where(confidence_gaps < self.tau)[0]

            if len(boundary_candidates) > 0:
                # Get candidate sample features
                candidate_features = self._features[boundary_candidates]

                # Cluster analysis
                clusterer = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = clusterer.fit_predict(candidate_features)

                # Find outliers as boundary samples
                boundary_points = boundary_candidates[cluster_labels == -1]

                # Update boundary buffer
                self.boundary_buffer.extend([
                    self._unlabeled_indices[i] for i in boundary_points
                ])
                
                print(f"Detected {len(boundary_points)} boundary samples without adversarial perturbation")
            
        except Exception as e:
            print(f"Error during boundary detection: {str(e)}")
            raise

class BoundaryRandomRankingSampling(BoundaryAwareSampling):
    """
    Ablation study 3: Boundary-aware sampling with random ranking

    Compared to the original BoundaryAwareSampling, this class uses random ranking to select
    samples instead of the combined scoring based on uncertainty, class weight, and diversity.
    """
    def __init__(self, model, tokenizer=None, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        print("Initializing BoundaryRandomRankingSampling - Ablation study with random sampling order")
    
    def compute_value_ranking(self, features: np.ndarray, **kwargs) -> List[int]:
        """
        Use random ranking to select samples
        """
        try:
            # Simply return randomly ordered indices
            return np.random.permutation(len(features)).tolist()
            
        except Exception as e:
            print(f"Error during random value ranking computation: {str(e)}")
            raise


def get_strategy(strategy_name: str, model, tokenizer=None, **kwargs) -> BaseStrategy:
    strategies = {
        'uncertainty': UncertaintySampling, # LeastConfidenceSampling
        'entropy': EntropyBasedSampling,
        'confidence': LeastConfidenceSampling,
        'margin': MarginSampling,
        'random': RandomSampling,
        'diversity': DiversitySampling,
        'density': DensitySampling,
        'graph_density': GraphDensitySampling,
        'bass': BoundaryAwareSampling,
        'qbc': QBCSampling,
        'coreset': CoreSetSampling,
        'imbalance': ImbalancedAwareSampling,
        'batl': BATLSampling,
        'bass_no_memory': BoundaryNoMemorySampling,
        'bass_no_adversarial': BoundaryNoAdversarialSampling,
        'bass_random': BoundaryRandomRankingSampling,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: {list(strategies.keys())}")
    
    try:
        if tokenizer is None:
            raise ValueError(f"Strategy {strategy_name} requires a tokenizer")

        if strategy_name == 'qbc':
            if 'model_path' not in kwargs:
                raise ValueError("QBC strategy requires model_path parameter")

        return strategies[strategy_name](model, tokenizer, **kwargs)

    except Exception as e:
        print(f"Error creating strategy {strategy_name}: {str(e)}")
        raise
