from dataclasses import dataclass
from typing import List, Optional
import os
import torch
from transformers import TrainingArguments

DATASET_ID = "DATASET_X"
MODEL_TYPE = "MODEL_Y"
STRATEGY_TYPE = "strategy_z"

@dataclass
class TrainingArgs:
    adapter_rank: int = 8
    adapter_scaling: int = 16
    adapter_regularization: float = 0.1
    
    mini_batch_size: int = 16
    optimizer_rate: float = 2e-5
    training_cycles: int = 3
    sequence_length: int = 128
    regularization_factor: float = 0.01
    
    results_directory: str = "./output_folder"
    checkpoint_directory: str = f"./models/FT-{MODEL_TYPE}-{DATASET_ID}"
    
    def __post_init__(self):
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
            
    def get_training_arguments(self):
        return TrainingArguments(
            output_dir=self.results_directory,
            learning_rate=self.optimizer_rate,
            per_device_train_batch_size=self.mini_batch_size,
            num_train_epochs=self.training_cycles,
            weight_decay=self.regularization_factor,
            logging_dir="./log_files",
            logging_steps=100,
            save_strategy="steps",
            eval_strategy="steps",
            eval_steps=100,
            save_steps=500,
            warmup_steps=0,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            gradient_accumulation_steps=2,
            fp16=True,
            dataloader_num_workers=0,
            report_to=["metrics_tracker"],
        )

@dataclass
class DataConfig:
    corpus_location: str = f'./corpus/{DATASET_ID}_processed_samples.npz'
    max_token_count: int = 128
    evaluation_samples: int = 5

@dataclass
class ModelConfig:
    pretrained_path: str = f'./pretrained/{MODEL_TYPE}-{DATASET_ID}'
    results_path: str = f"./outputs/{MODEL_TYPE}-{DATASET_ID}-{STRATEGY_TYPE}"
    offline_mode: bool = True
    
    def __post_init__(self):
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

@dataclass
class AdapterConfig:
    enable_adapter: bool = True
    adapter_task: str = "CLASSIFICATION"
    adapter_dimension: int = 16
    adapter_scale: int = 32
    adapter_dropout: float = 0.1
    target_layers: List[str] = None
    parameter_mode: str = "none"
    preserve_modules: List[str] = None

    def __post_init__(self):
        if self.target_layers is None:
            self.target_layers = ["attention_q", "attention_k", "attention_v", "ffn"]
        if self.preserve_modules is None:
            self.preserve_modules = ["output_layer"]

@dataclass
class TrainingConfig:
    def __init__(self):
        self.learning_rate = 2e-5
        self.batch_size = 16
        self.num_epochs = 3
        self.weight_decay = 0.01
        self.logging_dir = './logs'
        self.logging_steps = 100
        self.save_strategy = 'steps'
        self.eval_steps = 100
        self.save_steps = 500
        self.warmup_steps = 0
        self.metrics_project = f"{MODEL_TYPE}-training-{DATASET_ID}"
        self.fp16 = False
        self.gradient_accumulation_steps = 1

@dataclass
class LogConfig:
    verbosity: str = "INFO"
    log_pattern: str = '%(asctime)s - %(levelname)s - %(message)s'
    log_output: str = 'training.log'

@dataclass
class LearningConfig:
    seed_labeled_count: int = 10
    selection_batch: int = 100
    max_rounds: int = 100
    performance_target: float = 99.99
    random_seed: int = 42
    validation_ratio: float = 0.2
    processing_chunk: int = 500
    
    algorithm: str = STRATEGY_TYPE
    
    memory_buffer: int = 5000 
    history_buffer: int = 1000
    confidence_threshold: float = 0.1
    uncertainty_margin: float = 0.01
    weight_primary: float = 0.4
    weight_secondary: float = 0.3
    weight_tertiary: float = 0.3
    softmax_temperature: float = 1.0
    
    # Algorithm specific parameters
    lambda_factor: float = 1.0
    decision_boundary: float = 0.3
    
    cluster_distance: float = 0.5     # Clustering parameter 1
    cluster_density: int = 5          # Clustering parameter 2
    neighbor_count: int = 5
    
    vector_similarity: float = 0.7
    
    ensemble_size: int = 3
    ensemble_dropout: float = 0.2
    entropy_importance: float = 0.5
    divergence_importance: float = 0.5
    
    balance_criterion: float = 0.3
    
    def __post_init__(self):
        valid_algorithms = {'diversity_sampling', 'random_sampling', 'algorithm_a', 
                           'algorithm_b', 'class_balance', 'boundary_detection',
                           'uncertainty_sampling', 'entropy_based', 'representation_based'}
        
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {self.algorithm}. Must be one of {valid_algorithms}")

class Config:
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.adapter = AdapterConfig()
        self.training = TrainingConfig()
        self.log = LogConfig()
        self.learning = LearningConfig()
        self.hardware = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def hardware(self):
        return self._hardware

    @hardware.setter
    def hardware(self, value):
        self._hardware = value

    def get_algorithm_parameters(self) -> dict:
        params = {
            'cluster_distance': self.learning.cluster_distance,
            'cluster_density': self.learning.cluster_density,
            'neighbor_count': self.learning.neighbor_count
        }
        
        if self.learning.algorithm == 'hybrid_approach':
            params.update({
                'primary_weight': self.learning.weight_primary,
                'secondary_weight': self.learning.weight_secondary,
                'adaptation_step': self.learning.weight_update_step
            })
        
        return params

configuration = Config()
training_parameters = TrainingArgs()