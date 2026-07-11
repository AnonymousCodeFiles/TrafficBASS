from dataclasses import dataclass
from typing import List
import os
import torch
from transformers import TrainingArguments

# ============================================================
# Main control variables — modify here to switch experiment config
# ============================================================
dataset = "USTC"                # USTC, UNSW, DoH, MalAnd, ...
baseModel = "ALBERT"            # ALBERT, BERT, DistilBERT, RoBERTa, MobileBERT, Q8BERT, ...
alstrategy = "bass"             # bass, random, margin, confidence, entropy, coreset,
                                # batl, qbc, diversity, density, graph_density,
                                # sunb, uncertainty, imbalance,
                                # bass_no_memory, bass_no_adversarial, bass_random

# Cross-domain experiment parameters (only used by s55_al_crossdomain.py)
basePLM = None                  # Pre-training dataset source; defaults to dataset when None

# Ablation experiment parameters (only used by s54_al_ablation.py)
ft_str = "Lora"                 # Lora, Adapter, Full
is_random = "False"             # True / False — whether to use randomly initialized weights

# Data file suffix: set True to use _samp.npz, False to use .npz
use_sampled_data = False

# ============================================================

_effective_basePLM = basePLM if basePLM is not None else dataset


@dataclass
class TrainingArgs:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_length: int = 128
    weight_decay: float = 0.01

    output_dir: str = "./results"
    model_save_path: str = f"./ftune-model/Ft-{baseModel}-{dataset}"

    def __post_init__(self):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)

    def get_training_arguments(self):
        return TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            logging_dir="./logs",
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
            report_to=["wandb"],
        )


_data_suffix = "_concate_data_samp.npz" if use_sampled_data else "_concate_data.npz"


@dataclass
class DataConfig:
    data_path: str = f'./data/{dataset}{_data_suffix}'
    max_length: int = 128
    test_samples: int = 5


@dataclass
class ModelConfig:
    model_path: str = f'./plmd-model/{baseModel}-{_effective_basePLM}'
    output_dir: str = f"./results/{baseModel}-{dataset}-{alstrategy}"
    local_files_only: bool = True

    def __post_init__(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class LoRAConfig:
    use_lora: bool = True
    task_type: str = "SEQ_CLS"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"
    modules_to_save: List[str] = None

    adapter_size: int = 16
    adapter_layers: int = 2
    adapter_dropout: float = 0.1

    def __post_init__(self):
        self.use_lora = (ft_str.lower() == "lora")
        if self.target_modules is None:
            self.target_modules = ["query", "key", "value", "dense"]
        if self.modules_to_save is None:
            self.modules_to_save = ["classifier"]
        if self.adapter_size == 0:
            self.adapter_size = self.r


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
        self.wandb_project = f"{baseModel}-finetuning-{dataset}"
        self.fp16 = False
        self.gradient_accumulation_steps = 1


@dataclass
class LogConfig:
    level: str = "INFO"
    format: str = '%(asctime)s - %(levelname)s - %(message)s'
    log_file: str = 'training.log'


@dataclass
class ActiveLearningConfig:
    initial_labeled_samples: int = 10
    query_size: int = 100
    max_iterations: int = 100
    target_accuracy: float = 99.99
    random_seed: int = 42
    test_size: float = 0.2
    batch_size: int = 500

    strategy: str = alstrategy

    # BASS boundary-aware sampling parameters
    buffer_size: int = 5000
    memory_size: int = 1000
    tau: float = 0.1
    adv_eps: float = 0.01
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    temperature: float = 1.0

    # BATL triplet loss parameters
    lambda_scale: float = 1.0
    margin: float = 0.3

    # SUNB parameters
    uncertainty_weight: float = 0.5
    diversity_weight: float = 0.5
    weight_update_step: float = 0.1

    # DBSCAN / density / graph density parameters
    dbscan_eps: float = 0.5
    min_samples: int = 5
    k_neighbors: int = 5

    # Diversity parameters
    similarity_threshold: float = 0.7

    # QBC parameters
    n_committees: int = 3
    dropout_rate: float = 0.2
    vote_entropy_weight: float = 0.5
    kl_weight: float = 0.5

    balance_threshold: float = 0.3

    def __post_init__(self):
        valid_strategies = {
            'sunb', 'uncertainty', 'diversity', 'density', 'graph_density',
            'random', 'bass', 'qbc', 'batl', 'imbalance',
            'margin', 'confidence', 'entropy', 'coreset',
            'bass_no_memory', 'bass_no_adversarial', 'bass_random',
        }
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be one of {valid_strategies}")

        if self.uncertainty_weight + self.diversity_weight != 1.0:
            raise ValueError("Uncertainty weight and diversity weight must sum to 1.0")


class Config:
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.log = LogConfig()
        self.al = ActiveLearningConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def get_strategy_params(self) -> dict:
        params = {
            'eps': self.al.dbscan_eps,
            'min_samples': self.al.min_samples,
            'k_neighbors': self.al.k_neighbors
        }

        if self.al.strategy == 'sunb':
            params.update({
                'uncertainty_weight': self.al.uncertainty_weight,
                'diversity_weight': self.al.diversity_weight,
                'weight_update_step': self.al.weight_update_step
            })

        return params


config = Config()
training_args_ = TrainingArgs()
