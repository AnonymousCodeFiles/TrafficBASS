import torch
from transformers import (
    AlbertForMaskedLM, AlbertTokenizer, AlbertForSequenceClassification,
    BertForMaskedLM, BertTokenizer, BertForSequenceClassification,
    DistilBertForMaskedLM, DistilBertTokenizer, DistilBertForSequenceClassification,
    RobertaForMaskedLM, RobertaTokenizer, RobertaForSequenceClassification,
    AutoModelForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification,
    MobileBertForMaskedLM, MobileBertTokenizer, MobileBertForSequenceClassification,
    Trainer, TrainingArguments
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    PeftConfig
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import numpy as np
import random
import argparse
import os, json
import psutil
from tqdm import tqdm
from joblib import Parallel, delayed
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from utils.data_loader import get_device
from utils.dataset import PretrainingDataset

os.environ.setdefault("WANDB_MODE", "disabled")

def check_pretrained_model_exists(model_name, dataset_name):
    """Check if a pre-trained model already exists"""
    pretrain_dir = f'./plmd-model/{model_name}_{dataset_name}'
    
    # Check if the model directory exists
    if os.path.exists(pretrain_dir):
        # Check if key files exist - more thorough check
        required_files = [
            'config.json',  # basic config
            'pytorch_model.bin',  # model weights
            'tokenizer_config.json',  # tokenizer config
            'vocab.txt'  # vocabulary (some models may use a different filename)
        ]
        
        # Check main files
        if os.path.exists(os.path.join(pretrain_dir, 'config.json')) and \
           (os.path.exists(os.path.join(pretrain_dir, 'pytorch_model.bin')) or 
            os.path.exists(os.path.join(pretrain_dir, 'model.safetensors'))):
            print(f"Found pre-trained model directory: {pretrain_dir}")
            print(f"Files in directory: {os.listdir(pretrain_dir)}")
            return True, pretrain_dir
    
    print(f"No valid pre-trained model found: {pretrain_dir}")
    return False, pretrain_dir

def data_import(path):
    """Efficiently load and preprocess the dataset"""
    from utils.data_loader import parallel_load_data, estimate_memory_usage, get_optimal_chunk_size, process_chunk
    print("Loading data...")
    try:
        file_size = os.path.getsize(path)
        print(f"Data file size: {file_size / (1024**2):.2f} MB")

        with np.load(path, mmap_mode='r') as data:
            n_samples = len(data['features'])
            feature_shape = data['features'][0].shape if len(data['features']) > 0 else (0,)
            estimated_memory = estimate_memory_usage(n_samples, np.prod(feature_shape))
            print(f"Estimated memory requirement: {estimated_memory:.2f} GB")

        available_memory = psutil.virtual_memory().available / (1024**3)
        print(f"Available system memory: {available_memory:.2f} GB")

        if estimated_memory < available_memory * 0.7:
            X, y = parallel_load_data(
                path,
                n_jobs=max(1, psutil.cpu_count(logical=False) - 1)
            )
        else:
            chunk_size = get_optimal_chunk_size(n_samples, np.prod(feature_shape))
            print(f"Processing data in chunks of {chunk_size} samples")
            X = []
            y = []
            with np.load(path, mmap_mode='r') as data:
                for i in tqdm(range(0, n_samples, chunk_size), desc="Loading data chunks"):
                    end = min(i + chunk_size, n_samples)
                    chunk_features = process_chunk(data['features'][i:end])
                    X.extend(chunk_features)
                    y.extend(data['labels'][i:end])
            X = np.array(X)
            y = np.array(y, dtype=np.int64)

        print(f"\nTotal samples loaded: {len(y)}")
        print(f"Feature array shape: {X.shape}")
        print(f"Number of unique labels: {len(np.unique(y))}")
        return X, y

    except Exception as e:
        print(f"Error loading data: {e}")
        raise

class MobileBertMLMCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        
    def __call__(self, examples):
        # Batch process inputs
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            padding="longest"
        )
        
        device = batch["input_ids"].device
        
        # Create MLM labels (copy input_ids)
        labels = batch["input_ids"].clone()
        
        # Perform masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # We want to predict only 15% of tokens: 80% replaced with [MASK],
        # 10% replaced with random tokens, 10% kept unchanged
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        batch["input_ids"][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace with random tokens
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        batch["input_ids"][indices_random] = random_words[indices_random]
        
        # The remaining 10% are kept unchanged
        
        # Set labels of non-masked tokens to -100 so they are ignored in loss computation
        labels[~masked_indices] = -100
        batch["labels"] = labels
        
        return batch

# Custom pre-training dataset (s41-specific, with caching)
class PretrainingDataset(Dataset):
    def __init__(self, features, labels, tokenizer, max_length=128, mlm_probability=0.15, cache_size=1000):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.cache_size = min(cache_size, len(features))
        self.cache = {}  # Cache for processed samples
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Check if sample is already in cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Get a single feature and convert to string
        feature = str(self.features[idx])
        
        # Use tokenizer to process text
        encoding = self.tokenizer(
            feature,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get input_ids
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Create MLM labels (copy input_ids)
        labels = input_ids.clone()
        
        # Randomly mask some tokens
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels, already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Replace selected tokens with [MASK] token
        labels[~masked_indices] = -100  # Only compute loss for masked tokens
        input_ids[masked_indices] = self.tokenizer.mask_token_id
        
        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        # Manage cache size, cache recently processed samples
        if len(self.cache) >= self.cache_size:
            # Remove the oldest cache entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        # Add sample to cache
        self.cache[idx] = sample
        
        return sample

# Custom fine-tuning dataset
class FineTuningDataset(Dataset):
    def __init__(self, features, labels, tokenizer, max_length=128):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get a single feature and convert to string
        feature = str(self.features[idx])
        label = int(self.labels[idx])
        
        # Use tokenizer to process text
        encoding = self.tokenizer(
            feature,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Split data
def split_data(X, y):
    print("Splitting data into train and test sets...")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=43)
    print(f"Train set: {len(x_train)} samples")
    print(f"Test set: {len(x_test)} samples")
    return x_train, x_test, y_train, y_test

def get_model_and_tokenizer(model_name, num_labels=None, for_pretraining=True):
    base_path = './base-model'
    model_paths = {
        'ALBERT': f'{base_path}/albert-base-v2',
        'BERT': f'{base_path}/bert-base-uncased',
        'DistilBERT': f'{base_path}/distilbert-base-uncased',
        'RoBERTa': f'{base_path}/roberta-base',  # 'roberta-base'
        'TinyBERT': f'{base_path}/TinyBERT_General_4L_312D',
        'MobileBERT': f'{base_path}/mobilebert-uncased'
    }
    
    model_path = model_paths.get(model_name, model_paths['ALBERT'])
    
    if for_pretraining:
        # Pre-training model
        if model_name == 'ALBERT':
            model = AlbertForMaskedLM.from_pretrained(model_path)
            tokenizer = AlbertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        elif model_name == 'BERT':
            model = BertForMaskedLM.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        elif model_name == 'DistilBERT':
            model = DistilBertForMaskedLM.from_pretrained(model_path)
            tokenizer = DistilBertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        elif model_name == 'RoBERTa':
            model = RobertaForMaskedLM.from_pretrained(model_path)
            tokenizer = RobertaTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        elif model_name == 'TinyBERT':
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        elif model_name == 'MobileBERT':
            model = MobileBertForMaskedLM.from_pretrained(model_path)
            tokenizer = MobileBertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
            # MobileBERT-specific settings
            print("Applying MobileBERT-specific configuration...")
            # Ensure model is in training mode
            model.train()
            # Initialize weights (optional)
            if hasattr(model, 'init_weights'):
                model.init_weights()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    else:
        # Fine-tuning classification model
        if model_name == 'ALBERT':
            model = AlbertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
            tokenizer = AlbertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        elif model_name == 'BERT':
            model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
            tokenizer = BertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        elif model_name == 'DistilBERT':
            model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
            tokenizer = DistilBertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        elif model_name == 'RoBERTa':
            model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
            tokenizer = RobertaTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        elif model_name == 'TinyBERT':
            model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
            tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        elif model_name == 'MobileBERT':  # Add MobileBERT support
            model = MobileBertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
            tokenizer = MobileBertTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded to device: {device}")
        
    return model, tokenizer

# Get model-specific task type and target modules
def get_model_specific_params(model_name):
    if model_name == 'ALBERT':
        task_type = TaskType.SEQ_CLS
        target_modules = ["albert.encoder.albert_layer_groups.0.albert_layers.0.attention.query", 
                          "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.key",
                          "albert.encoder.albert_layer_groups.0.albert_layers.0.attention.value"]
    elif model_name == 'BERT':
        task_type = TaskType.SEQ_CLS
        target_modules = ["query", "key", "value"]
    elif model_name == 'DistilBERT':
        task_type = TaskType.SEQ_CLS
        target_modules = ["attention.q_lin", "attention.k_lin", "attention.v_lin"]
    elif model_name == 'RoBERTa':
        task_type = TaskType.SEQ_CLS
        target_modules = ["query", "key", "value"]
    elif model_name == 'TinyBERT':
        task_type = TaskType.SEQ_CLS
        target_modules = ["query", "key", "value"]
    elif model_name == 'MobileBERT':
        task_type = TaskType.SEQ_CLS
        target_modules = ["query", "key", "value"]
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return task_type, target_modules

# Add LoRA configuration to model
def add_lora_to_model(model, model_name, lora_r, lora_alpha, lora_dropout):
    task_type, target_modules = get_model_specific_params(model_name)
    
    # Customize LoRA config for different models
    peft_config = LoraConfig(
        task_type=task_type,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=None
    )
    
    # Get model with LoRA applied
    lora_model = get_peft_model(model, peft_config)
    print(f"Added LoRA configuration to model, params: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    return lora_model

class PretrainingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Compute various evaluation metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Compute AUC-ROC (requires special handling for multi-class)
    # For multi-class, use one-vs-rest approach
    try:
        if len(np.unique(labels)) > 2:
            # Get raw prediction probabilities
            proba = pred.predictions
            
            # Check if values are already probabilities (sum to 1 along class dimension)
            # If not, apply softmax normalization
            proba_sums = np.sum(proba, axis=1)
            if not np.allclose(proba_sums, 1.0):
                # Apply softmax normalization
                proba = np.exp(proba) / np.sum(np.exp(proba), axis=1, keepdims=True)
                print("Probability sums after softmax normalization:", np.sum(proba, axis=1)[:5])  # Print first 5 samples' probability sums for verification

            # Compute macro average AUC
            auc = roc_auc_score(labels, proba, multi_class='ovr', average='macro')
        else:
            # Binary classification
            proba = pred.predictions[:, 1]  # Assume positive class is index 1
            auc = roc_auc_score(labels, proba)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'confusion_matrix': cm
        }
    except Exception as e:
        print(f"Error computing ROC AUC: {str(e)}")
        # Return results without AUC
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm
        }

def pretrain_model(x_train, y_train, model, tokenizer, output_dir, epochs=3, batch_size=16):
    # Build pre-training Dataset
    train_dataset = PretrainingDataset(x_train, y_train, tokenizer)
    
    
    # Set training arguments - add GPU-related settings
    training_args = TrainingArguments(
        output_dir=output_dir + '_pretrain_checkpoints',
        evaluation_strategy="no",
        save_strategy='epoch',
        logging_dir=output_dir + "_logs",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=1000,
        logging_steps=100,  # More frequent logging
        save_total_limit=3,
        learning_rate=5e-5,  # Use the learning rate set above
        weight_decay=0.01,
        warmup_steps=500,
        report_to="none",  
        # Add gradient clipping to prevent gradient explosion
        max_grad_norm=1.0,
        # Reduce gradient accumulation steps for more frequent updates
        gradient_accumulation_steps=2,
        # Other settings remain unchanged
        fp16=True,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        no_cuda=False,
    )

    # Initialize Trainer and print parameters
    trainer = PretrainingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Parameter validation
    print(f"Pre-training parameter validation:")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Training epochs: {training_args.num_train_epochs}")
    print(f"  - Optimizer: {training_args.optim}")
    print(f"  - Gradient clipping: {training_args.max_grad_norm}")

    # Set more suitable learning rate and training params for MobileBERT
    if 'mobilebert' in str(type(model)).lower():
        data_collator = MobileBertMLMCollator(tokenizer)
        trainer = PretrainingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
    else:
        # Use the original trainer
        trainer = PretrainingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
    # Start training
    print(f"Starting model pre-training...")
    trainer.train()

    # Save the trained model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Pre-training complete. Model saved to {output_dir}")
    
    return model, tokenizer

def finetune_model(x_train, y_train, x_test, y_test, pretrained_model_path, 
                   model_name, num_classes, output_dir, epochs=3, batch_size=16, 
                   use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    # Load pre-trained model for classification
    model, tokenizer = get_model_and_tokenizer(model_name, num_labels=num_classes, for_pretraining=False)
    
    # If a pre-trained model path is provided, load weights from it
    if pretrained_model_path:
        if os.path.exists(pretrained_model_path):
            print(f"Loading pre-trained weights from {pretrained_model_path}...")
            # Only load base model parts, excluding the classification head
            if model_name == "ALBERT":
                pt_model = AlbertForMaskedLM.from_pretrained(pretrained_model_path)
                model.albert = pt_model.albert
            elif model_name == "BERT":
                pt_model = BertForMaskedLM.from_pretrained(pretrained_model_path)
                model.bert = pt_model.bert
            elif model_name == "DistilBERT":
                pt_model = DistilBertForMaskedLM.from_pretrained(pretrained_model_path)
                model.distilbert = pt_model.distilbert
            elif model_name == "RoBERTa":
                pt_model = RobertaForMaskedLM.from_pretrained(pretrained_model_path)
                model.roberta = pt_model.roberta
            elif model_name == "TinyBERT":
                pt_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_path)
                # TinyBERT uses standard transformer architecture, may need extra handling
                if hasattr(pt_model, 'bert'):
                    model.bert = pt_model.bert
                elif hasattr(model, 'transformer'):
                    model.transformer = pt_model.transformer
            elif model_name == "MobileBERT":  # Add MobileBERT support
                pt_model = MobileBertForMaskedLM.from_pretrained(pretrained_model_path)
                model.mobilebert = pt_model.mobilebert
            else:
                print(f"Warning: Unrecognized model type {model_name}, using default pre-trained weights")
    
    # If using LoRA, apply LoRA configuration
    if use_lora:
        model = add_lora_to_model(model, model_name, lora_r, lora_alpha, lora_dropout)
    
    # Build training and test datasets
    train_dataset = FineTuningDataset(x_train, y_train, tokenizer)
    test_dataset = FineTuningDataset(x_test, y_test, tokenizer)
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir + '_finetune_checkpoints',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=output_dir + "_logs",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=100,
        save_total_limit=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",  # Disable all reporting tools
        # Additional parameters to optimize GPU usage
        fp16=True,                       # Use mixed precision training
        gradient_accumulation_steps=2,   # Gradient accumulation
        dataloader_num_workers=4,        # Number of data loader workers
        ddp_find_unused_parameters=False,# Distributed training optimization
        no_cuda=False,                   # Ensure CUDA is used
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Start fine-tuning
    print(f"Starting model fine-tuning...")
    trainer.train()
    
    # Evaluate model
    print("Evaluating model performance...")
    print("Evaluating model performance...")
    
    # Record evaluation start time
    eval_start_time = time.time()
    results = trainer.evaluate()
    eval_end_time = time.time()
    
    # Compute total time and processing speed
    total_eval_time = eval_end_time - eval_start_time
    total_samples = len(x_test)
    samples_per_second = total_samples / total_eval_time
    
    # Extract confusion matrix
    cm = results.pop('eval_confusion_matrix', None)
    
    # Print evaluation metrics
    print(f"\n========== {model_name} Fine-tuning Performance ==========")
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    print(f"F1 Score: {results['eval_f1']:.4f}")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall: {results['eval_recall']:.4f}")
    print(f"AUC-ROC: {results.get('eval_auc', 'N/A')}")
    print(f"\nPerformance statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Total evaluation time: {total_eval_time:.2f} seconds")
    print(f"Samples per second: {samples_per_second:.2f}")
    
    # Save evaluation results to JSON file
    performance_metrics = {
        'model_name': model_name,
        'accuracy': results['eval_accuracy'],
        'f1': results['eval_f1'],
        'precision': results['eval_precision'],
        'recall': results['eval_recall'],
        'auc': results.get('eval_auc', None),
        'total_samples': total_samples,
        'eval_time': total_eval_time,
        'samples_per_second': samples_per_second
    }
    
    # Save performance metrics
    with open(f"{output_dir}_performance.json", 'w') as f:
        json.dump(performance_metrics, f, indent=4)
    
    # If confusion matrix exists, plot and save it
    if cm is not None:
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{model_name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{output_dir}_confusion_matrix.png")
        plt.close()
        
        # Also save as CSV
        pd.DataFrame(cm).to_csv(f"{output_dir}_confusion_matrix.csv")
        print(f"Confusion matrix saved to {output_dir}_confusion_matrix.png and {output_dir}_confusion_matrix.csv")
    
    # Print evaluation metrics
    print(f"\n========== {model_name} Fine-tuning Performance ==========")
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    print(f"F1 Score: {results['eval_f1']:.4f}")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall: {results['eval_recall']:.4f}")
    
    # Save the fine-tuned model
    if use_lora:
        # If using LoRA, save both the full model and LoRA weights
        model.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)
        
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuning complete. Model saved to {output_dir}")
    
    return results

# Add custom Trainer class to catch training errors
class SafeTrainer(Trainer):
    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Attempting to continue with evaluation and model saving...")
            return None  # Return None instead of exception so subsequent code can continue

def load_and_finetune_model(x_train, y_train, x_test, y_test, model_name, dataset_name, 
                           num_classes, epochs=3, batch_size=16, use_lora=True, 
                           lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """Load an existing pre-trained model and perform fine-tuning"""
    
    # Validate input data
    print("Validating input data...")
    print(f"Training data shape: {x_train.shape}, label shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}, label shape: {y_test.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Validate that labels are within valid range
    train_labels_valid = all(0 <= label < num_classes for label in y_train)
    test_labels_valid = all(0 <= label < num_classes for label in y_test)
    
    if not train_labels_valid or not test_labels_valid:
        print("Warning: Found label values outside valid range!")
        # Compute label distribution
        train_label_counts = np.bincount(y_train)
        print(f"Training set label distribution: {train_label_counts}")
    
    # Build pre-trained model path
    pretrained_model_path = f'./plmd-model/{model_name}-{dataset_name}'
    
    # Check if pre-trained model exists
    if not os.path.exists(pretrained_model_path):
        raise ValueError(f"Error: Pre-trained model directory does not exist: {pretrained_model_path}")
    
    print(f"Loading pre-trained model: {pretrained_model_path}")
    
    # Safely check directory contents
    try:
        files = os.listdir(pretrained_model_path)
        print(f"Files in directory: {files}")
        
        # Check if necessary model files are present
        has_config = 'config.json' in files
        has_model = any(f in files for f in ['pytorch_model.bin', 'model.safetensors'])
        
        if not has_config:
            print("Warning: config.json file not found!")
        if not has_model:
            print("Warning: Model weight file not found!")
            
    except Exception as e:
        print(f"Warning: Could not list directory contents: {e}")
    
    # Determine fine-tuned model output directory
    lora_suffix = "_lora" if use_lora else ""
    finetune_dir = f'./plmd-model/{model_name}_{dataset_name}_only_ft{lora_suffix}'
    
    print(f"Fine-tuned model will be saved to: {finetune_dir}")
    
    # Load pre-trained model for classification
    try:
        # Try loading classification model directly from pre-trained directory
        try:
            print(f"Attempting to load classification model directly from pre-trained directory...")
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_path, 
                num_labels=num_classes,
                ignore_mismatched_sizes=True  # Allow mismatched sizes
            )
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
            print("Successfully loaded pre-trained classification model directly")
        except Exception as direct_load_error:
            print(f"Direct loading failed: {str(direct_load_error)}")
            print("Attempting to load classifier using base model path...")
            
            base_path = './base-model'
            model_paths = {
                'ALBERT': f'{base_path}/albert-base-v2',
                'BERT': f'{base_path}/bert-base-uncased',
                'DistilBERT': f'{base_path}/distilbert-base-uncased',
                'RoBERTa': f'{base_path}/roberta-base',
                'TinyBERT': f'{base_path}/TinyBERT_General_4L_312D',
                'MobileBERT': f'{base_path}/mobilebert-uncased'
            }
            
            base_model_path = model_paths.get(model_name, model_paths['ALBERT'])
            model, tokenizer = get_model_and_tokenizer(model_name, num_labels=num_classes, for_pretraining=False)
            print(f"Successfully loaded base classifier model")
            
            # Now try to load pre-trained weights
            print("Attempting to load pre-trained weights...")
            if model_name == "ALBERT":
                # Try loading ALBERT MLM model
                try:
                    pt_model = AlbertForMaskedLM.from_pretrained(pretrained_model_path)
                    print("Successfully loaded ALBERT MLM model")
                    # Get base model part
                    model.albert = pt_model.albert
                    print("Successfully copied pre-trained albert layers to classification model")
                except Exception as e:
                    print(f"Failed to load ALBERT MLM model: {str(e)}")
                    # Try loading weights file directly
                    model_file = os.path.join(pretrained_model_path, 'model.safetensors') 
                    if os.path.exists(model_file):
                        try:
                            # Load using safetensors
                            from safetensors.torch import load_file
                            state_dict = load_file(model_file)
                            # Filter keys containing 'albert.' but not 'cls'
                            filtered_dict = {k: v for k, v in state_dict.items() 
                                            if k.startswith('albert.') and 'cls' not in k}
                            # Load filtered weights
                            load_result = model.albert.load_state_dict(filtered_dict, strict=False)
                            print(f"Loaded weights from safetensors file: missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")
                        except Exception as e2:
                            print(f"Failed to load weights from safetensors file: {str(e2)}")
            # Handle other models similarly...
            elif model_name == "BERT":
                try:
                    pt_model = BertForMaskedLM.from_pretrained(pretrained_model_path)
                    model.bert = pt_model.bert
                    print("Successfully loaded BERT pre-trained weights")
                except Exception as e:
                    print(f"Error loading BERT pre-trained model: {e}")
            elif model_name == "DistilBERT":
                try:
                    pt_model = DistilBertForMaskedLM.from_pretrained(pretrained_model_path)
                    model.distilbert = pt_model.distilbert
                    print("Successfully loaded DistilBERT pre-trained weights")
                except Exception as e:
                    print(f"Error loading DistilBERT pre-trained model: {e}")
            elif model_name == "RoBERTa":
                try:
                    pt_model = RobertaForMaskedLM.from_pretrained(pretrained_model_path)
                    model.roberta = pt_model.roberta
                    print("Successfully loaded RoBERTa pre-trained weights")
                except Exception as e:
                    print(f"Error loading RoBERTa pre-trained model: {e}")
            elif model_name == "TinyBERT":
                try:
                    pt_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_path)
                    if hasattr(pt_model, 'bert'):
                        model.bert = pt_model.bert
                        print("Successfully loaded TinyBERT pre-trained weights (bert)")
                    elif hasattr(model, 'transformer'):
                        model.transformer = pt_model.transformer
                        print("Successfully loaded TinyBERT pre-trained weights (transformer)")
                    else:
                        print("Warning: Cannot determine TinyBERT model structure, pre-trained weights may not be loaded correctly")
                except Exception as e:
                    print(f"Error loading TinyBERT pre-trained model: {e}")
            elif model_name == "MobileBERT":
                try:
                    pt_model = MobileBertForMaskedLM.from_pretrained(pretrained_model_path)
                    model.mobilebert = pt_model.mobilebert
                    print("Successfully loaded MobileBERT pre-trained weights")
                except Exception as e:
                    print(f"Error loading MobileBERT pre-trained model: {e}")
                
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Model loaded to device: {device}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # If using LoRA, apply LoRA configuration
    if use_lora:
        try:
            model = add_lora_to_model(model, model_name, lora_r, lora_alpha, lora_dropout)
            print(f"Using LoRA for fine-tuning, params: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        except Exception as e:
            print(f"Error applying LoRA configuration: {e}")
            print("Continuing with full parameter fine-tuning")
            use_lora = False
    else:
        print("Using full parameter fine-tuning")
    
    # Build training and test datasets
    try:
        print("Creating training and test datasets...")
        # Check some feature samples
        print(f"Feature sample: {x_train[0][:100]}...")
        print(f"Label sample: {y_train[0]}")
        
        max_length = 128
        print(f"Using max sequence length: {max_length}")
        
        train_dataset = FineTuningDataset(x_train, y_train, tokenizer, max_length=max_length)
        test_dataset = FineTuningDataset(x_test, y_test, tokenizer, max_length=max_length)
        
        # Check the first training sample
        first_item = train_dataset[0]
        print(f"First training sample input_ids shape: {first_item['input_ids'].shape}")
        print(f"First training sample label: {first_item['labels']}")
        
    except Exception as e:
        print(f"Error creating datasets: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    # Set training arguments
    print("Setting training arguments...")
    training_args = TrainingArguments(
        output_dir=finetune_dir + '_checkpoints',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=finetune_dir + "_logs",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=100,
        save_total_limit=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",  # Disable all reporting tools
        # Additional parameters to optimize GPU usage
        fp16=True,                       # Use mixed precision training
        gradient_accumulation_steps=2,   # Gradient accumulation
        dataloader_num_workers=2,        # Number of data loader workers - reduced to prevent issues
        ddp_find_unused_parameters=False,# Distributed training optimization
        no_cuda=False,                   # Ensure CUDA is used
    )

    # Custom evaluation metrics callback
    def safe_compute_metrics(pred):
        try:
            return compute_metrics(pred)
        except Exception as e:
            print(f"Error computing evaluation metrics: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return basic metrics
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            accuracy = accuracy_score(labels, preds)
            try:
                f1 = f1_score(labels, preds, average='weighted')
                precision = precision_score(labels, preds, average='weighted') 
                recall = recall_score(labels, preds, average='weighted')
            except:
                f1 = precision = recall = 0.0
                
            return {
                'accuracy': accuracy, 
                'f1': f1, 
                'precision': precision, 
                'recall': recall
            }

    # Use the safe version when initializing Trainer
    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=safe_compute_metrics
    )

    # Start fine-tuning
    print(f"Starting model fine-tuning...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Attempting to continue with evaluation and model saving...")
    
    # Evaluate model
    print("Evaluating model performance...")
    try:
        # Record evaluation start time
        eval_start_time = time.time()
        results = trainer.evaluate()
        eval_end_time = time.time()
        
        # Compute total time and processing speed
        total_eval_time = eval_end_time - eval_start_time
        total_samples = len(x_test)
        samples_per_second = total_samples / total_eval_time
        
        # Safely extract confusion matrix
        cm = None
        if results and 'eval_confusion_matrix' in results:
            cm = results.pop('eval_confusion_matrix', None)
        
        # Safely get metric values with defaults
        accuracy = results.get('eval_accuracy', 0.0) if results else 0.0
        f1 = results.get('eval_f1', 0.0) if results else 0.0
        precision = results.get('eval_precision', 0.0) if results else 0.0
        recall = results.get('eval_recall', 0.0) if results else 0.0
        auc = results.get('eval_auc', 'N/A') if results else 'N/A'
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Set default results
        accuracy = 0.0
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        auc = 'N/A'
        total_eval_time = 0.0
        total_samples = len(x_test)
        samples_per_second = 0.0
        cm = None
        results = {}
    
    # Print evaluation metrics
    print(f"\n========== {model_name} Fine-tuning Performance ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC-ROC: {auc}")
    print(f"\nPerformance statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Total evaluation time: {total_eval_time:.2f} seconds")
    print(f"Samples per second: {samples_per_second:.2f}")
    
    # Save evaluation results to JSON file
    try:
        performance_metrics = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc if isinstance(auc, (int, float)) else None,
            'total_samples': total_samples,
            'eval_time': total_eval_time,
            'samples_per_second': samples_per_second
        }
        
        # Save performance metrics
        with open(f"{finetune_dir}_performance.json", 'w') as f:
            json.dump(performance_metrics, f, indent=4)
        print(f"Performance metrics saved to {finetune_dir}_performance.json")
    except Exception as e:
        print(f"Error saving performance metrics: {str(e)}")
    
    # If confusion matrix exists, plot and save it
    if cm is not None:
        try:
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{model_name} Confusion Matrix')
            plt.tight_layout()
            plt.savefig(f"{finetune_dir}_confusion_matrix.png")
            plt.close()
            
            # Also save as CSV
            pd.DataFrame(cm).to_csv(f"{finetune_dir}_confusion_matrix.csv")
            print(f"Confusion matrix saved to {finetune_dir}_confusion_matrix.png and {finetune_dir}_confusion_matrix.csv")
        except Exception as e:
            print(f"Error saving confusion matrix: {str(e)}")
    
    # Save the fine-tuned model
    try:
        print(f"Saving fine-tuned model to {finetune_dir}...")
        if use_lora:
            # If using LoRA, save both the full model and LoRA weights
            model.save_pretrained(finetune_dir)
        else:
            model.save_pretrained(finetune_dir)
            
        tokenizer.save_pretrained(finetune_dir)
        print(f"Fine-tuning complete. Model saved to {finetune_dir}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return results or {'eval_accuracy': accuracy, 'eval_f1': f1}

def evaluate_finetuned_model(model_path, data_path, model_name, batch_size=32):
    """
    Load a fine-tuned model and perform prediction and evaluation on test data

    Args:
        model_path: Path to the fine-tuned model
        data_path: Dataset path
        model_name: Model name (ALBERT, BERT, etc.)
        batch_size: Batch size

    Returns:
        evaluation_results: Evaluation results dictionary
    """
    print(f"Loading fine-tuned model {model_name} from {model_path}...")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Error: Fine-tuned model directory does not exist: {model_path}")
    
    # Get device info
    device = get_device()
    
    # Load data
    print(f"Loading data from {data_path}...")
    X, y = data_import(data_path)
    
    # Split data to get the test set
    _, x_test, _, y_test = split_data(X, y)
    
    # Compute number of classes
    num_classes = len(np.unique(y))
    print(f"Dataset contains {num_classes} classes")
    
    # Load the fine-tuned model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_classes
        )
        model.to(device)
        print(f"Successfully loaded model to {device}")
    except Exception as e:
        print(f"Error loading fine-tuned model: {str(e)}")
        
        # Try alternative loading method - from specific model class
        try:
            print("Attempting to load using specific model class...")
            
            if model_name == 'ALBERT':
                model = AlbertForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
                tokenizer = AlbertTokenizer.from_pretrained(model_path)
            elif model_name == 'BERT':
                model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
                tokenizer = BertTokenizer.from_pretrained(model_path)
            elif model_name == 'DistilBERT':
                model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
                tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            elif model_name == 'RoBERTa':
                model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
                tokenizer = RobertaTokenizer.from_pretrained(model_path)
            elif model_name == 'TinyBERT':
                model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            elif model_name == 'MobileBERT':
                model = MobileBertForSequenceClassification.from_pretrained(model_path, num_labels=num_classes)
                tokenizer = MobileBertTokenizer.from_pretrained(model_path)
            else:
                raise ValueError(f"Unsupported model type: {model_name}")
                
            model.to(device)
            print(f"Successfully loaded using specific model class to {device}")
        except Exception as e2:
            print(f"Loading with specific model class also failed: {str(e2)}")
            raise
    
    # Create test dataset
    test_dataset = FineTuningDataset(x_test, y_test, tokenizer)
    
    # Use DataLoader for batching
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Store predictions and labels
    all_preds = []
    all_labels = []
    all_probs = []  # Store probabilities
    
    # Start prediction
    print("Starting prediction...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Model forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get predictions
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute total prediction time
    prediction_time = time.time() - start_time
    samples_per_second = len(all_labels) / prediction_time
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Try to compute ROC AUC
    try:
        # Multi-class ROC AUC
        if num_classes > 2:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        else:
            # Binary classification
            auc = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception as e:
        print(f"Error computing ROC AUC: {str(e)}")
        auc = None
    
    # Print evaluation results
    print("\n" + "="*50)
    print(f"{model_name} Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    if auc is not None:
        print(f"ROC AUC: {auc:.4f}")
    print("\nPerformance statistics:")
    print(f"Total prediction samples: {len(all_labels)}")
    print(f"Total prediction time: {prediction_time:.2f} seconds")
    print(f"Samples per second: {samples_per_second:.2f}")
    print("="*50)
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} Confusion Matrix')
    
    # Save confusion matrix plot
    cm_filename = f"{model_path}_eval_confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()
    print(f"Confusion matrix saved to: {cm_filename}")
    
    # Save confusion matrix as CSV
    cm_csv = f"{model_path}_eval_confusion_matrix.csv"
    pd.DataFrame(cm).to_csv(cm_csv)
    print(f"Confusion matrix CSV saved to: {cm_csv}")
    
    # Save prediction results to CSV
    predictions_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds
    })
    
    # Add prediction probabilities for each class
    for i in range(num_classes):
        predictions_df[f'prob_class_{i}'] = all_probs[:, i]
    
    predictions_csv = f"{model_path}_predictions.csv"
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"Prediction results saved to: {predictions_csv}")
    
    # Save evaluation results
    evaluation_results = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'auc': float(auc) if auc is not None else None,
        'total_samples': int(len(all_labels)),
        'prediction_time': float(prediction_time),
        'samples_per_second': float(samples_per_second)
    }
    
    results_json = f"{model_path}_evaluation_results.json"
    with open(results_json, 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"Evaluation results saved to: {results_json}")
    
    return evaluation_results

def main_ft_eval():
    """
    Main function - Load and evaluate a fine-tuned model
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a fine-tuned model')
    
    parser.add_argument('-m', '--model', type=str, 
                      choices=['ALBERT', 'BERT', 'DistilBERT', 'RoBERTa', 'TinyBERT', 'MobileBERT'], 
                      default='ALBERT', help='Select model type')
                      
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to fine-tuned model')
                      
    parser.add_argument('--data_path', type=str, default='./data/USTC_concate_data.npz',
                      help='Dataset path')
                      
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    
    args = parser.parse_args()
    
    # Perform model evaluation
    try:
        evaluate_finetuned_model(
            model_path=args.model_path,
            data_path=args.data_path,
            model_name=args.model,
            batch_size=args.batch_size
        )
        print("Model evaluation complete!")
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Get device info
    device = get_device()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pre-training and fine-tuning models')
    parser.add_argument('-m', '--model', type=str,
                      choices=['ALBERT', 'BERT', 'DistilBERT', 'RoBERTa', 'TinyBERT', 'MobileBERT'],
                      default='ALBERT', help='Select model type (ALBERT, BERT, DistilBERT, RoBERTa, TinyBERT, MobileBERT)')
    parser.add_argument('--data_path', type=str, default='./data/USTC_concate_data.npz', 
                      help='Dataset path')
    parser.add_argument('--pretrain_epochs', type=int, default=3, 
                      help='Number of pre-training epochs')
    parser.add_argument('--finetune_epochs', type=int, default=3, 
                      help='Number of fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                      help='Batch size')
    parser.add_argument('--skip_pretrain', action='store_true',
                      help='Skip the pre-training stage')
    parser.add_argument('--only_pretrain', action='store_true',
                      help='Only run pre-training, skip fine-tuning')
    parser.add_argument('--only_ft', action='store_true',
                      help='Only run fine-tuning, load from existing pre-trained model')
    parser.add_argument('--no_lora', action='store_true',
                      help='Do not use LoRA for fine-tuning (LoRA is used by default)')
    parser.add_argument('--lora_r', type=int, default=8,
                      help='LoRA rank parameter')
    parser.add_argument('--lora_alpha', type=int, default=16,
                      help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                      help='LoRA dropout parameter')
    parser.add_argument('--force_pretrain', action='store_true',
                      help='Force re-pre-training even if a pre-trained model already exists')
    
    args = parser.parse_args()
    
    # Auto-adjust batch size
    if torch.cuda.is_available():
        # Get GPU memory info
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

        # Dynamically adjust batch size based on GPU memory
        if args.model in ['BERT', 'RoBERTa']:
            # Larger models need more memory
            if gpu_mem >= 16:  # For GPUs with 16GB+
                suggested_batch = min(32, args.batch_size)
            elif gpu_mem >= 8:  # For 8GB GPUs
                suggested_batch = min(16, args.batch_size)
            else:  # For smaller memory
                suggested_batch = min(8, args.batch_size)
        else:
            # Smaller models can use larger batches
            if gpu_mem >= 16:
                suggested_batch = min(64, args.batch_size)
            elif gpu_mem >= 8:
                suggested_batch = min(32, args.batch_size)
            else:
                suggested_batch = min(16, args.batch_size)
        
        # If suggested batch size differs from user setting, inform the user
        if suggested_batch != args.batch_size:
            print(f"Note: Based on GPU memory ({gpu_mem:.1f}GB), recommended batch size is {suggested_batch} instead of {args.batch_size}")
            print(f"Using recommended batch size: {suggested_batch}")
            args.batch_size = suggested_batch
    
    # Extract dataset name
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0].split('_')[0]
    
    # Load data
    X, y = data_import(args.data_path)
    
    # Split data
    x_train, x_test, y_train, y_test = split_data(X, y)

    # Compute number of classes
    num_classes = len(np.unique(y))
    print(f"Dataset contains {num_classes} classes")
    
    # Check for conflicting command line arguments
    if args.only_pretrain and args.only_ft:
        print("Error: Cannot specify both --only_pretrain and --only_ft")
        return
    
    if args.only_pretrain and args.skip_pretrain:
        print("Error: Cannot specify both --only_pretrain and --skip_pretrain")
        return
    
    # Fine-tuning only mode - use existing pre-trained model
    if args.only_ft:
        print(f"======== Fine-tuning only mode: Loading pre-trained {args.model} model ========")
        use_lora = not args.no_lora
        try:
            results = load_and_finetune_model(
                x_train, y_train, x_test, y_test,
                args.model, dataset_name, num_classes,
                epochs=args.finetune_epochs, batch_size=args.batch_size,
                use_lora=use_lora, lora_r=args.lora_r,
                lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
            )
            print(f"Fine-tuning only complete, model performance: Accuracy={results['eval_accuracy']:.4f}, F1={results['eval_f1']:.4f}")
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
        return
    
    # Define output directories - modify pre-training directory naming as required
    pretrain_dir = f'./plmd-model/{args.model}_{dataset_name}'
    
    # Fine-tuning output directory
    lora_suffix = "_lora" if not args.no_lora else ""
    if args.skip_pretrain:
        finetune_dir = f'./plmd-model/withoutPretrain_{args.model}_{dataset_name}_ft{lora_suffix}'
        print("Skipping pre-training, output directory:", finetune_dir)
    else:
        finetune_dir = f'./plmd-model/{args.model}_{dataset_name}_full_ft{lora_suffix}'
        if args.no_lora:
            print(f"With pre-training, using full parameter fine-tuning, output directory: {finetune_dir}")
        else:
            print(f"With pre-training, using LoRA fine-tuning, output directory: {finetune_dir}")
    
    # Check if a pre-trained model already exists
    pretrain_exists, pretrain_path = check_pretrained_model_exists(args.model, dataset_name)
    
    # Pre-training stage
    if not args.skip_pretrain:
        if pretrain_exists and not args.force_pretrain:
            print(f"Found existing pre-trained model: {pretrain_path}")
            print(f"Will use it for fine-tuning directly (use --force_pretrain to force re-pre-training)")
            pretrained_model_path = pretrain_path
        else:
            if pretrain_exists and args.force_pretrain:
                print(f"Found existing pre-trained model, but re-pre-training due to --force_pretrain")
            
            print(f"======== Starting {args.model} pre-training ========")
            # Get pre-training model and tokenizer
            model, tokenizer = get_model_and_tokenizer(args.model, for_pretraining=True)
            
            # Pre-training
            pretrained_model, pretrained_tokenizer = pretrain_model(
                x_train, y_train, model, tokenizer, 
                pretrain_dir, epochs=args.pretrain_epochs, batch_size=args.batch_size
            )
            pretrained_model_path = pretrain_dir
        
        # If only pre-training, exit
        if args.only_pretrain:
            print(f"Pre-training only stage complete, model saved to {pretrain_dir}")
            return
    else:
        print("Skipping pre-training stage")
        pretrained_model_path = None
    
    # Fine-tuning stage
    print(f"======== Starting {args.model} fine-tuning ========")
    use_lora = not args.no_lora
    if use_lora:
        print(f"Using LoRA fine-tuning, params: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    else:
        print("Not using LoRA, performing full parameter fine-tuning")
        
    results = finetune_model(
        x_train, y_train, x_test, y_test,
        pretrained_model_path, args.model, num_classes,
        finetune_dir, epochs=args.finetune_epochs, batch_size=args.batch_size,
        use_lora=use_lora,
        lora_r=args.lora_r, 
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and "--model_path" in sys.argv:
        # If --model_path argument is present, run evaluation mode
        main_ft_eval()
    else:
        # Otherwise run normal training mode
        main()
    

# Task 1:
# # Use ALBERT model for pre-training only, skip fine-tuning
# python s41_full_ft.py -m BERT --only_pretrain


# Task 2: Use ALBERT model for pre-training and full fine-tuning
# python s41_full_ft.py -m ALBERT --only-ft --no_loara

# # Use ALBERT model with LoRA fine-tuning
# python script.py -m ALBERT --lora_r 16 --lora_alpha 32 --lora_dropout 0.05

# Evaluation mode
# Evaluate a fine-tuned model
# python s41_full_ft.py --model_path ./plmd-model/ALBERT_USTC_only_ft --model ALBERT --data_path ./data/USTC_concate_data.npz --batch_size 32

# # Use RoBERTa model
# python script.py -m RoBERTa

# # Use TinyBERT model and skip pre-training
# python script.py -m TinyBERT --skip_pretrain

# # Custom data path and parameters
# python script.py -m BERT --data_path ./my_data.npz --pretrain_epochs 5 --finetune_epochs 3 --batch_size 16 --use_lora

# # Fine-tuning only mode (with LoRA)
# python s41_full_ft.py -m ALBERT --only_ft

# # Fine-tuning only mode (full parameter fine-tuning)
# python s41_full_ft.py -m BERT --only_ft --no_lora

# # Fine-tuning only mode (custom LoRA parameters)
# python s41_full_ft.py -m RoBERTa --only_ft --lora_r 16 --lora_alpha 32 --lora_dropout 0.05