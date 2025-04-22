import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                           precision_recall_curve, confusion_matrix, roc_curve, auc)
import time, csv, logging, wandb, gc, os
from transformers import (Trainer, TrainingArguments, get_linear_schedule_with_warmup)
from transformers import GenericTokenizer, GenericTextModel
from peft import get_peft_model, AdapterConfig, TaskType
from param_config import config, training_args_
from AL_strategy import get_strategy, AdvancedSamplingMethod, CustomDataset
from al_cache import get_cache_key, save_strategy_state, load_strategy_state
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Tuple, List
import psutil
os.environ["WANDB_MODE"] = "disabled"

logging.basicConfig(
    level=getattr(logging, config.log.level),
    format=config.log.format,
    filename=config.log.log_file
)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, num_labels):
        if len(texts) != len(labels):
            raise ValueError(f"Texts length ({len(texts)}) does not match labels length ({len(labels)})")
        
        texts = [str(text) if not isinstance(text, str) else text for text in texts]
        
        unique_labels = np.unique(labels)
        self.num_labels = num_labels
        
        print(f"Unique labels in current batch: {unique_labels}")
        print(f"Expected number of labels: {self.num_labels}")
        
        if len(unique_labels) > self.num_labels:
            raise ValueError(f"Found {len(unique_labels)} classes, but model expects {self.num_labels} classes")
        
        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print(f"Label mapping for current batch: {label_mapping}")
        
        self.labels = torch.tensor([label_mapping[label] for label in labels], dtype=torch.long)
        
        if self.labels.min() < 0 or self.labels.max() >= self.num_labels:
            raise ValueError(f"Labels must be in range [0, {self.num_labels-1}], "
                           f"got range [{self.labels.min()}, {self.labels.max()}]")
        
        self.encodings = self._parallel_encode(texts, tokenizer, max_length=config.data.max_length)

    def _parallel_encode(self, texts, tokenizer, max_length=128, batch_size=128):
        
        num_workers = min(4, os.cpu_count() - 1) if os.cpu_count() > 1 else 0
        
        all_input_ids = []
        all_attention_masks = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                futures.append(executor.submit(
                    self._encode_batch, 
                    batch_texts, 
                    tokenizer, 
                    max_length
                ))
            
            for future in tqdm(futures, total=len(futures), desc="Encoding texts"):
                batch_input_ids, batch_attention_masks = future.result()
                all_input_ids.extend(batch_input_ids)
                all_attention_masks.extend(batch_attention_masks)
        
        return {
            'input_ids': torch.stack(all_input_ids),
            'attention_mask': torch.stack(all_attention_masks)
        }
    
    def _encode_batch(self, texts, tokenizer, max_length):
        try:
            encodings = tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            return encodings['input_ids'], encodings['attention_mask']
        except Exception as e:
            print(f"Error encoding batch: {str(e)}")
            print(f"Problematic texts: {texts[:2]}...")
            raise

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)

class CachedDataset:
    def __init__(self):
        self.cache = {}
    
    def get_or_create(self, texts, labels, tokenizer, key='test', num_labels=None):
        cache_key = f"{key}_{len(texts)}"
        
        if cache_key not in self.cache:
            if num_labels is None:
                num_labels = len(np.unique(labels))
                print(f"Automatically determined number of labels for {key} set: {num_labels}")
            
            print(f"Creating new dataset for {key} split with {len(texts)} samples")
            self.cache[cache_key] = TextDataset(texts, labels, tokenizer, num_labels)
            
        return self.cache[cache_key]
    
    def clear(self):
        self.cache.clear()

def validate_dataset(dataset):
    try:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
            
        sample_size = min(100, len(dataset))
        sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        for idx in sample_indices:
            item = dataset[idx]
            
            required_fields = ['input_ids', 'attention_mask', 'labels']
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing required field '{field}' at index {idx}")
            
            if not isinstance(item['input_ids'], torch.Tensor):
                raise ValueError(f"input_ids at index {idx} is not a torch.Tensor")
            if not isinstance(item['attention_mask'], torch.Tensor):
                raise ValueError(f"attention_mask at index {idx} is not a torch.Tensor")
            if not isinstance(item['labels'], torch.Tensor):
                raise ValueError(f"labels at index {idx} is not a torch.Tensor")
            
            if item['input_ids'].shape != item['attention_mask'].shape:
                raise ValueError(f"Shape mismatch at index {idx}")
            
            if item['labels'].min() < 0:
                raise ValueError(f"Negative label found at index {idx}")
            
            if item['input_ids'].shape[0] > config.data.max_length:
                raise ValueError(f"Input sequence length exceeds max_length at index {idx}")
        
        logging.info("Dataset validation passed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Dataset validation failed: {str(e)}")
        raise

def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params
    }

def calculate_inference_speed(model, test_dataset, device):
    model.eval()
    batch_size = 32
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
    end_time = time.time()
    
    total_samples = len(test_dataset)
    total_time = end_time - start_time
    samples_per_second = total_samples / total_time
    
    return samples_per_second

def save_pr_data(pr_data, save_path, iteration):
    if pr_data is None or not isinstance(pr_data, list):
        print(f"Warning: No valid PR data to save for iteration {iteration}")
        return
        
    try:
        all_data = []
        for class_data in pr_data:
            if isinstance(class_data, pd.DataFrame):
                class_data['iteration'] = iteration
                all_data.append(class_data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            combined_data.to_csv(save_path, index=False)
            print(f"PR curve data saved to {save_path}")
        else:
            print(f"Warning: No valid PR data to save for iteration {iteration}")
            
    except Exception as e:
        print(f"Error saving PR curve data: {str(e)}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
        return cm
    except Exception as e:
        print(f"Error saving confusion matrix: {str(e)}")
        return confusion_matrix(y_true, y_pred) if y_true is not None and y_pred is not None else None

def plot_pr_curve(y_true, y_pred_proba, save_path):
    n_classes = y_pred_proba.shape[1]
    plt.figure(figsize=(12, 8))
    
    curves_data = []
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        try:
            precision, recall, thresholds = precision_recall_curve(y_true_binary, y_score)
            
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'Class {i} (AUC = {pr_auc:.2f})')
            class_data = pd.DataFrame({
                'class': i,
                'precision': precision,
                'recall': recall,
                'thresholds': np.append(thresholds, thresholds[-1])
            })
            curves_data.append(class_data)
            
        except Exception as e:
            print(f"Error calculating PR curve for class {i}: {str(e)}")
            continue
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (One-vs-Rest)')
    plt.legend(loc='best')
    plt.grid(True)

    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error saving PR curve plot: {str(e)}")
    finally:
        plt.close()
    
    return curves_data

def plot_roc_curve(y_true, y_pred_proba, save_path):
    n_classes = y_pred_proba.shape[1]
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    plt.figure(figsize=(12, 8))
    
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(
            fpr[i], 
            tpr[i], 
            lw=2, 
            label=f'Class {i} (AUC = {roc_auc[i]:.2f})'
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    }

def compute_detailed_metrics(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    n_classes = y_pred_proba.shape[1]
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    precision_curve = {}
    recall_curve = {}
    pr_auc = {}
    
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        precision_curve[i], recall_curve[i], _ = precision_recall_curve(y_true_binary, y_score)
        pr_auc[i] = auc(recall_curve[i], precision_curve[i])
    
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'roc_curve_data': {
            'fpr': fpr,
            'tpr': tpr
        },
        'pr_curve_data': {
            'precision': precision_curve,
            'recall': recall_curve
        }
    }
    
    return results

def evaluate_model(model, test_dataset, device, save_dir, iteration, save_results=False):
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []
    
    dataloader = DataLoader(test_dataset, batch_size=32)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            
            probabilities.extend(probs.cpu().numpy())
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
    
    probabilities = np.array(probabilities)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    detailed_metrics = compute_detailed_metrics(true_labels, predictions, probabilities)
    
    if save_results:
        is_final = (str(iteration).lower() == 'final' or iteration == 'final')
        save_suffix = 'final' if is_final else f'iter_{iteration + 1}'
        
        cm_path = os.path.join(save_dir, f'confusion_matrix_{save_suffix}.png')
        plot_confusion_matrix(true_labels, predictions, cm_path)
        
        pr_path = os.path.join(save_dir, f'pr_curve_{save_suffix}.png')
        pr_data = plot_pr_curve(true_labels, probabilities, pr_path)
        
        roc_path = os.path.join(save_dir, f'roc_curve_{save_suffix}.png')
        roc_data = plot_roc_curve(true_labels, probabilities, roc_path)
        
        if not is_final:
            pr_data_path = os.path.join(save_dir, f'pr_curve_iter_{iteration + 1}_data.csv')
            save_pr_data(pr_data, pr_data_path, iteration)
        else:
            pr_data_path = os.path.join(save_dir, f'pr_curve_final_data.csv')
            save_pr_data(pr_data, pr_data_path, "final")
    
    inference_speed = calculate_inference_speed(model, test_dataset, device)
    
    results = detailed_metrics.copy()
    results['inference_speed'] = inference_speed
    results['model_parameters'] = count_model_parameters(model)
    
    return results

def setup_model(model_path, num_labels):
    from param_config import baseModel
    print(f"Setting up model with {num_labels} labels...")
    
    model_type = baseModel.lower()
    print(f"Using model type: {model_type}")
    
    tokenizer = GenericTokenizer.from_pretrained(model_path, local_files_only=True)
    model = GenericTextModel.from_pretrained(
        model_path,
        num_labels=num_labels,
        local_files_only=True
    )
    
    adapter_config = AdapterConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.adapt.r,
        adapt_alpha=config.adapt.adapt_alpha,
        adapt_dropout=config.adapt.adapt_dropout,
        target_modules=config.adapt.target_modules,
        bias=config.adapt.bias,
        modules_to_save=config.adapt.modules_to_save
    )
    
    model = get_peft_model(model, adapter_config)
    print(f"Model initialized with {num_labels} output classes")
    return model, tokenizer

def data_collator(data):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in data]),
        'attention_mask': torch.stack([x['attention_mask'] for x in data]),
        'labels': torch.stack([x['labels'] for x in data])
    }

def train_model(model, train_dataset, training_args):
    try:
        validate_dataset(train_dataset)
        
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, 
            [train_size, val_size]
        )
        
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            suggested_batch_size = min(16, max(1, int(gpu_mem / (1024**3) * 2)))
            
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            logging.info(f"Initial GPU memory usage: {initial_memory/1024**2:.2f} MB")
        else:
            suggested_batch_size = 8
            
        training_args.per_device_train_batch_size = min(
            training_args.per_device_train_batch_size,
            suggested_batch_size
        )
        
        effective_batch_size = 32
        training_args.gradient_accumulation_steps = max(1, effective_batch_size // training_args.per_device_train_batch_size)
        
        training_args.learning_rate = 1e-5
        training_args.warmup_ratio = 0.1
        
        training_args.fp16 = False
        training_args.bf16 = False
        
        training_args.logging_steps = 50
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = 100
        training_args.save_strategy = "steps"
        training_args.save_steps = 100
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_loss"
        
        num_workers = min(4, os.cpu_count() - 1) if os.cpu_count() > 1 else 0
        training_args.dataloader_num_workers = num_workers
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=0.01,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        num_training_steps = (
            len(train_subset) * training_args.num_train_epochs // 
            (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        )
        num_warmup_steps = int(num_training_steps * training_args.warmup_ratio)
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            optimizers=(optimizer, lr_scheduler),
            data_collator=data_collator,
        )
        
        with torch.no_grad():
            sample_batch = next(iter(DataLoader(train_subset, batch_size=1)))
            sample_output = model(**{k: v.to(device) for k, v in sample_batch.items()})
            if not torch.isfinite(sample_output.loss):
                raise ValueError("Model produces invalid loss before training")
        
        trainer.train()
        
        return model
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            logging.error("GPU out of memory error occurred during training")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logging.error(f"GPU {i} memory: {torch.cuda.memory_allocated(i)/1024**2:.2f}MB allocated")
        else:
            logging.error(f"Runtime error during training: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as cleanup_error:
            logging.error(f"Error during cleanup: {str(cleanup_error)}")

def save_detailed_results(config, results_history, total_time, al_strategy, save_dir, filename=None):
    if filename is None:
        adapt_suffix = "" if config.adapt.use_adapt else "-full-ft"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results_{config.al.strategy}{adapt_suffix}_{timestamp}.xlsx'
    
    results_path = os.path.join(save_dir, filename)

    try:
        required_fields = [
            'accuracy_list', 'precision_list', 'f1_list', 'recall_list', 
            'roc_auc_list', 'pr_auc_list', 'time_list', 'prediction_time_list',
            'parameter_counts', 'confusion_matrices'
        ]
        
        for field in required_fields:
            print(f"{field}: {len(results_history.get(field, []))}")
        
        lengths = [len(results_history.get(field, [])) for field in required_fields if field in results_history]
        min_length = min(lengths) if lengths else 0
        
        print(f"\nUsing minimum length across all metrics: {min_length}")
        
        main_data = {
            'Iteration': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
            'ROC_AUC_Macro': [],
            'PR_AUC_Macro': [],
            'Training_Time(s)': [],
            'Prediction_Time(s)': [],
            'Inference_Speed': [],
            'Total_Parameters': [],
            'Trainable_Parameters': []
        }

        for i in range(min_length):
            main_data['Iteration'].append(i + 1)
            
            try:
                acc = float(results_history['accuracy_list'][i])
                main_data['Accuracy'].append(f"{acc:.2f}")
            except (ValueError, TypeError, IndexError):
                main_data['Accuracy'].append("N/A")
                
            try:
                prec = float(results_history['precision_list'][i])
                main_data['Precision'].append(f"{prec:.2f}")
            except (ValueError, TypeError, IndexError):
                main_data['Precision'].append("N/A")
                
            try:
                rec = float(results_history['recall_list'][i])
                main_data['Recall'].append(f"{rec:.2f}")
            except (ValueError, TypeError, IndexError):
                main_data['Recall'].append("N/A")
                
            try:
                f1 = float(results_history['f1_list'][i])
                main_data['F1'].append(f"{f1:.2f}")
            except (ValueError, TypeError, IndexError):
                main_data['F1'].append("N/A")
            
            if 'roc_auc_list' in results_history and i < len(results_history['roc_auc_list']):
                try:
                    roc_values = [float(v) for v in results_history['roc_auc_list'][i].values()]
                    macro_roc_auc = np.mean(roc_values) if roc_values else 0
                    main_data['ROC_AUC_Macro'].append(f"{macro_roc_auc:.4f}")
                except (AttributeError, ValueError, TypeError):
                    main_data['ROC_AUC_Macro'].append("N/A")
            else:
                main_data['ROC_AUC_Macro'].append("N/A")
                
            if 'pr_auc_list' in results_history and i < len(results_history['pr_auc_list']):
                try:
                    pr_values = [float(v) for v in results_history['pr_auc_list'][i].values()]
                    macro_pr_auc = np.mean(pr_values) if pr_values else 0
                    main_data['PR_AUC_Macro'].append(f"{macro_pr_auc:.4f}")
                except (AttributeError, ValueError, TypeError):
                    main_data['PR_AUC_Macro'].append("N/A")
            else:
                main_data['PR_AUC_Macro'].append("N/A")
                
            try:
                train_time = float(results_history['time_list'][i])
                main_data['Training_Time(s)'].append(f"{train_time:.2f}")
            except (ValueError, TypeError, IndexError):
                main_data['Training_Time(s)'].append("N/A")
            
            if 'prediction_time_list' in results_history and i < len(results_history['prediction_time_list']):
                try:
                    pred_time = float(results_history['prediction_time_list'][i])
                    main_data['Prediction_Time(s)'].append(f"{pred_time:.2f}")
                except (ValueError, TypeError):
                    main_data['Prediction_Time(s)'].append("N/A")
            else:
                main_data['Prediction_Time(s)'].append("N/A")
                
            if 'inference_speed_list' in results_history and i < len(results_history['inference_speed_list']):
                try:
                    inf_speed = float(results_history['inference_speed_list'][i])
                    main_data['Inference_Speed'].append(f"{inf_speed:.2f}")
                except (ValueError, TypeError):
                    main_data['Inference_Speed'].append("N/A")
            else:
                main_data['Inference_Speed'].append("N/A")
            
            if i < len(results_history['parameter_counts']):
                try:
                    total_params = int(results_history['parameter_counts'][i]['total_params'])
                    trainable_params = int(results_history['parameter_counts'][i]['trainable_params'])
                    main_data['Total_Parameters'].append(total_params)
                    main_data['Trainable_Parameters'].append(trainable_params)
                except (KeyError, ValueError, TypeError, IndexError):
                    main_data['Total_Parameters'].append(0)
                    main_data['Trainable_Parameters'].append(0)
            else:
                main_data['Total_Parameters'].append(0)
                main_data['Trainable_Parameters'].append(0)

        df_main = pd.DataFrame(main_data)

        config_data = {
            'Parameter': ['Strategy', 'Initial Samples', 'Query Size', 'Total Time (s)'],
            'Value': [config.al.strategy, config.al.initial_labeled_samples, 
                     config.al.query_size, total_time]
        }
        df_config = pd.DataFrame(config_data)

        if results_history.get('confusion_matrices', []) and len(results_history['confusion_matrices']) > 0:
            try:
                final_confusion_matrix = results_history['confusion_matrices'][-1]
                if isinstance(final_confusion_matrix, np.ndarray) and final_confusion_matrix.size > 0:
                    df_confusion = pd.DataFrame(
                        final_confusion_matrix,
                        index=[f'True Class {i}' for i in range(len(final_confusion_matrix))],
                        columns=[f'Predicted Class {i}' for i in range(len(final_confusion_matrix))]
                    )
                else:
                    df_confusion = pd.DataFrame()
            except Exception as e:
                print(f"Error creating confusion matrix DataFrame: {str(e)}")
                df_confusion = pd.DataFrame()
        else:
            df_confusion = pd.DataFrame()
            
        roc_data = []
        pr_data = []
        
        if 'roc_curve_data' in results_history and results_history['roc_curve_data']:
            try:
                final_roc_data = results_history['roc_curve_data'][-1]
                
                for class_idx in final_roc_data['fpr'].keys():
                    for i in range(len(final_roc_data['fpr'][class_idx])):
                        try:
                            roc_data.append({
                                'Class': int(class_idx),
                                'FPR': float(final_roc_data['fpr'][class_idx][i]),
                                'TPR': float(final_roc_data['tpr'][class_idx][i])
                            })
                        except (ValueError, TypeError):
                            continue
            except Exception as e:
                print(f"Error processing ROC curve data: {str(e)}")
        
        if 'pr_curve_data' in results_history and results_history['pr_curve_data']:
            try:
                final_pr_data = results_history['pr_curve_data'][-1]
                
                for class_idx in final_pr_data['precision'].keys():
                    for i in range(min(len(final_pr_data['precision'][class_idx]), len(final_pr_data['recall'][class_idx]))):
                        try:
                            pr_data.append({
                                'Class': int(class_idx),
                                'Precision': float(final_pr_data['precision'][class_idx][i]),
                                'Recall': float(final_pr_data['recall'][class_idx][i])
                            })
                        except (ValueError, TypeError):
                            continue
            except Exception as e:
                print(f"Error processing PR curve data: {str(e)}")
        
        df_roc = pd.DataFrame(roc_data)
        df_pr = pd.DataFrame(pr_data)

        with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
            df_config.to_excel(writer, sheet_name='results', startrow=0, index=False)
            df_main.to_excel(writer, sheet_name='results', startrow=len(df_config)+2, index=False)
            
            if not df_confusion.empty:
                df_confusion.to_excel(writer, sheet_name='confusion_matrix')
            else:
                pd.DataFrame({'Message': ['No valid confusion matrix data available']}).to_excel(
                    writer, sheet_name='confusion_matrix')
            
            if not df_roc.empty:
                df_roc.to_excel(writer, sheet_name='roc_curve_data', index=False)
            else:
                pd.DataFrame({'Message': ['No valid ROC curve data available']}).to_excel(
                    writer, sheet_name='roc_curve_data')
                
            if not df_pr.empty:
                df_pr.to_excel(writer, sheet_name='pr_curve_data', index=False)
            else:
                pd.DataFrame({'Message': ['No valid PR curve data available']}).to_excel(
                    writer, sheet_name='pr_curve_data')

        logging.info(f"Detailed results saved to {results_path}")
        print(f"Detailed results successfully saved to {results_path}")
        
    except Exception as e:
        logging.error(f"Error saving detailed results: {str(e)}")
        print(f"Error saving detailed results: {str(e)}")
        import traceback
        traceback.print_exc()
        
        try:
            simplified_data = {
                'Iteration': list(range(1, min_length + 1)),
                'Accuracy': results_history.get('accuracy_list', ['N/A'] * min_length)[:min_length],
                'Training_Time': results_history.get('time_list', ['N/A'] * min_length)[:min_length]
            }
            
            simplified_df = pd.DataFrame(simplified_data)
            simplified_path = os.path.join(save_dir, f'simplified_{filename}')
            simplified_df.to_csv(simplified_path, index=False)
            
            print(f"Saved simplified results to {simplified_path}")
        except Exception as backup_error:
            logging.error(f"Failed to save simplified results: {str(backup_error)}")


def balanced_sample_data(features, labels, samples_per_class=1000):
    try:
        features = np.array([str(x) for x in features])
        labels = np.array(labels)
        
        unique_classes = np.unique(labels)
        print(f"Found {len(unique_classes)} unique classes: {unique_classes}")
        
        selected_features = []
        selected_labels = []
        
        for class_label in unique_classes:
            class_indices = np.where(labels == class_label)[0]
            print(f"Class {class_label}: total {len(class_indices)} samples")
            
            if len(class_indices) < samples_per_class:
                print(f"Warning: Class {class_label} has only {len(class_indices)} samples, "
                      f"less than requested {samples_per_class}")
                selected_indices = class_indices
            else:
                selected_indices = np.random.choice(
                    class_indices, 
                    size=samples_per_class, 
                    replace=False
                )
            
            selected_features.extend(features[selected_indices])
            selected_labels.extend(labels[selected_indices])
            
            print(f"Selected {len(selected_indices)} samples for class {class_label}")
        
        selected_features = np.array(selected_features)
        selected_labels = np.array(selected_labels)
        
        shuffle_idx = np.random.permutation(len(selected_labels))
        selected_features = selected_features[shuffle_idx]
        selected_labels = selected_labels[shuffle_idx]
        
        print("\nFinal dataset statistics:")
        print(f"Total samples: {len(selected_labels)}")
        for class_label in unique_classes:
            class_count = np.sum(selected_labels == class_label)
            print(f"Class {class_label}: {class_count} samples")
        
        return selected_features, selected_labels
        
    except Exception as e:
        logging.error(f"Error in balanced_sample_data: {str(e)}")
        raise

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


def main():
    try:
        required_paths = [
            config.data.data_path,
            config.model.model_path,
            config.model.output_dir, 
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required path not found: {path}")
        
        wandb.init(
            project=config.training.wandb_project,
            config=config,
            name=f"{config.al.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        adapt_suffix = "" if config.adapt.use_adapt else "-full-ft"
        output_dir_base = os.path.join(config.model.output_dir)
        if not config.adapt.use_adapt and "-full-ft" not in output_dir_base:
            output_dir_base += "-full-ft"
            
        os.makedirs(config.model.output_dir, exist_ok=True)
        results_dir = os.path.join(config.model.output_dir, 'results')
        plots_dir = os.path.join(config.model.output_dir, 'plots')
        models_dir = os.path.join(config.model.output_dir, 'models')
        index_dir = os.path.join(config.model.output_dir, 'index')
        for d in [results_dir, plots_dir, models_dir]:
            os.makedirs(d, exist_ok=True)

        dataset_cache = CachedDataset()
        
        print("Loading data...")
        try:
            file_size = os.path.getsize(config.data.data_path)
            print(f"Data file size: {file_size / (1024**2):.2f} MB")
            
            with np.load(config.data.data_path, mmap_mode='r') as data:
                n_samples = len(data['features'])
                feature_shape = data['features'][0].shape if len(data['features']) > 0 else (0,)
                estimated_memory = estimate_memory_usage(n_samples, np.prod(feature_shape))
                print(f"Estimated memory requirement: {estimated_memory:.2f} GB")
            
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            print(f"Available system memory: {available_memory:.2f} GB")
            
            if estimated_memory < available_memory * 0.7:
                X, y = parallel_load_data(
                    config.data.data_path,
                    n_jobs=max(1, psutil.cpu_count(logical=False) - 1)
                )
            else:
                chunk_size = get_optimal_chunk_size(n_samples, np.prod(feature_shape))
                print(f"Processing data in chunks of {chunk_size} samples")
                
                X = []
                y = []
                with np.load(config.data.data_path, mmap_mode='r') as data:
                    for i in tqdm(range(0, n_samples, chunk_size), desc="Loading data chunks"):
                        end = min(i + chunk_size, n_samples)
                        chunk_features = process_chunk(data['features'][i:end])
                        X.extend(chunk_features)
                        y.extend(data['labels'][i:end])
                
                X = np.array(X)
                y = np.array(y, dtype=np.int64)
            
            print("\nValidating loaded data...")
            print(f"Total samples loaded: {len(y)}")
            print(f"Feature array shape: {X.shape}")
            print(f"Label array shape: {y.shape}")
            print(f"Number of unique labels: {len(np.unique(y))}")
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
            
        gc.collect()

        print("Splitting data...")
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.al.test_size,
            stratify=y,
            random_state=config.al.random_seed
        )
        del X, y
        gc.collect()
        
        unique_labels = np.unique(y_train)
        num_labels = len(unique_labels)
        print(f"\nTotal unique labels: {sorted(unique_labels)}")
        print(f"Number of labels: {num_labels}")
        
        print(f"Training set size: {len(x_train)}")
        print(f"Test set size: {len(x_test)}")
        print(f"Initial labeled samples: {config.al.initial_labeled_samples}")
        print(f"Query size per iteration: {config.al.query_size}")
        
        if config.al.initial_labeled_samples + config.al.query_size > len(x_train):
            raise ValueError("Initial labeled samples + query size exceeds total samples")

        print("Initializing model and strategy...")
        model, tokenizer = setup_model(config.model.model_path, num_labels)
        device = torch.device(config.device)
        model = model.to(device)
        
        strategy_params = {
            'batch_size': getattr(config.al, 'batch_size', 128),
            'device': device
        }
        

        if config.al.strategy == 'boundary':
            strategy_params.update({
                'buffer_size': config.al.buffer_size,
                'memory_size': config.al.memory_size,
                'param1': config.al.tau,
                'param2': config.al.eps,
                'param3': config.al.alpha,
                'param4': config.al.beta,
                'param5': config.al.gamma,
                'temp': config.al.temperature
            })
            
        elif config.al.strategy in ['strategy1', 'strategy2']:
            strategy_params.update({
                'eps': getattr(config.al, 'eps', 0.5),
                'min_samples': getattr(config.al, 'min_samples', 5)
            })
        elif config.al.strategy == 'strategy3':
            strategy_params['k_neighbors'] = getattr(config.al, 'k_neighbors', 5)

        elif config.al.strategy == 'committee':
            strategy_params.update({
                'n_committees': getattr(config.al, 'n_committees', 3),
                'dropout_rate': getattr(config.al, 'dropout_rate', 0.2),
                'model_path': config.model.model_path
            })

        al_strategy = get_strategy(config.al.strategy, model, tokenizer, **strategy_params)

        results_history = {
            'accuracy_list': [],
            'precision_list': [],
            'f1_list': [],
            'recall_list': [],
            'time_list': [],
            'prediction_time_list': [],
            'inference_speed_list': [],
            'parameter_counts': [],
            'confusion_matrices': [],
            'roc_auc_list': [],
            'pr_auc_list': [],
            'roc_curve_data': [],
            'pr_curve_data': []
        }

        print("Selecting initial samples...")
        total_samples = len(x_train)
        np.random.seed(config.al.random_seed)
        labeled_indices = list(np.random.choice(
            total_samples,
            size=config.al.initial_labeled_samples,
            replace=False
        ))
        unlabeled_indices = list(set(range(total_samples)) - set(labeled_indices))

        print("Initializing unlabeled dataset...")
        unlabeled_features = x_train[unlabeled_indices]
        
        cached_state = load_strategy_state(config, unlabeled_indices, config.al.strategy)
        
        if cached_state is not None:
            print("Found cached strategy state, loading...")
            al_strategy._unlabeled_indices = cached_state['unlabeled_indices']
            al_strategy._remaining_indices = cached_state['remaining_indices']
            print(f"Loaded {len(al_strategy._remaining_indices)} remaining indices from cache")
        else:
            print("No cache found, initializing sampling from scratch...")
            init_start_time = time.time()
            
            al_strategy.initialize_sampling(unlabeled_features, unlabeled_indices)
            
            save_strategy_state(al_strategy, unlabeled_indices, config)
            
            print(f"Sampling initialization completed in {time.time() - init_start_time:.2f} seconds")

        al_strategy.initialize_sampling(unlabeled_features, unlabeled_indices)

        iteration = 0
        save_interval = 10
        evaluation_interval = 2
        start_time = time.time()
        
        while iteration < config.al.max_iterations:
            print(f"\n=== Starting iteration {iteration + 1} ===")
            remaining_samples = al_strategy.get_remaining_samples_count()
            print(f"Available unlabeled samples: {remaining_samples}")
            
            iter_start_time = time.time()
            
            try:
                current_x_train = [x_train[i] for i in labeled_indices]
                current_y_train = y_train[labeled_indices]

                print(f"Training model with {len(current_x_train)} samples...")
                train_dataset = TextDataset(
                    texts=current_x_train,
                    labels=current_y_train,
                    tokenizer=tokenizer,
                    num_labels=num_labels,
                )
                
                training_args = training_args_.get_training_arguments()
                
                torch.cuda.empty_cache()
                model = train_model(model, train_dataset, training_args)
                
                training_time = time.time() - iter_start_time
                results_history['time_list'].append(training_time)

                should_evaluate = (iteration % evaluation_interval == 0)
                should_save = (iteration % save_interval == 0) or (iteration == 0)
                
                if should_evaluate:
                    print("Evaluating model...")
                    test_dataset = dataset_cache.get_or_create(
                        texts=x_test,
                        labels=y_test,
                        tokenizer=tokenizer,
                        key='test',
                        num_labels=num_labels,
                    )

                    eval_start_time = time.time()
                    
                    eval_preds = []
                    eval_probs = []
                    eval_true = []
                    
                    eval_dataloader = DataLoader(test_dataset, batch_size=32)
                    model.eval()
                    
                    with torch.no_grad():
                        for batch in eval_dataloader:
                            batch = {k: v.to(device) for k, v in batch.items()}
                            outputs = model(**batch)
                            logits = outputs.logits
                            probs = torch.softmax(logits, dim=-1)
                            
                            eval_probs.extend(probs.cpu().numpy())
                            eval_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                            eval_true.extend(batch['labels'].cpu().numpy())
                    
                    prediction_time = time.time() - eval_start_time
                    results_history['prediction_time_list'].append(prediction_time)
                    
                    eval_probs = np.array(eval_probs)
                    eval_preds = np.array(eval_preds)
                    eval_true = np.array(eval_true)
                    
                    detailed_metrics = compute_detailed_metrics(eval_true, eval_preds, eval_probs)
                    
                    if should_save:
                        cm_path = os.path.join(plots_dir, f'confusion_matrix_iter_{iteration + 1}.png')
                        plot_confusion_matrix(eval_true, eval_preds, cm_path)
                        
                        pr_path = os.path.join(plots_dir, f'pr_curve_iter_{iteration + 1}.png')
                        plot_pr_curve(eval_true, eval_probs, pr_path)
                        
                        roc_path = os.path.join(plots_dir, f'roc_curve_iter_{iteration + 1}.png')
                        plot_roc_curve(eval_true, eval_probs, roc_path)
                    
                    results_history['accuracy_list'].append(detailed_metrics['accuracy'] * 100)
                    results_history['precision_list'].append(detailed_metrics['precision'] * 100)
                    results_history['f1_list'].append(detailed_metrics['f1_score'] * 100)
                    results_history['recall_list'].append(detailed_metrics['recall'] * 100)
                    results_history['roc_auc_list'].append(detailed_metrics['roc_auc'])
                    results_history['pr_auc_list'].append(detailed_metrics['pr_auc'])
                    results_history['inference_speed_list'].append(len(test_dataset) / prediction_time)
                    results_history['parameter_counts'].append(count_model_parameters(model))
                    results_history['confusion_matrices'].append(detailed_metrics['confusion_matrix'])
                    results_history['roc_curve_data'].append(detailed_metrics['roc_curve_data'])
                    results_history['pr_curve_data'].append(detailed_metrics['pr_curve_data'])

                    print(f"\nIteration {iteration + 1} results:")
                    print(f"Accuracy: {results_history['accuracy_list'][-1]:.2f}%")
                    print(f"Precision: {results_history['precision_list'][-1]:.2f}%")
                    print(f"F1 Score: {results_history['f1_list'][-1]:.2f}%")
                    print(f"Recall: {results_history['recall_list'][-1]:.2f}%")
                    print(f"Training time: {results_history['time_list'][-1]:.2f}s")
                    print(f"Prediction time: {results_history['prediction_time_list'][-1]:.2f}s")
                    print(f"Inference speed: {results_history['inference_speed_list'][-1]:.2f} samples/s")

                    wandb.log({
                        'accuracy': results_history['accuracy_list'][-1],
                        'precision': results_history['precision_list'][-1],
                        'f1_score': results_history['f1_list'][-1],
                        'recall': results_history['recall_list'][-1],
                        'training_time': results_history['time_list'][-1],
                        'prediction_time': results_history['prediction_time_list'][-1],
                        'inference_speed': results_history['inference_speed_list'][-1],
                        'total_parameters': count_model_parameters(model)['total_params'],
                        'trainable_parameters': count_model_parameters(model)['trainable_params'],
                        'iteration': iteration,
                        'labeled_samples': len(labeled_indices)
                    })
                else:
                    print(f"Iteration {iteration + 1} training completed in {training_time:.2f}s")
                    
                    if 'accuracy_list' in results_history and iteration > 0:
                        results_history['accuracy_list'].append(results_history['accuracy_list'][-1])
                        results_history['precision_list'].append(results_history['precision_list'][-1])
                        results_history['f1_list'].append(results_history['f1_list'][-1])
                        results_history['recall_list'].append(results_history['recall_list'][-1])
                        results_history['inference_speed_list'].append(results_history['inference_speed_list'][-1])
                        results_history['parameter_counts'].append(count_model_parameters(model))
                        results_history['confusion_matrices'].append(results_history['confusion_matrices'][-1])
                        results_history['prediction_time_list'].append(0.0)
                        
                        if 'roc_auc_list' in results_history:
                            results_history['roc_auc_list'].append(results_history['roc_auc_list'][-1])
                            results_history['pr_auc_list'].append(results_history['pr_auc_list'][-1])
                            results_history['roc_curve_data'].append(results_history['roc_curve_data'][-1])
                            results_history['pr_curve_data'].append(results_history['pr_curve_data'][-1])

                if results_history['accuracy_list'] and results_history['accuracy_list'][-1] >= config.al.target_accuracy:
                    print(f"\nReached target accuracy of {config.al.target_accuracy}%")
                    break

                if remaining_samples < config.al.query_size:
                    print(f"\nInsufficient unlabeled samples remaining. Saving current state...")
                    
                    insufficient_samples_path = os.path.join(models_dir, "insufficient_samples_final_model")
                    model.save_pretrained(insufficient_samples_path)
                    tokenizer.save_pretrained(insufficient_samples_path)
                    
                    if not should_evaluate:
                        print("Performing final evaluation...")
                        test_dataset = dataset_cache.get_or_create(
                            texts=x_test,
                            labels=y_test,
                            tokenizer=tokenizer,
                            key='test',
                            num_labels=num_labels
                        )
                        
                        eval_start_time = time.time()
                        
                        eval_preds = []
                        eval_probs = []
                        eval_true = []
                        
                        eval_dataloader = DataLoader(test_dataset, batch_size=32)
                        model.eval()
                        
                        with torch.no_grad():
                            for batch in eval_dataloader:
                                batch = {k: v.to(device) for k, v in batch.items()}
                                outputs = model(**batch)
                                logits = outputs.logits
                                probs = torch.softmax(logits, dim=-1)
                                
                                eval_probs.extend(probs.cpu().numpy())
                                eval_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                                eval_true.extend(batch['labels'].cpu().numpy())
                        
                        prediction_time = time.time() - eval_start_time
                        
                        eval_probs = np.array(eval_probs)
                        eval_preds = np.array(eval_preds)
                        eval_true = np.array(eval_true)
                        
                        detailed_metrics = compute_detailed_metrics(eval_true, eval_preds, eval_probs)
                        
                        cm_path = os.path.join(plots_dir, f'confusion_matrix_final.png')
                        plot_confusion_matrix(eval_true, eval_preds, cm_path)
                        
                        pr_path = os.path.join(plots_dir, f'pr_curve_final.png')
                        plot_pr_curve(eval_true, eval_probs, pr_path)
                        
                        roc_path = os.path.join(plots_dir, f'roc_curve_final.png')
                        plot_roc_curve(eval_true, eval_probs, roc_path)
                        
                        results_history['accuracy_list'].append(detailed_metrics['accuracy'] * 100)
                        results_history['precision_list'].append(detailed_metrics['precision'] * 100)
                        results_history['f1_list'].append(detailed_metrics['f1_score'] * 100)
                        results_history['recall_list'].append(detailed_metrics['recall'] * 100)
                        results_history['roc_auc_list'].append(detailed_metrics['roc_auc'])
                        results_history['pr_auc_list'].append(detailed_metrics['pr_auc'])
                        results_history['time_list'].append(training_time)
                        results_history['prediction_time_list'].append(prediction_time)
                        results_history['inference_speed_list'].append(len(test_dataset) / prediction_time)
                        results_history['parameter_counts'].append(count_model_parameters(model))
                        results_history['confusion_matrices'].append(detailed_metrics['confusion_matrix'])
                        results_history['roc_curve_data'].append(detailed_metrics['roc_curve_data'])
                        results_history['pr_curve_data'].append(detailed_metrics['pr_curve_data'])
                        
                        print(f"\nFinal evaluation results:")
                        print(f"Accuracy: {results_history['accuracy_list'][-1]:.2f}%")
                        print(f"Precision: {results_history['precision_list'][-1]:.2f}%")
                        print(f"F1 Score: {results_history['f1_list'][-1]:.2f}%")
                        print(f"Recall: {results_history['recall_list'][-1]:.2f}%")
                    
                    early_stop_time = time.time() - start_time
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    insufficient_samples_filename = f'insufficient_samples_results_{config.al.strategy}_{timestamp}.xlsx'

                    save_detailed_results(
                        config=config,
                        results_history=results_history,
                        total_time=early_stop_time,
                        al_strategy=al_strategy,
                        save_dir=results_dir,
                        filename=insufficient_samples_filename
                    )
                    
                    print(f"Early stopping due to insufficient samples. State saved.")
                    break

                print("Selecting next batch of samples...")

                if config.al.strategy == 'advanced_sampling':
                    value_ranking = al_strategy.compute_ranking(unlabeled_features)
                    selected_indices = value_ranking[:config.al.query_size]
                    
                    if selected_indices:
                        selected_labels = y_train[selected_indices]
                        al_strategy.update_model_state(selected_indices, selected_labels)
                else:
                    selected_indices = al_strategy.select_next_batch(config.al.query_size)

                if selected_indices:
                    labeled_indices.extend(selected_indices)
                    unlabeled_indices = list(set(unlabeled_indices) - set(selected_indices))
                    unlabeled_features = x_train[unlabeled_indices]
                    print(f"Selected {len(selected_indices)} new samples. "
                          f"Total labeled samples: {len(labeled_indices)}")
                else:
                    print("No samples selected in this iteration.")

                if iteration % config.training.save_steps == 0:
                    checkpoint_path = os.path.join(models_dir, f"checkpoint_iter_{iteration}")
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)

                iteration += 1

            except Exception as e:
                logging.error(f"Error in iteration {iteration}: {str(e)}")
                try:
                    model = model.cpu()
                    checkpoint_path = os.path.join(models_dir, f"error_checkpoint_iter_{iteration}.pt")
                    torch.save({
                        'iteration': iteration,
                        'model_state': model.state_dict(),
                        'labeled_indices': labeled_indices,
                        'results_history': results_history
                    }, checkpoint_path)
                except Exception as save_error:
                    logging.error(f"Failed to save error checkpoint: {str(save_error)}")
                raise

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        
        if results_history['accuracy_list']:
            print(f"Final Accuracy: {results_history['accuracy_list'][-1]:.2f}%")
            print(f"Final Precision: {results_history['precision_list'][-1]:.2f}%")
            print(f"Final F1 Score: {results_history['f1_list'][-1]:.2f}%")
            print(f"Final Recall: {results_history['recall_list'][-1]:.2f}%")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_results_filename = f'results_{config.al.strategy}_{timestamp}.xlsx'

        save_detailed_results(
            config=config,
            results_history=results_history,
            total_time=total_time,
            al_strategy=al_strategy,
            save_dir=results_dir,
            filename=final_results_filename
        )

        print("Saving final model...")
        final_model_path = os.path.join(models_dir, "final_model")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
    finally:
        try:
            wandb.finish()
            dataset_cache.clear()
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as cleanup_error:
            logging.error(f"Error during cleanup: {str(cleanup_error)}")

if __name__ == "__main__":
    main()