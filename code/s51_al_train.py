# s51_al_train.py — Active Learning fine-tuning
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import time, logging, wandb, gc, os
from transformers import (Trainer, TrainingArguments, get_linear_schedule_with_warmup)
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import psutil

from utils.data_loader import (data_import, get_device, parallel_load_data,
                               estimate_memory_usage, get_optimal_chunk_size, process_chunk)
from utils.dataset import TextDataset, CachedDataset, validate_dataset
from utils.metrics import (count_model_parameters, calculate_inference_speed, save_pr_data,
                           plot_confusion_matrix, plot_pr_curve, plot_roc_curve,
                           compute_detailed_metrics, evaluate_model, save_detailed_results)
from param_config import config, training_args_
from AL_strategy import get_strategy, BoundaryAwareSampling, CustomDataset

os.environ.setdefault("WANDB_MODE", "disabled")

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.log.level),
    format=config.log.format,
    filename=config.log.log_file
)


def data_collator(data):
    """Data collator function for batching."""
    return {
        'input_ids': torch.stack([x['input_ids'] for x in data]),
        'attention_mask': torch.stack([x['attention_mask'] for x in data]),
        'labels': torch.stack([x['labels'] for x in data])
    }


def setup_model(model_path, num_labels):
    """Initialize model and tokenizer."""
    print(f"Setting up model with {num_labels} labels...")

    tokenizer = AlbertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AlbertForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        local_files_only=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        modules_to_save=config.lora.modules_to_save
    )

    model = get_peft_model(model, lora_config)
    print(f"Model initialized with {num_labels} output classes")

    return model, tokenizer


def train_model(model, train_dataset, training_args):
    """Optimized model training function."""
    try:
        # Validate dataset
        validate_dataset(train_dataset)

        # Split into training and validation sets
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size]
        )

        # Optimize GPU memory usage
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            suggested_batch_size = min(16, max(1, int(gpu_mem / (1024**3) * 2)))

            # Clear GPU memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            logging.info(f"Initial GPU memory usage: {initial_memory/1024**2:.2f} MB")
        else:
            suggested_batch_size = 8

        training_args.per_device_train_batch_size = min(
            training_args.per_device_train_batch_size,
            suggested_batch_size
        )

        # Use gradient accumulation to maintain effective batch size
        effective_batch_size = 32
        training_args.gradient_accumulation_steps = max(1, effective_batch_size // training_args.per_device_train_batch_size)

        # Use progressive learning rate
        training_args.learning_rate = 1e-5
        training_args.warmup_ratio = 0.1

        # Disable mixed precision training, use standard precision
        training_args.fp16 = False
        training_args.bf16 = False

        # Reduce evaluation frequency to speed up training
        training_args.logging_steps = 50
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = 100
        training_args.save_strategy = "steps"
        training_args.save_steps = 100
        training_args.load_best_model_at_end = True
        training_args.metric_for_best_model = "eval_loss"

        # Use more efficient data loading
        num_workers = min(4, os.cpu_count() - 1) if os.cpu_count() > 1 else 0
        training_args.dataloader_num_workers = num_workers

        # Ensure model is on the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Create optimized optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=0.01,
            eps=1e-8,
            betas=(0.9, 0.999)  # Default values, good general-purpose settings
        )

        # Create learning rate scheduler
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

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            optimizers=(optimizer, lr_scheduler),
            data_collator=data_collator,
        )

        # Validate model output before training
        with torch.no_grad():
            sample_batch = next(iter(DataLoader(train_subset, batch_size=1)))
            sample_output = model(**{k: v.to(device) for k, v in sample_batch.items()})
            if not torch.isfinite(sample_output.loss):
                raise ValueError("Model produces invalid loss before training")

        # Train model
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


def main():
    """Main function: execute active learning training and evaluation pipeline."""
    try:
        # Validate required directories and files
        required_paths = [
            config.data.data_path,
            config.model.model_path,
            config.model.output_dir,
        ]

        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required path not found: {path}")

        # Initialize wandb
        wandb.init(
            project=config.training.wandb_project,
            config=config,
            name=f"{config.al.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Create output directory
        # Add LoRA usage identifier
        lora_suffix = "" if config.lora.use_lora else "-full-ft"
        output_dir_base = os.path.join(config.model.output_dir)
        if not config.lora.use_lora and "-full-ft" not in output_dir_base:
            output_dir_base += "-full-ft"

        os.makedirs(config.model.output_dir, exist_ok=True)
        results_dir = os.path.join(config.model.output_dir, 'results')
        plots_dir = os.path.join(config.model.output_dir, 'plots')
        models_dir = os.path.join(config.model.output_dir, 'models')
        for d in [results_dir, plots_dir, models_dir]:
            os.makedirs(d, exist_ok=True)

        # Create dataset cache manager
        dataset_cache = CachedDataset()

        # 1. Data loading and preprocessing
        print("Loading data...")
        try:
            # Get data file size
            file_size = os.path.getsize(config.data.data_path)
            print(f"Data file size: {file_size / (1024**2):.2f} MB")

            # Estimate memory requirements
            with np.load(config.data.data_path, mmap_mode='r') as data:
                n_samples = len(data['features'])
                feature_shape = data['features'][0].shape if len(data['features']) > 0 else (0,)
                estimated_memory = estimate_memory_usage(n_samples, np.prod(feature_shape))
                print(f"Estimated memory requirement: {estimated_memory:.2f} GB")

            # Decide processing method based on system resources
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            print(f"Available system memory: {available_memory:.2f} GB")

            if estimated_memory < available_memory * 0.7:  # If memory is sufficient
                # Load directly
                X, y = parallel_load_data(
                    config.data.data_path,
                    n_jobs=max(1, psutil.cpu_count(logical=False) - 1)
                )
            else:
                # Load in chunks
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

            # Validate data
            print("\nValidating loaded data...")
            print(f"Total samples loaded: {len(y)}")
            print(f"Feature array shape: {X.shape}")
            print(f"Label array shape: {y.shape}")
            print(f"Number of unique labels: {len(np.unique(y))}")

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

            # Clean up memory
        gc.collect()

        # 2. Dataset split
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

        # 3. Validate configuration
        if config.al.initial_labeled_samples + config.al.query_size > len(x_train):
            raise ValueError("Initial labeled samples + query size exceeds total samples")

        # 4. Model and strategy initialization
        print("Initializing model and strategy...")
        model, tokenizer = setup_model(config.model.model_path, num_labels)
        device = torch.device(config.device)
        model = model.to(device)

        # 5. Initialize strategy parameters
        strategy_params = {
            'batch_size': getattr(config.al, 'batch_size', 128),
            'device': device
        }

        if config.al.strategy == 'boundary':
            strategy_params.update({
                'buffer_size': config.al.buffer_size,
                'memory_size': config.al.memory_size,
                'tau': config.al.tau,
                'eps': config.al.eps,
                'alpha': config.al.alpha,
                'beta': config.al.beta,
                'gamma': config.al.gamma,
                'temperature': config.al.temperature
            })

        elif config.al.strategy in ['density', 'sunb']:
            strategy_params.update({
                'eps': getattr(config.al, 'eps', 0.5),
                'min_samples': getattr(config.al, 'min_samples', 5)
            })
        elif config.al.strategy == 'graph_density':
            strategy_params['k_neighbors'] = getattr(config.al, 'k_neighbors', 5)
        elif config.al.strategy == 'qbc':
            strategy_params.update({
                'n_committees': getattr(config.al, 'n_committees', 3),
                'dropout_rate': getattr(config.al, 'dropout_rate', 0.2),
                'model_path': config.model.model_path  # Add model path
            })

        al_strategy = get_strategy(config.al.strategy, model, tokenizer, **strategy_params)

        # 6. Initialize results tracking
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

        # 7. Initial sample selection
        print("Selecting initial samples...")
        total_samples = len(x_train)
        np.random.seed(config.al.random_seed)
        labeled_indices = list(np.random.choice(
            total_samples,
            size=config.al.initial_labeled_samples,
            replace=False
        ))
        unlabeled_indices = list(set(range(total_samples)) - set(labeled_indices))

        # 8. Initialize sampling strategy
        print("Initializing unlabeled dataset...")
        unlabeled_features = x_train[unlabeled_indices]  # Use feature array directly
        # Initialize sampling strategy
        al_strategy.initialize_sampling(unlabeled_features, unlabeled_indices)

        # 9. Active learning iterations
        iteration = 0
        save_interval = 10  # Save every 10 iterations
        evaluation_interval = 2  # Evaluate every 2 iterations
        start_time = time.time()

        while iteration < config.al.max_iterations:
            print(f"\n=== Starting iteration {iteration + 1} ===")
            remaining_samples = al_strategy.get_remaining_samples_count()
            print(f"Available unlabeled samples: {remaining_samples}")

            iter_start_time = time.time()

            try:
                # Prepare current training data
                current_x_train = [x_train[i] for i in labeled_indices]
                current_y_train = y_train[labeled_indices]

                # Train model
                print(f"Training model with {len(current_x_train)} samples...")
                train_dataset = TextDataset(
                    texts=current_x_train,
                    labels=current_y_train,
                    tokenizer=tokenizer,
                    num_labels=num_labels  # Pass precomputed num_labels
                )

                # Get training arguments
                training_args = training_args_.get_training_arguments()

                torch.cuda.empty_cache()
                model = train_model(model, train_dataset, training_args)

                # Record training time
                training_time = time.time() - iter_start_time
                results_history['time_list'].append(training_time)

                # Evaluate model - only at specific intervals
                should_evaluate = (iteration % evaluation_interval == 0)
                should_save = (iteration % save_interval == 0) or (iteration == 0)

                if should_evaluate:
                    print("Evaluating model...")
                    test_dataset = dataset_cache.get_or_create(
                        texts=x_test,
                        labels=y_test,
                        tokenizer=tokenizer,
                        key='test',
                        num_labels=num_labels  # Pass precomputed num_labels
                    )

                    # Record evaluation start time
                    eval_start_time = time.time()

                    # Get predictions and probabilities from model
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

                    # Calculate prediction time
                    prediction_time = time.time() - eval_start_time
                    results_history['prediction_time_list'].append(prediction_time)

                    # Calculate detailed metrics
                    eval_probs = np.array(eval_probs)
                    eval_preds = np.array(eval_preds)
                    eval_true = np.array(eval_true)

                    # Calculate detailed evaluation metrics
                    detailed_metrics = compute_detailed_metrics(eval_true, eval_preds, eval_probs)

                    # Save visualizations
                    if should_save:
                        # Confusion matrix
                        cm_path = os.path.join(plots_dir, f'confusion_matrix_iter_{iteration + 1}.png')
                        plot_confusion_matrix(eval_true, eval_preds, cm_path)

                        # PR curve
                        pr_path = os.path.join(plots_dir, f'pr_curve_iter_{iteration + 1}.png')
                        plot_pr_curve(eval_true, eval_probs, pr_path)

                        # ROC curve
                        roc_path = os.path.join(plots_dir, f'roc_curve_iter_{iteration + 1}.png')
                        plot_roc_curve(eval_true, eval_probs, roc_path)

                    # Update results tracking
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

                    # Print current results
                    print(f"\nIteration {iteration + 1} results:")
                    print(f"Accuracy: {results_history['accuracy_list'][-1]:.2f}%")
                    print(f"Precision: {results_history['precision_list'][-1]:.2f}%")
                    print(f"F1 Score: {results_history['f1_list'][-1]:.2f}%")
                    print(f"Recall: {results_history['recall_list'][-1]:.2f}%")
                    print(f"Training time: {results_history['time_list'][-1]:.2f}s")
                    print(f"Prediction time: {results_history['prediction_time_list'][-1]:.2f}s")
                    print(f"Inference speed: {results_history['inference_speed_list'][-1]:.2f} samples/s")

                    # Log to wandb
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
                    # If not evaluating, only record training time
                    print(f"Iteration {iteration + 1} training completed in {training_time:.2f}s")

                    # Still need to ensure list lengths are consistent, add placeholders
                    if 'accuracy_list' in results_history and iteration > 0:
                        results_history['accuracy_list'].append(results_history['accuracy_list'][-1])
                        results_history['precision_list'].append(results_history['precision_list'][-1])
                        results_history['f1_list'].append(results_history['f1_list'][-1])
                        results_history['recall_list'].append(results_history['recall_list'][-1])
                        results_history['inference_speed_list'].append(results_history['inference_speed_list'][-1])
                        results_history['parameter_counts'].append(count_model_parameters(model))
                        results_history['confusion_matrices'].append(results_history['confusion_matrices'][-1])
                        results_history['prediction_time_list'].append(0.0)  # Placeholder

                        # If present, also add placeholders
                        if 'roc_auc_list' in results_history:
                            results_history['roc_auc_list'].append(results_history['roc_auc_list'][-1])
                            results_history['pr_auc_list'].append(results_history['pr_auc_list'][-1])
                            results_history['roc_curve_data'].append(results_history['roc_curve_data'][-1])
                            results_history['pr_curve_data'].append(results_history['pr_curve_data'][-1])

                # Check stopping conditions
                if results_history['accuracy_list'] and results_history['accuracy_list'][-1] >= config.al.target_accuracy:
                    print(f"\nReached target accuracy of {config.al.target_accuracy}%")
                    break

                if remaining_samples < config.al.query_size:
                    print(f"\nInsufficient unlabeled samples remaining. Saving current state...")

                    # Save current model
                    insufficient_samples_path = os.path.join(models_dir, "insufficient_samples_final_model")
                    model.save_pretrained(insufficient_samples_path)
                    tokenizer.save_pretrained(insufficient_samples_path)

                    # Calculate and save final evaluation metrics (if not already evaluated this iteration)
                    if not should_evaluate:
                        print("Performing final evaluation...")
                        test_dataset = dataset_cache.get_or_create(
                            texts=x_test,
                            labels=y_test,
                            tokenizer=tokenizer,
                            key='test',
                            num_labels=num_labels
                        )

                        # Record evaluation start time
                        eval_start_time = time.time()

                        # Get predictions and probabilities from model
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

                        # Calculate prediction time
                        prediction_time = time.time() - eval_start_time

                        # Calculate detailed metrics
                        eval_probs = np.array(eval_probs)
                        eval_preds = np.array(eval_preds)
                        eval_true = np.array(eval_true)

                        detailed_metrics = compute_detailed_metrics(eval_true, eval_preds, eval_probs)

                        # Save visualizations
                        cm_path = os.path.join(plots_dir, f'confusion_matrix_final.png')
                        plot_confusion_matrix(eval_true, eval_preds, cm_path)

                        pr_path = os.path.join(plots_dir, f'pr_curve_final.png')
                        plot_pr_curve(eval_true, eval_probs, pr_path)

                        roc_path = os.path.join(plots_dir, f'roc_curve_final.png')
                        plot_roc_curve(eval_true, eval_probs, roc_path)

                        # Update results tracking
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

                    # Save detailed results for current state
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

                # Select next batch of samples
                print("Selecting next batch of samples...")
                if isinstance(al_strategy, BoundaryAwareSampling):
                    # Use boundary-aware sampling strategy
                    value_ranking = al_strategy.compute_value_ranking(unlabeled_features)
                    selected_indices = value_ranking[:config.al.query_size]

                    # Update memory
                    if selected_indices:
                        selected_labels = y_train[selected_indices]
                        al_strategy.update_memory(selected_indices, selected_labels)
                else:
                    selected_indices = al_strategy.select_next_batch(config.al.query_size)

                # Update dataset
                if selected_indices:
                    labeled_indices.extend(selected_indices)
                    unlabeled_indices = list(set(unlabeled_indices) - set(selected_indices))
                    unlabeled_features = x_train[unlabeled_indices]
                    print(f"Selected {len(selected_indices)} new samples. "
                          f"Total labeled samples: {len(labeled_indices)}")
                else:
                    print("No samples selected in this iteration.")

                # Save checkpoint
                if iteration % config.training.save_steps == 0:
                    checkpoint_path = os.path.join(models_dir, f"checkpoint_iter_{iteration}")
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)

                iteration += 1

            except Exception as e:
                logging.error(f"Error in iteration {iteration}: {str(e)}")
                # Save error state
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

        # 10. Summarize and save results
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")

        if results_history['accuracy_list']:
            print(f"Final Accuracy: {results_history['accuracy_list'][-1]:.2f}%")
            print(f"Final Precision: {results_history['precision_list'][-1]:.2f}%")
            print(f"Final F1 Score: {results_history['f1_list'][-1]:.2f}%")
            print(f"Final Recall: {results_history['recall_list'][-1]:.2f}%")

        # Save final results
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

        # Save final model
        print("Saving final model...")
        final_model_path = os.path.join(models_dir, "final_model")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise
    finally:
        # Clean up resources
        try:
            wandb.finish()
            dataset_cache.clear()
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as cleanup_error:
            logging.error(f"Error during cleanup: {str(cleanup_error)}")

if __name__ == "__main__":
    main()
