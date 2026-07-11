import os
import time
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             precision_recall_curve, confusion_matrix, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns


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
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    return cm


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

        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

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

    return {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}


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

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'roc_curve_data': {'fpr': fpr, 'tpr': tpr},
        'pr_curve_data': {'precision': precision_curve, 'recall': recall_curve}
    }


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
        cm_path = os.path.join(save_dir, f'confusion_matrix_iter_{iteration + 1}.png')
        plot_confusion_matrix(true_labels, predictions, cm_path)

        pr_path = os.path.join(save_dir, f'pr_curve_iter_{iteration + 1}.png')
        pr_data = plot_pr_curve(true_labels, probabilities, pr_path)

        roc_path = os.path.join(save_dir, f'roc_curve_iter_{iteration + 1}.png')
        roc_data = plot_roc_curve(true_labels, probabilities, roc_path)

        pr_data_path = os.path.join(save_dir, f'pr_curve_iter_{iteration + 1}_data.csv')
        save_pr_data(pr_data, pr_data_path, iteration)

    inference_speed = calculate_inference_speed(model, test_dataset, device)

    results = detailed_metrics.copy()
    results['inference_speed'] = inference_speed
    results['model_parameters'] = count_model_parameters(model)

    return results


def save_detailed_results(config, results_history, total_time, al_strategy, save_dir, filename=None):
    if filename is None:
        from datetime import datetime
        lora_suffix = "" if config.lora.use_lora else "-full-ft"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'results_{config.al.strategy}{lora_suffix}_{timestamp}.xlsx'

    results_path = os.path.join(save_dir, filename)

    try:
        required_fields = [
            'accuracy_list', 'precision_list', 'f1_list', 'recall_list',
            'roc_auc_list', 'pr_auc_list', 'time_list', 'prediction_time_list',
            'parameter_counts', 'confusion_matrices'
        ]

        lengths = [len(results_history.get(field, [])) for field in required_fields if field in results_history]
        min_length = min(lengths) if lengths else 0
        print(f"Using minimum length across all metrics: {min_length}")

        main_data = {
            'Iteration': [], 'Accuracy': [], 'Precision': [], 'Recall': [],
            'F1': [], 'ROC_AUC_Macro': [], 'PR_AUC_Macro': [],
            'Training_Time(s)': [], 'Prediction_Time(s)': [],
            'Inference_Speed': [], 'Total_Parameters': [], 'Trainable_Parameters': []
        }

        for i in range(min_length):
            main_data['Iteration'].append(i + 1)
            main_data['Accuracy'].append(f"{results_history['accuracy_list'][i]:.2f}")
            main_data['Precision'].append(f"{results_history['precision_list'][i]:.2f}")
            main_data['Recall'].append(f"{results_history['recall_list'][i]:.2f}")
            main_data['F1'].append(f"{results_history['f1_list'][i]:.2f}")

            if 'roc_auc_list' in results_history and i < len(results_history['roc_auc_list']):
                macro_roc_auc = np.mean([v for v in results_history['roc_auc_list'][i].values()])
                main_data['ROC_AUC_Macro'].append(f"{macro_roc_auc:.4f}")
            else:
                main_data['ROC_AUC_Macro'].append("N/A")

            if 'pr_auc_list' in results_history and i < len(results_history['pr_auc_list']):
                macro_pr_auc = np.mean([v for v in results_history['pr_auc_list'][i].values()])
                main_data['PR_AUC_Macro'].append(f"{macro_pr_auc:.4f}")
            else:
                main_data['PR_AUC_Macro'].append("N/A")

            main_data['Training_Time(s)'].append(f"{results_history['time_list'][i]:.2f}")

            if 'prediction_time_list' in results_history and i < len(results_history['prediction_time_list']):
                main_data['Prediction_Time(s)'].append(f"{results_history['prediction_time_list'][i]:.2f}")
            else:
                main_data['Prediction_Time(s)'].append("N/A")

            if 'inference_speed_list' in results_history and i < len(results_history['inference_speed_list']):
                main_data['Inference_Speed'].append(f"{results_history['inference_speed_list'][i]:.2f}")
            else:
                main_data['Inference_Speed'].append("N/A")

            main_data['Total_Parameters'].append(results_history['parameter_counts'][i]['total_params'])
            main_data['Trainable_Parameters'].append(results_history['parameter_counts'][i]['trainable_params'])

        df_main = pd.DataFrame(main_data)

        config_data = {
            'Parameter': ['Strategy', 'Initial Samples', 'Query Size', 'Total Time (s)'],
            'Value': [config.al.strategy, config.al.initial_labeled_samples,
                     config.al.query_size, total_time]
        }
        df_config = pd.DataFrame(config_data)

        if results_history.get('confusion_matrices', []):
            final_confusion_matrix = results_history['confusion_matrices'][-1]
            df_confusion = pd.DataFrame(
                final_confusion_matrix,
                index=[f'True Class {i}' for i in range(len(final_confusion_matrix))],
                columns=[f'Predicted Class {i}' for i in range(len(final_confusion_matrix))]
            )
        else:
            df_confusion = pd.DataFrame()

        roc_data = []
        pr_data = []

        if 'roc_curve_data' in results_history and results_history['roc_curve_data']:
            final_roc_data = results_history['roc_curve_data'][-1]
            for class_idx in final_roc_data['fpr'].keys():
                for i in range(len(final_roc_data['fpr'][class_idx])):
                    roc_data.append({
                        'Class': class_idx,
                        'FPR': final_roc_data['fpr'][class_idx][i],
                        'TPR': final_roc_data['tpr'][class_idx][i]
                    })

        if 'pr_curve_data' in results_history and results_history['pr_curve_data']:
            final_pr_data = results_history['pr_curve_data'][-1]
            for class_idx in final_pr_data['precision'].keys():
                for i in range(min(len(final_pr_data['precision'][class_idx]), len(final_pr_data['recall'][class_idx]))):
                    pr_data.append({
                        'Class': class_idx,
                        'Precision': final_pr_data['precision'][class_idx][i],
                        'Recall': final_pr_data['recall'][class_idx][i]
                    })

        df_roc = pd.DataFrame(roc_data)
        df_pr = pd.DataFrame(pr_data)

        with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
            df_config.to_excel(writer, sheet_name='results', startrow=0, index=False)
            df_main.to_excel(writer, sheet_name='results', startrow=len(df_config)+2, index=False)
            df_confusion.to_excel(writer, sheet_name='confusion_matrix')
            if not df_roc.empty:
                df_roc.to_excel(writer, sheet_name='roc_curve_data', index=False)
            if not df_pr.empty:
                df_pr.to_excel(writer, sheet_name='pr_curve_data', index=False)

        logging.info(f"Detailed results saved to {results_path}")
        print(f"Detailed results successfully saved to {results_path}")

    except Exception as e:
        logging.error(f"Error saving detailed results: {str(e)}")
        print(f"Error saving detailed results: {str(e)}")
        import traceback
        traceback.print_exc()
