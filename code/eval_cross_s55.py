# eval_cross_s55.py — Cross-domain model evaluation
import os
import time
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AlbertForSequenceClassification, AlbertTokenizer
from peft import PeftModel
from sklearn.model_selection import train_test_split

from utils.dataset import TextDataset, CachedDataset, validate_dataset
from utils.metrics import (count_model_parameters, calculate_inference_speed,
                           plot_confusion_matrix, plot_pr_curve, plot_roc_curve,
                           compute_detailed_metrics)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(data_path):
    try:
        data = np.load(data_path, allow_pickle=True)
        logger.info(f"Keys in NPZ file: {list(data.keys())}")

        if 'features' in data:
            texts = data['features']
        elif 'text' in data:
            texts = data['text']
        elif 'x' in data:
            texts = data['x']
        else:
            first_key = list(data.keys())[0]
            logger.warning(f"Standard key for text data not found, using first key '{first_key}'")
            texts = data[first_key]

        if 'labels' in data:
            labels = data['labels']
        elif 'label' in data:
            labels = data['label']
        elif 'y' in data:
            labels = data['y']
        else:
            if len(data.keys()) > 1:
                second_key = list(data.keys())[1]
                labels = data[second_key]
            else:
                raise ValueError("Not enough arrays in NPZ file for text and labels")

        if len(texts) != len(labels):
            raise ValueError(f"Number of texts ({len(texts)}) does not match number of labels ({len(labels)})")

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        unique_labels = np.unique(labels)
        logger.info(f"Dataset loaded: total {len(texts)}, train {len(train_texts)}, test {len(test_texts)}")
        logger.info(f"Number of labels: {len(unique_labels)}")

        return train_texts, test_texts, train_labels, test_labels, len(unique_labels)

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def load_model(pretrained_model_path, ft_model_path, num_labels):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        tokenizer = AlbertTokenizer.from_pretrained(pretrained_model_path)
        base_model = AlbertForSequenceClassification.from_pretrained(
            pretrained_model_path, num_labels=num_labels
        )
        model = PeftModel.from_pretrained(base_model, ft_model_path)
        model.to(device)

        logger.info(f"Model loaded: {pretrained_model_path} + LoRA fine-tuned {ft_model_path}")
        return model, tokenizer, device

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def evaluate(model, test_dataset, device, results_path):
    model.eval()
    predictions = []
    true_labels = []
    probabilities = []

    dataloader = DataLoader(test_dataset, batch_size=32)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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

    results = compute_detailed_metrics(true_labels, predictions, probabilities)

    cm_path = os.path.join(results_path, 'confusion_matrix_final.png')
    plot_confusion_matrix(true_labels, predictions, cm_path)

    pr_path = os.path.join(results_path, 'pr_curve_final.png')
    plot_pr_curve(true_labels, probabilities, pr_path)

    roc_path = os.path.join(results_path, 'roc_curve_final.png')
    plot_roc_curve(true_labels, probabilities, roc_path)

    results['inference_speed'] = calculate_inference_speed(model, test_dataset, device)
    results['model_parameters'] = count_model_parameters(model)

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-domain model evaluation")
    parser.add_argument('--pretrained_model_path', default='./plmd-model/ALBERT-UNSW',
                        help='Pre-trained model path')
    parser.add_argument('--ft_model_path', default='./ftune-model/ALBERT-PreUNSW-FtUSTC-bass',
                        help='Fine-tuned model path (LoRA/PEFT weights)')
    parser.add_argument('--data_path', default='./data/USTC_concate_data.npz',
                        help='Evaluation dataset path')
    parser.add_argument('--results_path', default='./results/cross_domain_eval',
                        help='Results output directory')
    args = parser.parse_args()

    os.makedirs(args.results_path, exist_ok=True)

    _, test_texts, _, test_labels, num_labels = load_dataset(args.data_path)

    model, tokenizer, device = load_model(args.pretrained_model_path, args.ft_model_path, num_labels)

    dataset_cache = CachedDataset()
    test_dataset = dataset_cache.get_or_create(
        test_texts, test_labels, tokenizer, key='test', num_labels=num_labels
    )

    results = evaluate(model, test_dataset, device, args.results_path)

    print("\nModel Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")

    print("\nROC AUC:")
    avg_auc = 0
    for i, auc_value in results['roc_auc'].items():
        print(f"Class {i}: {auc_value:.4f}")
        avg_auc += auc_value
    print(f"Average AUC: {avg_auc/len(results['roc_auc']):.4f}")

    print(f"\nInference speed: {results['inference_speed']:.2f} samples/sec")
    print(f"Model parameters: {results['model_parameters']}")

    with open(os.path.join(args.results_path, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1_score']:.4f}\n")
        f.write(f"Average AUC: {avg_auc/len(results['roc_auc']):.4f}\n")
        f.write(f"Inference speed: {results['inference_speed']:.2f} samples/sec\n")

    logger.info("Evaluation complete, results saved")


if __name__ == "__main__":
    from tqdm import tqdm
    main()
