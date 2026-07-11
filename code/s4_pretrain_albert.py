import argparse
import numpy as np
import torch
from transformers import AlbertForMaskedLM, AlbertTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from utils.dataset import PretrainingDataset


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def data_import(path):
    data = np.load(path)
    print("Successfully loaded data")
    X, y = data["features"], data["labels"]
    return X, y


def split_data(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=43)
    return x_train, x_test, y_train, y_test


def pretrain_albert(x_train, y_train, model, tokenizer, plm_path, epochs=3, batch_size=16):
    train_dataset = PretrainingDataset(x_train, y_train, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy=None,
        save_strategy='no',
        load_best_model_at_end=True,
        logging_dir="./logs",
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_steps=1000,
        logging_steps=500,
        save_total_limit=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    model.save_pretrained(plm_path)
    tokenizer.save_pretrained(plm_path)


def main(data_path, plm_path, base_model_path, epochs=3, batch_size=16):
    X, y = data_import(data_path)
    x_train, x_test, y_train, y_test = split_data(X, y)

    model = AlbertForMaskedLM.from_pretrained(
        base_model_path,
        num_labels=len(np.unique(y))
    )
    tokenizer = AlbertTokenizer.from_pretrained(base_model_path)

    print("Pretraining ALBERT model...")
    pretrain_albert(x_train, y_train, model, tokenizer, plm_path, epochs=epochs, batch_size=batch_size)
    print("Pretraining complete. Model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALBERT MLM pre-training")
    parser.add_argument('--data_path', default='./data/USTC_concate_data.npz', help='Training data NPZ path')
    parser.add_argument('--output_dir', default='./plmd-model/ALBERT-USTC', help='Pre-trained model output directory')
    parser.add_argument('--base_model', default='./base-model/albert-base-v2', help='Base model path')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    main(args.data_path, args.output_dir, args.base_model, args.epochs, args.batch_size)
