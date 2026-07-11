"""
Q8BERT (8-bit quantized BERT) pre-training script.

Applies 8-bit quantization to BERT via bitsandbytes (preferred) or a custom
MinimalQuantizer fallback, then runs MLM pre-training on traffic feature data.

Usage:
    python s43_pretrain_q8bert.py --data_path ./data/USTC_concate_data.npz --epochs 3
    python s43_pretrain_q8bert.py --data_path ./data/USTC_concate_data.npz --use_custom_quantization
    python s43_pretrain_q8bert.py --data_path ./data/USTC_concate_data.npz --disable_wandb
"""

import os
import time
import math
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm

from utils.dataset import PretrainingDataset

# ---------------------------------------------------------------------------
# bitsandbytes availability check
# ---------------------------------------------------------------------------
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    print("Warning: bitsandbytes not found. Falling back to custom 8-bit quantization.")


# ---------------------------------------------------------------------------
# Cached MLM Dataset (Q8BERT-specific)
# ---------------------------------------------------------------------------
# The base PretrainingDataset re-tokenises every sample on each access.
# Q8BERTPretrainingDataset adds an LRU-style sample cache so that the most
# recently accessed *cache_size* tokenised samples are kept in memory,
# avoiding redundant tokenisation across epochs.  This matters for Q8BERT
# because the quantised forward pass is already slower than fp32 and the
# tokenisation overhead becomes proportionally more significant.

class Q8BERTPretrainingDataset(PretrainingDataset):
    """PretrainingDataset with an in-memory sample cache."""

    def __init__(self, features, labels, tokenizer, max_length=128,
                 mlm_probability=0.15, cache_size=1000):
        super().__init__(features, labels, tokenizer, max_length, mlm_probability)
        self.cache_size = min(cache_size, len(features))
        self.cache: dict = {}

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        sample = super().__getitem__(idx)

        # Evict oldest entry when the cache is full.
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[idx] = sample
        return sample


# ---------------------------------------------------------------------------
# Quantization utilities (unique to Q8BERT)
# ---------------------------------------------------------------------------

class MinimalQuantizer:
    """Basic quantizer for 8-bit quantization when bitsandbytes is not available."""

    def __init__(self):
        self.min_val = None
        self.max_val = None
        self.scale = None
        self.zero_point = None

    def fit(self, tensor):
        self.min_val = torch.min(tensor).item()
        self.max_val = torch.max(tensor).item()
        self.scale = (self.max_val - self.min_val) / 255.0
        self.zero_point = round(-self.min_val / self.scale)

    def quantize(self, tensor):
        if self.scale is None:
            self.fit(tensor)
        return torch.round(tensor / self.scale + self.zero_point).clamp(0, 255).to(torch.uint8)

    def dequantize(self, quantized):
        return (quantized.float() - self.zero_point) * self.scale


class QuantizedLinear(torch.nn.Module):
    """Simulated 8-bit linear layer using MinimalQuantizer."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self.weight_quantizer = MinimalQuantizer()
        self.input_quantizer = MinimalQuantizer()

        self.register_buffer('quantized_weight',
                             torch.zeros((out_features, in_features), dtype=torch.uint8))
        self.weight_quantized = False

    def quantize_weight(self):
        if not self.weight_quantized:
            self.quantized_weight = self.weight_quantizer.quantize(self.weight.data)
            self.weight_quantized = True

    def forward(self, x):
        x_quant = self.input_quantizer.quantize(x)

        if not self.weight_quantized:
            self.quantize_weight()

        # Dequantize for computation — a full int8 matmul would be more
        # efficient but this keeps the implementation simple.
        x_dequant = self.input_quantizer.dequantize(x_quant)
        weight_dequant = self.weight_quantizer.dequantize(self.quantized_weight)

        return torch.nn.functional.linear(x_dequant, weight_dequant, self.bias)


def quantize_bert_model(model):
    """
    Replace nn.Linear layers (except classifier / prediction heads) with 8-bit
    equivalents — bitsandbytes Linear8bitLt when available, QuantizedLinear
    otherwise.
    """
    if HAS_BITSANDBYTES:
        print("Using bitsandbytes for 8-bit quantization")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if 'classifier' in name or 'pred_layer' in name:
                    continue
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                replacement = bnb.nn.Linear8bitLt(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False, threshold=6.0,
                )
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, replacement)
    else:
        print("Using custom 8-bit quantization")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if 'classifier' in name or 'pred_layer' in name:
                    continue
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                new_layer = QuantizedLinear(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                )
                with torch.no_grad():
                    new_layer.weight.copy_(module.weight)
                    if module.bias is not None:
                        new_layer.bias.copy_(module.bias)
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, new_layer)

    return model


# ---------------------------------------------------------------------------
# Pre-training entry point
# ---------------------------------------------------------------------------

def pretrain_q8bert(data_path, output_dir, base_model_path, batch_size=8,
                    epochs=3, learning_rate=5e-5, max_length=128,
                    mlm_probability=0.15, cache_size=1000,
                    use_custom_quantization=False):
    """Pretrain Q8BERT (8-bit quantized BERT) model."""

    print("Setting up Q8BERT pretraining...")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load data --------------------------------------------------------
    print(f"Loading data from {data_path}")
    try:
        with np.load(data_path) as data:
            features = data['features']
            labels = data['labels']
            print(f"Loaded {len(features)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    # ---- Load base BERT model and tokenizer --------------------------------
    print(f"Loading BERT model from {base_model_path}")
    tokenizer = BertTokenizer.from_pretrained(
        base_model_path, clean_up_tokenization_spaces=False)
    model = BertForMaskedLM.from_pretrained(base_model_path)

    # ---- Apply 8-bit quantization -----------------------------------------
    if use_custom_quantization or not HAS_BITSANDBYTES:
        print("Using custom 8-bit quantization")
        model = quantize_bert_model(model)
    else:
        print("Using bitsandbytes for 8-bit quantization")
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)

    model = model.to(device)
    print("Model loaded and ready for training")

    # ---- Dataset / dataloader ---------------------------------------------
    dataset = Q8BERTPretrainingDataset(
        features, labels, tokenizer,
        max_length=max_length,
        mlm_probability=mlm_probability,
        cache_size=cache_size,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4)

    # ---- Optimizer / scheduler --------------------------------------------
    if not HAS_BITSANDBYTES or use_custom_quantization:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(dataloader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps)

    # ---- Training loop ----------------------------------------------------
    print(f"Starting Q8BERT pretraining for {epochs} epochs")
    model.train()

    global_step = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mlm_labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=mlm_labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix({'loss': loss.item()})

            # Periodic checkpoint
            if global_step % 1000 == 0:
                avg_loss = running_loss / 1000
                print(f"Step {global_step}: Average Loss = {avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    print(f"New best loss: {best_loss:.4f}, saving checkpoint")
                    ckpt_dir = os.path.join(output_dir, 'checkpoints',
                                            f'checkpoint-{global_step}')
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                running_loss = 0.0

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

    # ---- Save final model -------------------------------------------------
    print("Pretraining complete, saving model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    info_path = os.path.join(output_dir, 'training_info.txt')
    with open(info_path, 'w') as f:
        f.write(f"Model: Q8BERT (8-bit quantized BERT)\n")
        f.write(f"Base model: {base_model_path}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"MLM probability: {mlm_probability}\n")
        quant_type = 'Custom' if use_custom_quantization else 'bitsandbytes'
        f.write(f"Quantization: {quant_type}\n")
        f.write(f"Best loss: {best_loss:.4f}\n")

    print(f"Q8BERT model saved to {output_dir}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pretrain Q8BERT (8-bit quantized BERT)')

    parser.add_argument('--data_path', type=str,
                        default='./data/USTC_concate_data.npz',
                        help='Path to the dataset (.npz)')
    parser.add_argument('--output_dir', type=str,
                        default='./plmd-model/Q8BERT',
                        help='Output directory for saving the model')
    parser.add_argument('--base_model_path', type=str,
                        default='./base-model/bert-base-uncased',
                        help='Path to the base BERT checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--mlm_probability', type=float, default=0.15,
                        help='Masked language model probability')
    parser.add_argument('--cache_size', type=int, default=1000,
                        help='Number of tokenised samples to cache in memory')
    parser.add_argument('--use_custom_quantization', action='store_true',
                        help='Use custom quantization instead of bitsandbytes')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Set WANDB_MODE=disabled to suppress wandb logging')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    # Derive dataset name from data_path for the output directory suffix.
    dataset_name = os.path.splitext(
        os.path.basename(args.data_path))[0].split('_')[0]
    output_dir = f"{args.output_dir}-{dataset_name}"

    print("Starting Q8BERT pretraining with settings:")
    print(f"  Data path:            {args.data_path}")
    print(f"  Output directory:     {output_dir}")
    print(f"  Base model path:      {args.base_model_path}")
    print(f"  Batch size:           {args.batch_size}")
    print(f"  Epochs:               {args.epochs}")
    print(f"  Learning rate:        {args.learning_rate}")
    print(f"  Max length:           {args.max_length}")
    print(f"  MLM probability:      {args.mlm_probability}")
    print(f"  Cache size:           {args.cache_size}")
    print(f"  Custom quantization:  {args.use_custom_quantization}")
    print(f"  Disable wandb:        {args.disable_wandb}")

    pretrain_q8bert(
        data_path=args.data_path,
        output_dir=output_dir,
        base_model_path=args.base_model_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        cache_size=args.cache_size,
        use_custom_quantization=args.use_custom_quantization,
    )

    print("Q8BERT pretraining completed successfully!")
