# PLM-AL: Encrypted Traffic Classification with Pre-trained Language Models and Active Learning

This project implements a complete **encrypted/malicious network traffic classification** research pipeline, combining Pre-trained Language Models (PLMs) with Active Learning (AL) strategies, supporting PEFT/LoRA fine-tuning.

## Features

- **Multi-model support**: ALBERT, BERT, DistilBERT, RoBERTa, MobileBERT, Q8BERT
- **14+ active learning strategies**: including BASS (Boundary-Aware Sampling), imbalance-aware sampling, BATL, QBC, CoreSet, etc.
- **LoRA/Adapter fine-tuning**: parameter-efficient fine-tuning based on PEFT
- **Complete experiment pipeline**: raw PCAP → feature extraction → MLM pre-training → AL fine-tuning → evaluation

## Requirements

- Python >= 3.8
- PyTorch >= 1.10 (CUDA version recommended)
- GPU memory >= 8GB (16GB+ recommended)

## Installation

```bash
pip install -r requirements.txt
```

Optional dependencies (install as needed):

```bash
pip install bitsandbytes    # Q8BERT quantized pre-training
pip install scapy            # PCAP file processing
```

## Directory Structure

```
.
├── param_config.py              # Unified experiment configuration
├── AL_strategy.py               # Active learning strategy implementations
├── utils/
│   ├── data_loader.py           # Data loading utilities
│   ├── dataset.py               # Dataset classes (TextDataset, PretrainingDataset)
│   ├── metrics.py               # Evaluation metrics and visualization
│   └── cache.py                 # AL strategy state caching
├── s1_merge_pcap.py             # Step 1: PCAP merging
├── s2_feature2npz.py            # Step 2: Feature extraction → NPZ
├── s3_rf_baseline.py            # Step 3: Random Forest baseline
├── s4_pretrain_albert.py        # Step 4: ALBERT MLM pre-training
├── s41_full_ft.py               # Step 4.1: Multi-model pre-train + fine-tune (CLI-driven)
├── s43_pretrain_q8bert.py       # Step 4.3: Q8BERT quantized pre-training
├── s51_al_train.py              # Step 5.1: AL fine-tuning
├── s52_al_compare.py            # Step 5.2: AL strategy comparison
├── s53_al_compare_models.py     # Step 5.3: Cross-model AL comparison
├── s54_al_ablation.py           # Step 5.4: Ablation study
├── s55_al_crossdomain.py        # Step 5.5: Cross-domain experiment
├── eval_cross_s55.py            # Cross-domain model evaluation
└── requirements.txt
```

### Directories to Create at Runtime

```
./base-model/           # HuggingFace base model checkpoints
./data/                 # Training data (NPZ format)
./plmd-model/           # MLM pre-training output
./ftune-model/          # Fine-tuned model output
./results/              # Experiment results output
./logs/                 # Training logs
```

## Data Preparation

### 1. Base Models

Download from HuggingFace and place in the `./base-model/` directory:

```
./base-model/albert-base-v2/
./base-model/bert-base-uncased/
./base-model/distilbert-base-uncased/
./base-model/roberta-base/
./base-model/google-mobilebert-uncased/
./base-model/paraphrase-MiniLM-L6-v2/    # Used by some AL strategies
```

All models are loaded with `local_files_only=True` and must exist locally.

### 2. Training Data

Training data is in NPZ format containing two arrays:
- `features`: feature matrix (CIC statistical features + encoded packet length sequences)
- `labels`: integer label array

Path convention: `./data/<DATASET>_concate_data.npz`

Supported dataset identifiers: `USTC`, `UNSW`, `DoH`, `MalAnd`, etc.

## Full Pipeline

### Step 1: PCAP Merging (Optional)

```bash
python s1_merge_pcap.py --source_dir /path/to/pcaps --target_dir /path/to/merged
```

### Step 2: Feature Extraction

```bash
python s2_feature2npz.py file1.json file2.json -o ./data/USTC_concate_data.npz -b 1000
```

Provide a label mapping file (JSON format):

```bash
python s2_feature2npz.py data/*.json -o ./data/USTC_concate_data.npz --label_mapping_file labels.json
```

### Step 3: Random Forest Baseline

```bash
python s3_rf_baseline.py ./data/USTC_concate_data.npz
```

### Step 4: MLM Pre-training

**ALBERT pre-training:**

```bash
python s4_pretrain_albert.py \
    --data_path ./data/USTC_concate_data.npz \
    --output_dir ./plmd-model/ALBERT-USTC \
    --base_model ./base-model/albert-base-v2 \
    --epochs 3 --batch_size 32
```

**Multi-model pre-train + fine-tune (recommended):**

```bash
python s41_full_ft.py -m albert \
    --data_path ./data/USTC_concate_data.npz \
    --pretrain_epochs 3 --finetune_epochs 3 --batch_size 32
```

Supported models: `albert`, `bert`, `distilbert`, `roberta`, `mobilebert`, `tinybert`

Optional arguments:

```bash
--skip_pretrain          # Skip pre-training, fine-tune directly
--only_pretrain          # Pre-train only
--only_ft                # Fine-tune only (requires existing pre-trained model)
--no_lora                # Full fine-tuning without LoRA
--lora_r 8               # LoRA rank
--lora_alpha 16          # LoRA alpha
--force_pretrain         # Force re-run pre-training
```

**Q8BERT quantized pre-training:**

```bash
python s43_pretrain_q8bert.py \
    --data_path ./data/USTC_concate_data.npz \
    --output_dir ./plmd-model/Q8BERT-USTC
```

### Step 5: Active Learning Fine-tuning

All AL experiment scripts are configured via `param_config.py`. Edit the top-level variables before running:

```python
# Top of param_config.py
dataset = "USTC"              # Dataset
baseModel = "ALBERT"          # Base model
alstrategy = "bass"           # AL strategy
```

**Basic AL training:**

```bash
python s51_al_train.py
```

**AL strategy comparison (with caching):**

```bash
python s52_al_compare.py
```

**Cross-model comparison:**

Set `baseModel` to different models and run:

```bash
python s53_al_compare_models.py
```

**Ablation study:**

Edit ablation parameters in `param_config.py`:

```python
ft_str = "Lora"           # Lora / Adapter / Full
is_random = "False"       # Whether to use random initialization
alstrategy = "bass_no_memory"  # bass_no_memory / bass_no_adversarial / bass_random
```

```bash
python s54_al_ablation.py
```

**Cross-domain experiment:**

```python
dataset = "UNSW"          # Fine-tuning target domain
basePLM = "USTC"          # Pre-training source domain
```

```bash
python s55_al_crossdomain.py
```

**Cross-domain model evaluation:**

```bash
python eval_cross_s55.py \
    --pretrained_model_path ./plmd-model/ALBERT-USTC \
    --ft_model_path ./ftune-model/ALBERT-PreUSTC-FtUNSW-bass \
    --data_path ./data/UNSW_concate_data.npz
```

## Configuration

`param_config.py` is the unified experiment configuration file containing the following modules:

| Config Class | Description |
|--------|------|
| `TrainingArgs` | LoRA parameters, learning rate, batch size, training epochs |
| `DataConfig` | Data path, max sequence length |
| `ModelConfig` | Model path, output directory |
| `LoRAConfig` | LoRA/Adapter configuration |
| `ActiveLearningConfig` | AL strategy parameters (query size, buffer size, tau, etc.) |

Three top-level control variables drive automatic path generation:

```python
dataset    →  data path, output directory
baseModel  →  model path, output directory
alstrategy →  strategy selection, results directory
```

## Supported Active Learning Strategies

| Strategy | Description |
|--------|------|
| `bass` | **BASS** — Boundary-Aware Sampling (core method of this project) |
| `imbalance` | Imbalance-aware sampling |
| `entropy` | Entropy-based sampling |
| `margin` | Margin sampling |
| `confidence` | Least confidence sampling |
| `uncertainty` | Uncertainty sampling |
| `random` | Random sampling |
| `coreset` | CoreSet sampling |
| `diversity` | Diversity sampling |
| `density` | Density-based sampling |
| `graph_density` | Graph density sampling |
| `qbc` | Query-By-Committee |
| `batl` | Batch Acquisition with Triplet Loss |
| `sunb` | SUNB sampling |
| `bass_no_memory` | BASS ablation: no memory mechanism |
| `bass_no_adversarial` | BASS ablation: no adversarial perturbation |
| `bass_random` | BASS ablation: random ranking |

## Output

After each experiment run, results are saved to `./results/<baseModel>-<dataset>-<strategy>/`:

```
results/
├── results/          # Metrics data in Excel format
├── plots/            # Confusion matrices, PR curves, ROC curves
├── models/           # Model checkpoints
└── index/            # AL strategy state cache
```

## Citation

If this project is helpful for your research, please cite:

```bibtex
@article{plm_al_traffic,
  title={TODO: Paper Title},
  author={TODO: Authors},
  journal={TODO: Journal},
  year={2025}
}
```

## License

TODO: Add license information
