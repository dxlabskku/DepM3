# DepM$^3$: Modality-Agnostic Mixture-of-Experts with Mamba for Depression Detection

**Multi-Encoder Ensemble with Mixture of Experts for Multimodal Depression Detection using State Space Models**

> A multi-encoder ensemble framework that processes multimodal inputs (Audio + Visual + Text) by combining Mamba-based State Space Models with Mixture of Experts for depression detection.

---

## Architecture Overview

<p align="center">
  <img src="architecture.png" width="100%" alt="DepMamba-MoE-Ensemble Architecture"/>
</p>

### Key Components

| Component | Description |
|-----------|-------------|
| **TriModalCoSSM** | Collaborative State Space Model — Processes three modalities (Audio, Visual, Text) in parallel via Bidirectional Mamba, learning cross-modal interactions through a shared state transition matrix A |
| **EnSSM_MoE** | Enhanced SSM with Mixture of Experts — Performs dynamic expert routing on fused multimodal features. Modality-aware gating enables robustness to missing modalities |
| **Ensemble Fusion** | Fuses outputs from multiple encoders via Early (feature-level) or Late (logits-level) fusion strategies |
| **Diversity Loss** | Cosine similarity-based diversity loss between encoders, encouraging each encoder to learn distinct patterns |

---

## Project Structure

```
DepMamba_MoE_ensemble/
├── README.md
├── main_multiencoder_ensemble.py     # End-to-end training script
├── config/
│   ├── config.yaml                   # Base (single encoder) configuration
│   ├── config_moe.yaml               # MoE-specific configuration
│   └── config_ensemble.yaml          # Multi-encoder ensemble configuration
└── models/
    ├── __init__.py
    ├── base.py                       # Base network classes
    ├── DepMamba_multiencoder_ensemble.py  # Main ensemble model
    ├── mamba/
    │   ├── bimamba.py                # Bidirectional Mamba (v1/v2)
    │   ├── mamba_blocks.py           # Mamba block wrappers
    │   ├── mm_bimamba.py             # Multimodal Bidirectional Mamba
    │   ├── trimodal_mamba.py         # Tri-modal Mamba (A+V+T)
    │   └── selective_scan_interface.py   # Selective scan CUDA kernels
    └── moe/
        ├── __init__.py
        └── multimodal_moe.py         # Multimodal Mixture of Experts
```

---

## Method

### 1. TriModal Collaborative SSM (TriModalCoSSM)

Applies independent Bidirectional Mamba encoder layers to each modality (Audio, Visual, Text), while **sharing the state transition matrix A across modalities** to enable collaborative representation learning at the sequence level.

### 2. Mixture of Experts (MoE)

- **ModalityExpert**: Each expert network specializes in specific modality patterns
- **ModalityAwareGating**: Detects which modalities are present in the input and routes to appropriate experts via Top-K sparse routing
- **Load Balancing Loss**: Balances expert utilization through various strategies including KL divergence, variance, combined, and entropy

### 3. Multi-Encoder Ensemble

Multiple encoders are trained in parallel to extract multimodal features from diverse perspectives, with ensemble fusion producing the final prediction.

**Early Fusion** (feature-level):
- `average`, `weighted`, `attention`, `concat`

**Late Fusion** (logits-level, recommended):
- `average`, `weighted`, `attention`, `voting`, `stacking`, `gating`

### 4. Loss Function

The total loss is a weighted sum of three components:

```
L_total = L_cls + λ_div * L_diversity + λ_moe * L_balance
```

- **L_cls**: Binary Cross-Entropy (BCEWithLogitsLoss)
- **L_diversity**: Cosine similarity-based diversity loss between encoder representations
- **L_balance**: MoE expert load balancing loss

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- [mamba-ssm](https://github.com/state-spaces/mamba)
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
- [speechbrain](https://github.com/speechbrain/speechbrain)
- einops
- PyYAML
- tqdm
- numpy

---

## Usage

### Training

```bash
# Default: 3 encoders, late fusion (weighted), MoE enabled
python main_multiencoder_ensemble.py

# Custom configuration
python main_multiencoder_ensemble.py \
    --num_encoders 3 \
    --ensemble_stage late \
    --fusion_type weighted \
    --use_moe True \
    --num_experts 6 \
    --top_k_experts 2 \
    --dataset dvlog \
    --epochs 120 \
    --batch_size 16 \
    --learning_rate 8e-5 \
    --gpu 0

# LMVD dataset
python main_multiencoder_ensemble.py \
    --dataset lmvd \
    --num_encoders 3 \
    --fusion_type weighted
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_encoders` | 3 | Number of encoders in the ensemble |
| `--ensemble_stage` | `late` | Ensemble stage (`early`: feature fusion, `late`: logits ensemble) |
| `--fusion_type` | `weighted` | Ensemble fusion strategy |
| `--use_moe` | `True` | Whether to use MoE in encoders |
| `--num_experts` | 6 | Number of experts per MoE layer |
| `--top_k_experts` | 2 | Number of top-K experts to activate |
| `--diversity_loss_weight` | 0.01 | Weight for diversity loss |
| `--moe_loss_weight` | 0.01 | Weight for MoE load balancing loss |
| `--dataset` | `dvlog` | Dataset selection (`dvlog` / `lmvd`) |
| `--num_iterations` | 3 | Number of training iterations |

### Configuration

Detailed hyperparameters can be adjusted via YAML configuration files:

- `config/config.yaml` — Base single-encoder configuration
- `config/config_moe.yaml` — MoE-specific configuration
- `config/config_ensemble.yaml` — Multi-encoder ensemble configuration (primary config file)

---

## Supported Datasets

| Dataset | Audio Feature | Visual Feature | Text Feature |
|---------|---------------|----------------|--------------|
| **D-Vlog** | LDDs (dim=25) | Facial landmarks (dim=136) | BERT embeddings (dim=768) |
| **LMVD** | VGGish (dim=128) | Facial landmarks (dim=136) | BERT embeddings (dim=768) |

---

## Model Configurations

### Mamba SSM

| Parameter | D-Vlog | LMVD |
|-----------|--------|------|
| `d_state` | 12 | 16 |
| `expand` | 4 | 4 |
| `d_conv` | 4 | 4 |
| `bidirectional` | True | True |

### Ensemble

| Parameter | Value |
|-----------|-------|
| `num_encoders` | 3 |
| `mm_input_size` | 256 |
| `mm_output_sizes` | [256, 64] |
| `num_layers` | 2 |
| `d_ffn` | 1024 |
| `activation` | GELU |

---

## Acknowledgements

- [DepMamba](https://github.com/Jiaxin-Ye/DepMamba) — Based on the original implementation by Jiaxin Ye et al. (2024)
- [Mamba](https://github.com/state-spaces/mamba) — Albert Gu and Tri Dao (2023)
- [SpeechBrain](https://github.com/speechbrain/speechbrain) — Speech processing toolkit

---

## Citation

If you use this code, please cite:

```bibtex
@article{ye2024depmamba,
  title={DepMamba: Progressive Fusion Mamba for Multimodal Depression Detection},
  author={Ye, Jiaxin and others},
  year={2024}
}
```
