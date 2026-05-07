# Few-Shot Continual Learning for Audio Deepfake Detection

This repository implements an audio deepfake detection pipeline based on few-shot learning and continual learning. The system is built around the ASVspoof 2019 LA benchmark and combines dual-stream audio features with a prototypical detector and Elastic Weight Consolidation (EWC) so it can adapt to new spoofing attacks without fully forgetting older ones.

At a high level, the pipeline:

- preprocesses raw audio into fixed 3-second speech segments,
- extracts fused SSL and cepstral features,
- trains on known spoofing attacks,
- evaluates on held-out attacks with few-shot episodes,
- supports post-training adaptation and scoring of local recordings.

## Core Ideas

- Dual-stream features: WavLM + HuBERT embeddings are fused with MFCC, LFCC, and CQCC features.
- Few-shot detection: a prototypical network learns binary genuine-vs-spoof decision boundaries from support/query episodes.
- Continual learning: EWC regularization is used to reduce catastrophic forgetting during adaptation.
- Realistic audio conditions: preprocessing includes resampling, peak normalization, VAD-based trimming, and codec simulation for telephony-style degradation.

## Project Structure

| Path | Purpose |
| --- | --- |
| `main.py` | Main CLI entry point for training, evaluation, adaptation, recordings scoring, and synthetic testing. |
| `config.py` | Central configuration for dataset paths, hyperparameters, model settings, and inference thresholds. |
| `data_preprocessing.py` | Audio loading, normalization, VAD trimming, segmentation, codec simulation, and dataset wrappers. |
| `feature_engineering.py` | Dual-stream feature extraction with SSL and cepstral features. |
| `model.py` | Deepfake detector, prototypical network, and EWC logic. |
| `training.py` | Episodic few-shot training pipeline. |
| `evaluation.py` | EER, accuracy, precision, recall, F1, backward transfer, and evaluation routines. |
| `inference.py` | Local scoring pipeline and few-shot adaptation utilities. |
| `generate_figures.py` | Generates report figures from saved training and evaluation outputs. |
| `sip_listener.py` | Experimental SIP call listener for capturing and optionally scoring recordings. |
| `documentation_and_results/` | Methodology notes and supporting write-up material. |
| `outputs/` | Checkpoints, evaluation JSON files, and cached recording-reference state. |
| `recordings/` | Local audio files to score with `--mode recordings`. |

## Environment Setup

Use a Python virtual environment and install the pinned dependencies from `requirements.txt`.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- A CUDA-capable GPU is strongly recommended for full training and evaluation, but CPU is enough for code inspection and small experiments.
- The first run downloads Hugging Face SSL backbones (`microsoft/wavlm-base-plus` and `facebook/hubert-base-ls960`), so an internet connection is required initially.
- Weights & Biases logging is enabled by default. Use `--no-wandb` if you do not want experiment tracking.

## Dataset Layout

By default the code expects the ASVspoof 2019 LA data under `./data/LA`. You can also override the dataset location with either:

- the `ASVSPOOF_ROOT` environment variable, or
- the `--dataset-root` CLI argument.

Expected layout:

```text
data/LA/
  ASVspoof2019_LA_cm_protocols/
  ASVspoof2019_LA_train/flac/
  ASVspoof2019_LA_dev/flac/
  ASVspoof2019_LA_eval/flac/
```

Training uses known attacks `A01`-`A06`, while evaluation and adaptation focus on held-out attacks `A07`-`A19`.

## Quick Start

If you want to verify the full code path without relying on the dataset, start with synthetic test mode:

```bash
python main.py --mode test
```

This runs a full end-to-end smoke test covering preprocessing, feature extraction, the model forward pass, episodic training logic, adaptation, and evaluation metrics.

## Main CLI Modes

The repository is primarily driven through `main.py`.

### 1. Train on known attacks

```bash
python main.py --mode train --dataset-root ./data/LA --no-wandb
```

Typical optional overrides:

```bash
python main.py --mode train \
  --dataset-root ./data/LA \
  --epochs 10 \
  --episodes 200 \
  --k-shot 5 \
  --lr 1e-4 \
  --ewc-lambda 5000 \
  --no-wandb
```

Artifacts created:

- `outputs/checkpoints/best_model.pt`
- `outputs/checkpoints/final_model_with_ewc.pt`

### 2. Evaluate few-shot performance on held-out attacks

```bash
python main.py --mode evaluate --dataset-root ./data/LA --no-wandb
```

For a faster approximate run:

```bash
python main.py --mode evaluate \
  --dataset-root ./data/LA \
  --eval-episodes 3 \
  --eval-q-query 20 \
  --fast-eval \
  --no-wandb
```

Artifact created:

- `outputs/evaluation_results.json`

### 3. Adapt to a new held-out attack

```bash
python main.py --mode adapt --dataset-root ./data/LA --no-wandb
```

This mode expects `outputs/checkpoints/final_model_with_ewc.pt` to exist.

### 4. Run the full pipeline

```bash
python main.py --mode full --dataset-root ./data/LA --no-wandb
```

This runs training, evaluation, and adaptation in sequence.

### 5. Score local recordings

Place audio files in the `recordings/` directory, then run:

```bash
python main.py --mode recordings --dataset-root ./data/LA --no-wandb
```

Supported extensions are `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`, and `.aac`.

This mode:

- loads `outputs/checkpoints/best_model.pt`,
- bootstraps reference prototypes from the ASVspoof dev split,
- caches the reference state in `outputs/recordings_reference.pt`,
- writes per-file predictions to `outputs/recordings_evaluation.json`.

### 6. Synthetic pipeline test

```bash
python main.py --mode test
```

This is the safest starting point if you are checking the installation or demonstrating the architecture without waiting for a full dataset run.

## Important CLI Options

`main.py` supports these commonly used flags:

- `--dataset-root`: path to the ASVspoof 2019 LA dataset root.
- `--epochs`: number of training epochs.
- `--episodes`: number of training episodes per epoch.
- `--k-shot`: support examples per class for episodic training.
- `--lr`: learning rate.
- `--ewc-lambda`: EWC regularization strength.
- `--seed`: random seed.
- `--eval-episodes`: number of evaluation episodes per attack.
- `--eval-q-query`: number of query examples per class during evaluation.
- `--feature-batch-size`: batch size for evaluation feature extraction.
- `--num-workers`: parallel workers for evaluation preprocessing.
- `--fast-eval`: reduced evaluation workload for faster approximate metrics.
- `--no-wandb`: disable Weights & Biases tracking.

Deeper defaults for paths, feature dimensions, training settings, attack splits, and thresholds are defined in `config.py`.

## Outputs

Common generated artifacts include:

- `outputs/checkpoints/best_model.pt`: best validation checkpoint from training.
- `outputs/checkpoints/final_model_with_ewc.pt`: final trained model plus stored Fisher information and EWC state.
- `outputs/evaluation_results.json`: few-shot evaluation metrics by attack type and shot count.
- `outputs/recordings_reference.pt`: cached prototypes and threshold used for recordings mode.
- `outputs/recordings_evaluation.json`: deepfake scores and verdicts for files in `recordings/`.
- `figures/`: plots generated by `generate_figures.py`.

## Figure Generation

After training and evaluation outputs are available, you can generate report figures with:

```bash
python generate_figures.py
```

The script writes PDF and PNG figures into `figures/`.

## Documentation

For a plain-language explanation of the method and how the implementation maps to the paper, see:

- `documentation_and_results/UNDERSTANDING_THE_PAPER.md`
- `documentation_and_results/methodology_implementation.tex`

## Experimental SIP Capture

The repository also contains an experimental SIP listener:

```bash
python sip_listener.py
python sip_listener.py --model outputs/checkpoints/best_model.pt
```

This script is intended for research capture workflows and writes call audio into `recordings/`. Treat it as an auxiliary utility rather than the main training/evaluation interface.

## Practical Run Order

If you are new to the codebase, this is the simplest order to follow:

1. Install dependencies.
2. Run `python main.py --mode test`.
3. Confirm the ASVspoof dataset path.
4. Train with `--mode train`.
5. Evaluate with `--mode evaluate`.
6. Optionally score local files with `--mode recordings`.

## Troubleshooting

- If training or evaluation is very slow, reduce `--episodes`, `--eval-episodes`, or use `--fast-eval`.
- If model downloads fail on first run, verify internet access and local Hugging Face cache permissions.
- If recordings mode cannot score files, confirm that `best_model.pt` exists and the dataset dev split is available for prototype bootstrapping.
- If W&B initialization fails, rerun with `--no-wandb`.

## License and Data

This repository includes code and supporting outputs for academic/research use. The ASVspoof dataset and any third-party pretrained models remain subject to their respective licenses and usage terms.# Deep-Fake-Audio-detection
