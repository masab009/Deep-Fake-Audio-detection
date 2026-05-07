"""
Main Entry Point — Few-Shot Continual Learning Audio Deepfake Detection
Runs the full pipeline: preprocessing → feature extraction → training → evaluation → adaptation

Usage:
    python main.py --mode train          # Train on known attacks
    python main.py --mode evaluate       # Evaluate on held-out attacks
    python main.py --mode adapt          # Few-shot adaptation demo
    python main.py --mode full           # Run complete pipeline
    python main.py --mode test           # Run with synthetic data (no dataset needed)
"""

import argparse
import os
import sys
import json
import time
import random
from typing import Dict, List

import numpy as np
import torch

from config import Config, get_config
from data_preprocessing import (
    ASVspoofDataset,
    EpisodicSampler,
    preprocess_audio,
    peak_normalize,
    vad_trim,
    segment_audio,
    apply_g711,
    apply_amr_nb,
    apply_opus,
)
from feature_engineering import (
    DualStreamFeatureExtractor,
    SSLFeatureExtractor,
    CepstralFeatureExtractor,
    compute_mfcc,
    compute_lfcc,
    compute_cqcc,
)
from model import DeepfakeDetector, PrototypicalNetwork, EWC
from training import Trainer, train_pipeline, set_seed
from inference import InferencePipeline, FewShotAdapter, ProductionSystem
from evaluation import (
    compute_all_metrics,
    compute_eer,
    compute_backward_transfer,
    evaluate_fewshot_adaptation,
    evaluate_continual_learning,
    print_metrics,
)


# --------------------------------------------------------------------------
# Synthetic data generation for testing without ASVspoof dataset
# --------------------------------------------------------------------------

def generate_synthetic_dataset(
    num_genuine: int = 200,
    num_spoof: int = 600,
    num_attacks: int = 6,
    sr: int = 16000,
    duration: float = 3.0,
) -> Dict:
    """
    Generate synthetic audio data that mimics ASVspoof structure.
    Genuine: random speech-like signal with formant structure
    Spoof: genuine + synthesis artifacts (harmonics, noise patterns)
    """
    samples_per_seg = int(sr * duration)
    data = {"audio": [], "labels": [], "attack_types": [], "utterance_ids": []}

    # Generate genuine samples
    for i in range(num_genuine):
        t = np.linspace(0, duration, samples_per_seg, dtype=np.float32)
        # Simulate speech: fundamental + formants + noise
        f0 = np.random.uniform(80, 300)
        signal = (
            0.5 * np.sin(2 * np.pi * f0 * t)
            + 0.3 * np.sin(2 * np.pi * f0 * 2 * t)
            + 0.1 * np.sin(2 * np.pi * f0 * 3 * t)
            + 0.05 * np.random.randn(samples_per_seg)
        ).astype(np.float32)
        signal = signal / (np.max(np.abs(signal)) + 1e-8)
        data["audio"].append(signal)
        data["labels"].append(0)
        data["attack_types"].append(0)  # bonafide
        data["utterance_ids"].append(f"genuine_{i:04d}")

    # Generate spoof samples with different attack characteristics
    attacks_per_type = num_spoof // num_attacks
    for atk_idx in range(num_attacks):
        for i in range(attacks_per_type):
            t = np.linspace(0, duration, samples_per_seg, dtype=np.float32)
            f0 = np.random.uniform(80, 300)

            # Base signal
            signal = 0.5 * np.sin(2 * np.pi * f0 * t)

            # Attack-specific artifacts
            if atk_idx == 0:  # TTS-like: over-smooth, missing micro-prosody
                signal += 0.2 * np.sin(2 * np.pi * f0 * 2 * t)
            elif atk_idx == 1:  # VC-like: spectral mismatch
                signal += 0.3 * np.sin(2 * np.pi * (f0 * 1.5) * t)
                signal += 0.1 * np.random.randn(samples_per_seg)
            elif atk_idx == 2:  # Neural TTS: very smooth with periodic artifacts
                signal = 0.6 * np.sin(2 * np.pi * f0 * t)
                signal += 0.15 * np.sin(2 * np.pi * 50 * t)  # 50Hz buzz
            elif atk_idx == 3:  # Vocoder artifacts
                signal += 0.2 * np.sign(np.sin(2 * np.pi * f0 * 4 * t))
            elif atk_idx == 4:  # Replay-like: room reverb simulation
                signal = np.convolve(signal, np.random.randn(100) * 0.01, mode='same')
                signal += 0.05 * np.random.randn(samples_per_seg)
            else:  # Generic synthesis artifacts
                signal += 0.15 * np.sin(2 * np.pi * (f0 + 50 * atk_idx) * t)
                signal += 0.08 * np.random.randn(samples_per_seg)

            signal = signal.astype(np.float32)
            signal = signal / (np.max(np.abs(signal)) + 1e-8)
            data["audio"].append(signal)
            data["labels"].append(1)
            data["attack_types"].append(atk_idx + 1)
            data["utterance_ids"].append(f"spoof_A{atk_idx+1:02d}_{i:04d}")

    return data


class SyntheticDataset(torch.utils.data.Dataset):
    """Dataset wrapper for synthetic data."""

    def __init__(self, data: Dict, attack_filter=None):
        self.audio = data["audio"]
        self.labels = data["labels"]
        self.attack_types = data["attack_types"]
        self.utterance_ids = data["utterance_ids"]

        if attack_filter is not None:
            indices = [
                i for i in range(len(self.labels))
                if self.attack_types[i] in attack_filter or self.labels[i] == 0
            ]
            self.audio = [self.audio[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.attack_types = [self.attack_types[i] for i in indices]
            self.utterance_ids = [self.utterance_ids[i] for i in indices]

        # Build metadata-like structure for EpisodicSampler compatibility
        import pandas as pd
        self.metadata = pd.DataFrame({
            "utterance_id": self.utterance_ids,
            "label": ["bonafide" if l == 0 else "spoof" for l in self.labels],
            "attack_type": [f"A{a:02d}" if a > 0 else "-" for a in self.attack_types],
        })
        self.label_map = {"bonafide": 0, "spoof": 1}
        all_attacks = sorted(self.metadata["attack_type"].unique().tolist())
        self.attack_map = {a: i for i, a in enumerate(all_attacks)}

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return {
            "audio": torch.tensor(self.audio[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "attack_type": torch.tensor(self.attack_types[idx], dtype=torch.long),
            "utterance_id": self.utterance_ids[idx],
            "attack_name": f"A{self.attack_types[idx]:02d}" if self.attack_types[idx] > 0 else "-",
        }


# --------------------------------------------------------------------------
# Test mode: validates entire pipeline with synthetic data
# --------------------------------------------------------------------------

def run_test_mode(cfg: Config):
    """Run complete pipeline test with synthetic data — no real dataset needed."""
    print("=" * 60)
    print("  PIPELINE TEST MODE — Synthetic Data")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    set_seed(cfg.seed)

    # --- Stage 1: Data Preprocessing Test ---
    print("\n[Stage 1] Testing Data Preprocessing...")
    sr = cfg.audio.sample_rate
    duration = cfg.audio.segment_duration
    raw_audio = np.random.randn(int(sr * 5)).astype(np.float32) * 0.5

    # Peak normalization
    normalized = peak_normalize(raw_audio)
    assert np.max(np.abs(normalized)) <= 1.0 + 1e-6, "Peak normalization failed"
    print("  ✓ Peak normalization")

    # Codec simulation
    g711_out = apply_g711(normalized)
    assert g711_out.shape == normalized.shape, "G.711 shape mismatch"
    print("  ✓ G.711 codec simulation")

    amr_out = apply_amr_nb(normalized, sr)
    assert amr_out.shape == normalized.shape, "AMR-NB shape mismatch"
    print("  ✓ AMR-NB codec simulation")

    opus_out = apply_opus(normalized, sr)
    assert opus_out.shape == normalized.shape, "Opus shape mismatch"
    print("  ✓ Opus codec simulation")

    # VAD trimming
    trimmed = vad_trim(normalized, sr)
    assert len(trimmed) > 0, "VAD trimmed everything"
    print(f"  ✓ VAD trimming ({len(normalized)} → {len(trimmed)} samples)")

    # Segmentation
    segments = segment_audio(normalized, sr, duration, cfg.audio.segment_overlap)
    expected_samples = int(duration * sr)
    for seg in segments:
        assert len(seg) == expected_samples, f"Segment length {len(seg)} != {expected_samples}"
    print(f"  ✓ Sliding window segmentation ({len(segments)} segments)")

    # --- Stage 2: Feature Engineering Test ---
    print("\n[Stage 2] Testing Feature Engineering...")

    # Cepstral features
    cepstral_ext = CepstralFeatureExtractor(cfg.features)
    test_segment = segments[0]
    cepstral_feats = cepstral_ext.extract(test_segment, sr)
    expected_cepstral_dim = cfg.features.n_mfcc + cfg.features.n_lfcc + cfg.features.n_cqcc
    assert cepstral_feats.shape == (expected_cepstral_dim,), \
        f"Cepstral dim {cepstral_feats.shape} != ({expected_cepstral_dim},)"
    print(f"  ✓ Cepstral features: MFCC({cfg.features.n_mfcc}) + LFCC({cfg.features.n_lfcc}) + CQCC({cfg.features.n_cqcc})")

    # MFCC
    mfcc = compute_mfcc(test_segment, sr, n_mfcc=cfg.features.n_mfcc)
    assert mfcc.shape == (cfg.features.n_mfcc,)
    print(f"  ✓ MFCC: {mfcc.shape}")

    # LFCC
    lfcc = compute_lfcc(test_segment, sr, n_lfcc=cfg.features.n_lfcc)
    assert lfcc.shape == (cfg.features.n_lfcc,)
    print(f"  ✓ LFCC: {lfcc.shape}")

    # CQCC
    cqcc = compute_cqcc(test_segment, sr, n_cqcc=cfg.features.n_cqcc)
    assert cqcc.shape == (cfg.features.n_cqcc,)
    print(f"  ✓ CQCC: {cqcc.shape}")

    # SSL features (downloads models on first run)
    print("  Loading SSL models (WavLM + HuBERT)...")
    ssl_ext = SSLFeatureExtractor(cfg.features, device=device)
    ssl_ext = ssl_ext.to(device)
    test_tensor = torch.tensor(test_segment).unsqueeze(0).to(device)
    ssl_feats = ssl_ext(test_tensor)
    assert ssl_feats.shape == (1, cfg.features.ssl_embedding_dim * 2), \
        f"SSL dim {ssl_feats.shape} != (1, {cfg.features.ssl_embedding_dim * 2})"
    print(f"  ✓ SSL features (WavLM + HuBERT): {ssl_feats.shape}")

    # Full dual-stream
    dual_ext = DualStreamFeatureExtractor(cfg.features, device=device)
    dual_ext = dual_ext.to(device)
    fused = dual_ext(test_tensor)
    assert fused.shape == (1, cfg.features.fused_dim), \
        f"Fused dim {fused.shape} != (1, {cfg.features.fused_dim})"
    print(f"  ✓ Dual-stream fused features: {fused.shape} (dim={cfg.features.fused_dim})")

    # --- Stage 3: Model Architecture Test ---
    print("\n[Stage 3] Testing Model Architecture...")

    model = DeepfakeDetector(
        feature_dim=cfg.features.fused_dim,
        model_cfg=cfg.model,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Forward pass (eval mode for batch_size=1 with BatchNorm)
    model.eval()
    output = model(fused)
    assert "embeddings" in output
    assert "classification_logits" in output
    assert "attack_logits" in output
    assert output["embeddings"].shape == (1, cfg.model.embedding_dim)
    assert output["classification_logits"].shape == (1, 2)  # binary
    print(f"  ✓ Forward pass: embeddings={output['embeddings'].shape}, cls={output['classification_logits'].shape}")

    # Prototypical loss test
    model.train()
    batch_size = 10
    dummy_features = torch.randn(batch_size, cfg.features.fused_dim).to(device)
    dummy_labels = torch.randint(0, 2, (batch_size,)).to(device)

    support_f = dummy_features[:5]
    support_l = dummy_labels[:5]
    query_f = dummy_features[5:]
    query_l = dummy_labels[5:]

    proto_loss, proto_acc = model.proto_net.prototypical_loss(
        support_f, support_l, query_f, query_l
    )
    print(f"  ✓ Prototypical loss: {proto_loss.item():.4f}, acc: {proto_acc.item():.4f}")

    # Full episodic step
    losses = model.episodic_step(
        support_f, support_l, query_f, query_l,
        support_attack_types=torch.randint(0, 7, (5,)).to(device),
        query_attack_types=torch.randint(0, 7, (5,)).to(device),
    )
    print(f"  ✓ Episodic step: total_loss={losses['total_loss'].item():.4f}")

    # --- Stage 4: Training Test (mini) ---
    print("\n[Stage 4] Testing Training Pipeline (mini episodes)...")

    # Generate synthetic data
    syn_data = generate_synthetic_dataset(
        num_genuine=100, num_spoof=300, num_attacks=6
    )

    # Known attacks for training: 1-4, held-out: 5-6
    known_attack_ids = [0, 1, 2, 3, 4]  # 0=bonafide included via filter
    held_out_attack_ids = [5, 6]

    train_syn = SyntheticDataset(syn_data, attack_filter=known_attack_ids)
    eval_syn = SyntheticDataset(syn_data, attack_filter=held_out_attack_ids)

    print(f"  Synthetic train set: {len(train_syn)} samples")
    print(f"  Synthetic eval set: {len(eval_syn)} samples")

    # Episodic sampler
    train_sampler = EpisodicSampler(
        dataset=train_syn,
        n_way=2,
        k_shot=5,
        q_query=10,
        num_episodes=5,
    )

    # Mini training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for ep_idx, episode in enumerate(train_sampler):
        items_s = [train_syn[i] for i in episode["support_indices"]]
        items_q = [train_syn[i] for i in episode["query_indices"]]

        s_audio = torch.stack([it["audio"] for it in items_s]).to(device)
        s_labels = torch.stack([it["label"] for it in items_s]).to(device)
        s_atk = torch.stack([it["attack_type"] for it in items_s]).to(device)
        q_audio = torch.stack([it["audio"] for it in items_q]).to(device)
        q_labels = torch.stack([it["label"] for it in items_q]).to(device)
        q_atk = torch.stack([it["attack_type"] for it in items_q]).to(device)

        with torch.no_grad():
            s_features = dual_ext(s_audio)
            q_features = dual_ext(q_audio)

        optimizer.zero_grad()
        losses = model.episodic_step(
            s_features, s_labels, q_features, q_labels,
            support_attack_types=s_atk, query_attack_types=q_atk,
        )
        losses["total_loss"].backward()
        optimizer.step()

        print(
            f"  Episode {ep_idx+1}: loss={losses['total_loss'].item():.4f} "
            f"proto_acc={losses['proto_acc'].item():.4f} "
            f"cls_acc={losses['cls_acc'].item():.4f}"
        )

    print("  ✓ Episodic training loop works")

    # --- EWC Test ---
    print("\n  Computing Fisher Information (EWC)...")
    fisher_loader = torch.utils.data.DataLoader(train_syn, batch_size=16, shuffle=True)
    model.compute_and_store_fisher(fisher_loader, dual_ext, device, num_samples=50)
    ewc_pen = model.ewc.penalty()
    print(f"  ✓ EWC penalty after Fisher computation: {ewc_pen.item():.6f}")

    # --- Stage 5: Inference & Adaptation Test ---
    print("\n[Stage 5] Testing Inference & Adaptation...")

    model.eval()

    # Test prediction
    test_audio = torch.randn(4, int(sr * duration)).to(device)
    with torch.no_grad():
        test_features = dual_ext(test_audio)

    # Update prototypes first
    proto_support = torch.randn(20, cfg.features.fused_dim).to(device)
    proto_labels = torch.cat([torch.zeros(10), torch.ones(10)]).long().to(device)
    model.update_prototypes(proto_support, proto_labels)

    scores = model.predict(test_features)
    print(f"  ✓ Deepfake scores: {scores.detach().cpu().numpy()}")

    # Few-shot adaptation simulation
    print("\n  Simulating few-shot adaptation to novel attack...")
    adapter = FewShotAdapter(model, dual_ext, cfg, device=device)

    novel_support_audio = torch.randn(10, int(sr * duration)).to(device)
    novel_support_labels = torch.cat([torch.zeros(5), torch.ones(5)]).long()
    adapter.update_prototypes(novel_support_audio, novel_support_labels)
    print("  ✓ Prototype centroid update (momentum-based)")

    # Re-score after adaptation
    scores_after = model.predict(test_features)
    print(f"  ✓ Scores after adaptation: {scores_after.detach().cpu().numpy()}")

    # --- Evaluation Metrics Test ---
    print("\n[Evaluation] Testing Metrics Computation...")

    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9, 0.6, 0.4, 0.85, 0.15])

    metrics = compute_all_metrics(y_true, y_scores)
    print_metrics(metrics, prefix="Test Metrics")

    # BWT test
    perf_matrix = np.array([
        [0.95, 0.0],
        [0.90, 0.92],
    ])
    bwt = compute_backward_transfer(perf_matrix)
    print(f"  ✓ Backward Transfer: {bwt:.4f} (negative = forgetting)")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  ALL PIPELINE TESTS PASSED")
    print("=" * 60)
    print(f"\n  Pipeline components verified:")
    print(f"    1. Data Preprocessing (resample, normalize, codecs, VAD, segmentation)")
    print(f"    2. Feature Engineering (WavLM + HuBERT + MFCC + LFCC + CQCC)")
    print(f"    3. Model (Prototypical Network + EWC + classification heads)")
    print(f"    4. Training (episodic few-shot protocol)")
    print(f"    5. Inference & Adaptation (scoring + prototype updates + EWC)")
    print(f"    6. Evaluation Metrics (EER, Accuracy, Precision, Recall, F1, BWT)")
    print(f"\n  Fused feature dimension: {cfg.features.fused_dim}")
    print(f"  Embedding dimension: {cfg.model.embedding_dim}")
    print(f"  Model parameters: {trainable_params:,}")
    print(f"  Device: {device}")

    return True


# --------------------------------------------------------------------------
# Full pipeline with real ASVspoof dataset
# --------------------------------------------------------------------------

def run_train_mode(cfg: Config):
    """Train on ASVspoof 2019 LA dataset."""
    print("Starting training on ASVspoof 2019 LA...")
    model = train_pipeline(cfg)
    return model


def run_evaluate_mode(
    cfg: Config,
    eval_num_episodes: int = 10,
    eval_q_query: int = 50,
    feature_batch_size: int = None,
    num_workers: int = None,
    fast_eval: bool = False,
):
    """Evaluate on held-out attacks from ASVspoof 2019 LA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if fast_eval:
        eval_num_episodes = min(eval_num_episodes, 3)
        eval_q_query = min(eval_q_query, 20)
        if feature_batch_size is None:
            feature_batch_size = 128 if device == "cuda" else 32
        print(
            "Fast evaluation enabled: using fewer episodes and queries "
            "for approximate metrics."
        )

    # Load model
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Train first.")
        return

    feature_extractor = DualStreamFeatureExtractor(cfg.features, device=device).to(device)
    model = DeepfakeDetector(cfg.features.fused_dim, cfg.model).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Eval dataset with held-out attacks
    eval_dataset = ASVspoofDataset(
        protocol_path=cfg.paths.eval_protocol_path,
        flac_dir=cfg.paths.eval_flac_dir,
        cfg=cfg.audio,
        apply_codecs=False,
    )

    print(f"Evaluation set: {len(eval_dataset)} samples")

    # Few-shot evaluation across different k values
    results = evaluate_fewshot_adaptation(
        model, feature_extractor, eval_dataset,
        attack_types=cfg.train.held_out_attacks,
        k_shots=cfg.train.few_shot_k,
        device=device,
        num_episodes=eval_num_episodes,
        q_query=eval_q_query,
        feature_batch_size=feature_batch_size,
        num_workers=num_workers,
    )

    # Print results
    for attack in results:
        for k_key, metrics in results[attack].items():
            print_metrics(metrics, prefix=f"{attack} — {k_key}")

    # Save results
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    results_path = os.path.join(cfg.paths.output_dir, "evaluation_results.json")
    # Convert to serializable
    serializable = {}
    for attack in results:
        serializable[attack] = {}
        for k_key, metrics in results[attack].items():
            serializable[attack][k_key] = {k: float(v) for k, v in metrics.items()}

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {results_path}")


def run_adapt_mode(cfg: Config):
    """Demonstrate few-shot adaptation on a held-out attack."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, "final_model_with_ewc.pt")
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Train first.")
        return

    feature_extractor = DualStreamFeatureExtractor(cfg.features, device=device).to(device)
    model = DeepfakeDetector(cfg.features.fused_dim, cfg.model).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore EWC state
    if "ewc_fisher" in checkpoint:
        model.ewc.fisher_info = checkpoint["ewc_fisher"]
        model.ewc.saved_params = checkpoint["ewc_params"]

    # Eval dataset
    eval_dataset = ASVspoofDataset(
        protocol_path=cfg.paths.eval_protocol_path,
        flac_dir=cfg.paths.eval_flac_dir,
        cfg=cfg.audio,
        apply_codecs=False,
    )

    # Production system
    system = ProductionSystem(model, feature_extractor, cfg, device=device)

    # Simulate adaptation to first held-out attack
    target_attack = cfg.train.held_out_attacks[0]
    print(f"\nAdapting to new attack: {target_attack}")

    adaptation_metrics = system.trigger_adaptation(
        eval_dataset, target_attack, k_shot=10
    )
    print(f"Adaptation metrics: {adaptation_metrics}")

    # Evaluate post-adaptation
    print("\nPost-adaptation evaluation...")
    eval_sampler = EpisodicSampler(
        dataset=eval_dataset, n_way=2, k_shot=5, q_query=20, num_episodes=50
    )

    all_scores = []
    all_labels = []
    model.eval()

    for episode in eval_sampler:
        items_s = [eval_dataset[i] for i in episode["support_indices"]]
        items_q = [eval_dataset[i] for i in episode["query_indices"]]

        s_audio = torch.stack([it["audio"] for it in items_s]).to(device)
        s_labels = torch.stack([it["label"] for it in items_s]).to(device)
        q_audio = torch.stack([it["audio"] for it in items_q]).to(device)
        q_labels = torch.stack([it["label"] for it in items_q])

        with torch.no_grad():
            s_features = feature_extractor(s_audio)
            q_features = feature_extractor(q_audio)

        prototypes = model.proto_net.compute_prototypes(s_features, s_labels)
        scores = model.predict(q_features, prototypes)
        all_scores.extend(scores.cpu().numpy().tolist())
        all_labels.extend(q_labels.numpy().tolist())

    metrics = compute_all_metrics(np.array(all_labels), np.array(all_scores))
    print_metrics(metrics, prefix=f"Post-Adaptation ({target_attack})")


def run_recordings_mode(cfg: Config):
    """Score local recordings in the project recordings directory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Train first.")
        return

    feature_extractor = DualStreamFeatureExtractor(cfg.features, device=device).to(device)
    model = DeepfakeDetector(cfg.features.fused_dim, cfg.model).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pipeline = InferencePipeline(model, feature_extractor, cfg, device=device)

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    reference_cache_path = os.path.join(cfg.paths.output_dir, "recordings_reference.pt")
    if os.path.exists(reference_cache_path):
        reference_state = torch.load(reference_cache_path, map_location=device)
        pipeline.model.proto_net.update_prototypes(reference_state["prototypes"].to(device))
        pipeline.threshold = float(reference_state["threshold"])
        print(
            f"Loaded cached reference prototypes and threshold "
            f"({pipeline.threshold:.4f})"
        )
    else:
        reference_dataset = ASVspoofDataset(
            protocol_path=cfg.paths.dev_protocol_path,
            flac_dir=cfg.paths.dev_flac_dir,
            cfg=cfg.audio,
            attack_filter=cfg.train.known_attacks,
            apply_codecs=False,
        )
        bootstrap_stats = pipeline.bootstrap_from_reference_dataset(
            reference_dataset,
            support_per_class=50,
            calibration_per_class=100,
            batch_size=64 if device == "cuda" else 16,
            seed=cfg.seed,
        )
        torch.save(
            {
                "prototypes": pipeline.model.proto_net.class_prototypes.detach().cpu(),
                "threshold": pipeline.threshold,
                "stats": bootstrap_stats,
            },
            reference_cache_path,
        )
        print(
            f"Bootstrapped reference prototypes from dev split with threshold "
            f"{pipeline.threshold:.4f}"
        )

    recordings_dir = os.path.join(os.path.dirname(__file__), "recordings")
    if not os.path.isdir(recordings_dir):
        print(f"Recordings directory not found: {recordings_dir}")
        return

    audio_extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}
    recording_paths = [
        os.path.join(recordings_dir, name)
        for name in sorted(os.listdir(recordings_dir))
        if os.path.splitext(name)[1].lower() in audio_extensions
    ]

    if not recording_paths:
        print(f"No supported audio files found in {recordings_dir}")
        return

    print(f"Scoring {len(recording_paths)} recording(s) from {recordings_dir}")
    print(f"Using deepfake threshold: {pipeline.threshold:.4f}")

    results = []
    for audio_path in recording_paths:
        result = pipeline.score_audio(audio_path)
        result["file"] = os.path.basename(audio_path)
        results.append(result)
        verdict = "FAKE" if result["is_fake"] else "GENUINE"
        print(
            f"  {result['file']}: score={result['deepfake_score']:.4f} "
            f"verdict={verdict} segments={result['num_segments']}"
        )

    results_path = os.path.join(cfg.paths.output_dir, "recordings_evaluation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


def run_full_pipeline(cfg: Config):
    """Run the complete pipeline: train → evaluate → adapt."""
    print("\n" + "=" * 60)
    print("  FULL PIPELINE EXECUTION")
    print("=" * 60)

    print("\n--- Phase 1: Training ---")
    model = run_train_mode(cfg)

    print("\n--- Phase 2: Evaluation ---")
    run_evaluate_mode(cfg)

    print("\n--- Phase 3: Adaptation ---")
    run_adapt_mode(cfg)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Few-Shot Continual Learning Audio Deepfake Detection"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["train", "evaluate", "adapt", "full", "test", "recordings"],
        help="Pipeline mode to run",
    )
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="Path to ASVspoof 2019 LA dataset")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--k-shot", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ewc-lambda", type=float, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Episodes per attack during evaluation")
    parser.add_argument("--eval-q-query", type=int, default=50,
                        help="Query samples per class during evaluation")
    parser.add_argument("--feature-batch-size", type=int, default=None,
                        help="Batch size for batched feature extraction during evaluation")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Parallel workers for evaluation preprocessing")
    parser.add_argument("--fast-eval", action="store_true",
                        help="Run a faster approximate evaluation with fewer episodes and queries")

    args = parser.parse_args()

    # Build config with overrides
    overrides = {"seed": args.seed}
    if args.dataset_root:
        overrides["paths.dataset_root"] = args.dataset_root
    if args.epochs:
        overrides["train.num_epochs"] = args.epochs
    if args.episodes:
        overrides["train.num_episodes_train"] = args.episodes
    if args.k_shot:
        overrides["train.k_shot"] = args.k_shot
    if args.lr:
        overrides["train.learning_rate"] = args.lr
    if args.ewc_lambda:
        overrides["model.ewc_lambda"] = args.ewc_lambda
    if args.no_wandb:
        overrides["train.use_wandb"] = False

    cfg = get_config(**overrides)

    if args.mode == "test":
        run_test_mode(cfg)
    elif args.mode == "train":
        run_train_mode(cfg)
    elif args.mode == "evaluate":
        run_evaluate_mode(
            cfg,
            eval_num_episodes=args.eval_episodes,
            eval_q_query=args.eval_q_query,
            feature_batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            fast_eval=args.fast_eval,
        )
    elif args.mode == "adapt":
        run_adapt_mode(cfg)
    elif args.mode == "full":
        run_full_pipeline(cfg)
    elif args.mode == "recordings":
        run_recordings_mode(cfg)


if __name__ == "__main__":
    main()
