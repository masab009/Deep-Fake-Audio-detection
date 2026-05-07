"""
Stage 5: Inference and Adaptation Module
- Threshold-based deepfake scoring
- Few-shot adaptation via prototype centroid updates
- EWC-regularized adaptation for new attacks
- Production-ready inference pipeline
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, InferenceConfig
from data_preprocessing import (
    ASVspoofDataset,
    EpisodicSampler,
    preprocess_audio,
)
from evaluation import compute_eer
from feature_engineering import DualStreamFeatureExtractor
from model import DeepfakeDetector


class InferencePipeline:
    """
    Production inference pipeline for audio deepfake detection.
    Handles single-call scoring and batch processing.
    """

    def __init__(
        self,
        model: DeepfakeDetector,
        feature_extractor: DualStreamFeatureExtractor,
        cfg: Config,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.feature_extractor = feature_extractor.to(device)
        self.cfg = cfg
        self.device = device
        self.threshold = cfg.inference.deepfake_threshold

        self.model.eval()
        self.feature_extractor.eval()

    @torch.inference_mode()
    def _extract_features_from_dataset(
        self,
        dataset: ASVspoofDataset,
        indices: List[int],
        batch_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract fused features for selected dataset items."""
        if not indices:
            raise ValueError("No reference indices were provided for feature extraction")

        feature_batches = []
        label_batches = []

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_items = [dataset[idx] for idx in batch_indices]
            batch_audio = torch.stack([item["audio"] for item in batch_items]).to(self.device)
            batch_labels = torch.stack([item["label"] for item in batch_items]).to(self.device)
            batch_features = self.feature_extractor(batch_audio)

            feature_batches.append(batch_features)
            label_batches.append(batch_labels)

        return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)

    @torch.inference_mode()
    def bootstrap_from_reference_dataset(
        self,
        reference_dataset: ASVspoofDataset,
        support_per_class: int = 50,
        calibration_per_class: int = 100,
        batch_size: int = 32,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Initialize class prototypes and a decision threshold from a labeled
        reference split. This makes single-file scoring meaningful for local
        recordings.
        """
        rng = random.Random(seed)

        bonafide_indices = reference_dataset.metadata.index[
            reference_dataset.metadata["label"] == "bonafide"
        ].tolist()
        spoof_indices = reference_dataset.metadata.index[
            reference_dataset.metadata["label"] == "spoof"
        ].tolist()

        if not bonafide_indices or not spoof_indices:
            raise ValueError("Reference dataset must contain both bonafide and spoof samples")

        rng.shuffle(bonafide_indices)
        rng.shuffle(spoof_indices)

        support_count = min(support_per_class, len(bonafide_indices), len(spoof_indices))
        support_bonafide = bonafide_indices[:support_count]
        support_spoof = spoof_indices[:support_count]

        calibration_bonafide = bonafide_indices[support_count : support_count + calibration_per_class]
        calibration_spoof = spoof_indices[support_count : support_count + calibration_per_class]

        if not calibration_bonafide:
            calibration_bonafide = bonafide_indices[:support_count]
        if not calibration_spoof:
            calibration_spoof = spoof_indices[:support_count]

        support_indices = support_bonafide + support_spoof
        support_labels = torch.tensor(
            [0] * len(support_bonafide) + [1] * len(support_spoof),
            dtype=torch.long,
            device=self.device,
        )
        support_features, _ = self._extract_features_from_dataset(
            reference_dataset,
            support_indices,
            batch_size=batch_size,
        )
        prototypes = self.model.proto_net.compute_prototypes(support_features, support_labels)
        self.model.proto_net.update_prototypes(prototypes)

        calibration_indices = calibration_bonafide + calibration_spoof
        calibration_features, calibration_labels = self._extract_features_from_dataset(
            reference_dataset,
            calibration_indices,
            batch_size=batch_size,
        )
        calibration_scores = self.model.predict(calibration_features, prototypes)
        _, threshold = compute_eer(
            calibration_labels.detach().cpu().numpy(),
            calibration_scores.detach().cpu().numpy(),
        )
        self.threshold = float(threshold)

        return {
            "support_per_class": float(support_count),
            "calibration_bonafide": float(len(calibration_bonafide)),
            "calibration_spoof": float(len(calibration_spoof)),
            "threshold": self.threshold,
        }

    @torch.no_grad()
    def score_audio(self, audio_path: str) -> Dict:
        """
        Score a single audio file.

        Returns:
            dict with 'deepfake_score', 'is_fake', 'segments_scores'
        """
        segments = preprocess_audio(audio_path, self.cfg.audio, apply_codecs=False)

        all_scores = []
        for seg in segments:
            audio_tensor = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(self.device)
            features = self.feature_extractor(audio_tensor)
            score = self.model.predict(features)
            all_scores.append(score.item())

        avg_score = np.mean(all_scores)

        return {
            "deepfake_score": float(avg_score),
            "is_fake": bool(avg_score > self.threshold),
            "segment_scores": [float(score) for score in all_scores],
            "num_segments": int(len(segments)),
        }

    @torch.no_grad()
    def score_batch(self, audio_paths: List[str]) -> List[Dict]:
        """Score a batch of audio files."""
        return [self.score_audio(path) for path in audio_paths]

    @torch.no_grad()
    def score_tensor(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Score preprocessed waveform tensors directly.

        Args:
            waveforms: (batch, num_samples)

        Returns:
            scores: (batch,) deepfake probability scores
        """
        features = self.feature_extractor(waveforms.to(self.device))
        return self.model.predict(features)


class FewShotAdapter:
    """
    Few-shot adaptation module for handling new attack types.
    Updates prototypical centroids without full retraining.
    Uses EWC regularization to prevent catastrophic forgetting.
    """

    def __init__(
        self,
        model: DeepfakeDetector,
        feature_extractor: DualStreamFeatureExtractor,
        cfg: Config,
        device: str = "cpu",
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.cfg = cfg
        self.device = device
        self.momentum = cfg.inference.centroid_momentum

        # Track adaptation history
        self.adaptation_history: List[Dict] = []

    @torch.no_grad()
    def update_prototypes(
        self,
        support_audio: torch.Tensor,
        support_labels: torch.Tensor,
    ):
        """
        Update class prototypes using a few-shot support set.
        Uses momentum-based update to blend old and new centroids.

        Args:
            support_audio: (n_support, num_samples) raw audio
            support_labels: (n_support,) binary labels
        """
        self.model.eval()
        self.feature_extractor.eval()

        features = self.feature_extractor(support_audio.to(self.device))
        new_prototypes = self.model.proto_net.compute_prototypes(
            features, support_labels.to(self.device)
        )

        # Momentum-based update
        old_prototypes = self.model.proto_net.class_prototypes
        if old_prototypes.abs().sum() > 0:  # if prototypes have been initialized
            updated = (
                self.momentum * old_prototypes
                + (1 - self.momentum) * new_prototypes
            )
        else:
            updated = new_prototypes

        self.model.proto_net.update_prototypes(updated)

    def adapt_to_new_attack(
        self,
        support_dataset: ASVspoofDataset,
        attack_name: str,
        k_shot: int = 10,
        num_adaptation_steps: int = 50,
        lr: float = 1e-5,
    ) -> Dict[str, float]:
        """
        Full few-shot adaptation to a newly detected attack type.

        Steps:
        1. Sample support set from confirmed examples of new attack
        2. Update prototypical centroids
        3. Fine-tune with EWC regularization

        Args:
            support_dataset: dataset containing new attack examples
            attack_name: identifier for the new attack
            k_shot: number of support examples
            num_adaptation_steps: gradient steps for fine-tuning
            lr: learning rate for adaptation

        Returns:
            adaptation metrics
        """
        print(f"Adapting to new attack: {attack_name} with {k_shot} examples")

        # Create episodic sampler for the new attack
        sampler = EpisodicSampler(
            dataset=support_dataset,
            n_way=2,
            k_shot=k_shot,
            q_query=min(15, k_shot),
            num_episodes=num_adaptation_steps,
        )

        # Step 1: Update prototypes with support examples
        episode = sampler.sample_fewshot_episode(attack_name, k=k_shot)
        support_items = [support_dataset[i] for i in episode["support_indices"]]
        support_audio = torch.stack([item["audio"] for item in support_items]).to(self.device)
        support_labels = torch.stack([item["label"] for item in support_items]).to(self.device)

        self.update_prototypes(support_audio, support_labels)

        # Step 2: Fine-tune embedding network with EWC
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )

        adaptation_losses = []
        for ep_idx, episode in enumerate(sampler):
            support_items = [support_dataset[i] for i in episode["support_indices"]]
            query_items = [support_dataset[i] for i in episode["query_indices"]]

            s_audio = torch.stack([item["audio"] for item in support_items]).to(self.device)
            s_labels = torch.stack([item["label"] for item in support_items]).to(self.device)
            s_atk = torch.stack([item["attack_type"] for item in support_items]).to(self.device)

            q_audio = torch.stack([item["audio"] for item in query_items]).to(self.device)
            q_labels = torch.stack([item["label"] for item in query_items]).to(self.device)
            q_atk = torch.stack([item["attack_type"] for item in query_items]).to(self.device)

            with torch.no_grad():
                s_features = self.feature_extractor(s_audio)
                q_features = self.feature_extractor(q_audio)

            optimizer.zero_grad()
            losses = self.model.episodic_step(
                s_features, s_labels, q_features, q_labels,
                support_attack_types=s_atk, query_attack_types=q_atk,
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            adaptation_losses.append(losses["total_loss"].item())

        self.model.eval()

        metrics = {
            "attack_name": attack_name,
            "k_shot": k_shot,
            "num_steps": num_adaptation_steps,
            "final_loss": adaptation_losses[-1] if adaptation_losses else 0,
            "avg_loss": np.mean(adaptation_losses) if adaptation_losses else 0,
        }

        self.adaptation_history.append(metrics)
        print(f"Adaptation complete. Final loss: {metrics['final_loss']:.4f}")

        return metrics


class ProductionSystem:
    """
    Production-grade system combining inference and adaptation.
    Simulates the feedback loop described in the paper:
    - Incoming calls scored for deepfakes
    - Flagged calls sent for review
    - New attacks trigger few-shot adaptation
    """

    def __init__(
        self,
        model: DeepfakeDetector,
        feature_extractor: DualStreamFeatureExtractor,
        cfg: Config,
        device: str = "cpu",
    ):
        self.inference = InferencePipeline(model, feature_extractor, cfg, device)
        self.adapter = FewShotAdapter(model, feature_extractor, cfg, device)
        self.cfg = cfg
        self.device = device

        # Score distribution monitoring
        self.score_history: List[float] = []
        self.flagged_calls: List[Dict] = []

    def process_call(self, audio_path: str) -> Dict:
        """
        Process an incoming call through the detection pipeline.
        """
        result = self.inference.score_audio(audio_path)

        self.score_history.append(result["deepfake_score"])

        if result["is_fake"]:
            result["action"] = "FLAG_FOR_REVIEW"
            self.flagged_calls.append(result)
        else:
            result["action"] = "AUTHENTICATE"

        return result

    def trigger_adaptation(
        self,
        support_dataset: ASVspoofDataset,
        attack_name: str,
        k_shot: int = 10,
    ) -> Dict:
        """
        Trigger few-shot adaptation when a new attack is confirmed.
        """
        # Compute Fisher before adaptation (if not already done)
        metrics = self.adapter.adapt_to_new_attack(
            support_dataset, attack_name, k_shot=k_shot
        )

        return metrics

    def get_score_statistics(self) -> Dict:
        """Get summary statistics of recent scores."""
        if not self.score_history:
            return {"count": 0}

        scores = np.array(self.score_history)
        return {
            "count": len(scores),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "median": float(np.median(scores)),
            "flagged_ratio": len(self.flagged_calls) / len(scores),
        }
