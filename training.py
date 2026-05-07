"""
Stage 4: Training Pipeline Module
- Episodic few-shot training on ASVspoof 2019 LA
- Holds out attack types to simulate unseen attacks
- Integrates EWC for continual learning
- W&B experiment tracking
"""

import os
import time
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import Config
from data_preprocessing import ASVspoofDataset, EpisodicSampler
from feature_engineering import DualStreamFeatureExtractor
from model import DeepfakeDetector
from evaluation import compute_all_metrics


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_episode(dataset, indices, device):
    """Collate a list of dataset indices into batched tensors."""
    items = [dataset[i] for i in indices]
    audio = torch.stack([item["audio"] for item in items]).to(device)
    labels = torch.stack([item["label"] for item in items]).to(device)
    attack_types = torch.stack([item["attack_type"] for item in items]).to(device)
    return audio, labels, attack_types


class Trainer:
    """
    Episodic few-shot trainer for the DeepfakeDetector.
    """

    def __init__(self, cfg: Config, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize W&B
        self.use_wandb = cfg.train.use_wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=cfg.train.wandb_project,
                    entity=cfg.train.wandb_entity,
                    config={
                        "n_way": cfg.train.n_way,
                        "k_shot": cfg.train.k_shot,
                        "q_query": cfg.train.q_query,
                        "num_episodes_train": cfg.train.num_episodes_train,
                        "num_epochs": cfg.train.num_epochs,
                        "learning_rate": cfg.train.learning_rate,
                        "ewc_lambda": cfg.model.ewc_lambda,
                        "embedding_dim": cfg.model.embedding_dim,
                        "hidden_dim": cfg.model.hidden_dim,
                    },
                )
                self.wandb = wandb
            except Exception as e:
                print(f"W&B initialization failed: {e}. Continuing without tracking.")
                self.use_wandb = False

    def setup_data(self) -> Tuple[ASVspoofDataset, ASVspoofDataset, EpisodicSampler, EpisodicSampler]:
        """Set up training and dev datasets with episodic samplers."""
        # Training dataset — only known attacks
        train_dataset = ASVspoofDataset(
            protocol_path=self.cfg.paths.train_protocol_path,
            flac_dir=self.cfg.paths.train_flac_dir,
            cfg=self.cfg.audio,
            attack_filter=self.cfg.train.known_attacks,
            apply_codecs=True,
        )

        # Dev dataset — known attacks for validation
        dev_dataset = ASVspoofDataset(
            protocol_path=self.cfg.paths.dev_protocol_path,
            flac_dir=self.cfg.paths.dev_flac_dir,
            cfg=self.cfg.audio,
            attack_filter=self.cfg.train.known_attacks,
            apply_codecs=False,
        )

        # Episodic samplers
        train_sampler = EpisodicSampler(
            dataset=train_dataset,
            n_way=self.cfg.train.n_way,
            k_shot=self.cfg.train.k_shot,
            q_query=self.cfg.train.q_query,
            num_episodes=self.cfg.train.num_episodes_train,
        )

        dev_sampler = EpisodicSampler(
            dataset=dev_dataset,
            n_way=self.cfg.train.n_way,
            k_shot=self.cfg.train.k_shot,
            q_query=self.cfg.train.q_query,
            num_episodes=self.cfg.train.num_episodes_eval,
        )

        return train_dataset, dev_dataset, train_sampler, dev_sampler

    def setup_model(self) -> Tuple[DeepfakeDetector, DualStreamFeatureExtractor]:
        """Initialize model and feature extractor."""
        feature_extractor = DualStreamFeatureExtractor(
            cfg=self.cfg.features,
            device=self.device,
        ).to(self.device)

        model = DeepfakeDetector(
            feature_dim=self.cfg.features.fused_dim,
            model_cfg=self.cfg.model,
        ).to(self.device)

        return model, feature_extractor

    def train_epoch(
        self,
        model: DeepfakeDetector,
        feature_extractor: DualStreamFeatureExtractor,
        dataset: ASVspoofDataset,
        sampler: EpisodicSampler,
        optimizer: optim.Optimizer,
    ) -> Dict[str, float]:
        """Train for one epoch (a set of episodes)."""
        model.train()
        feature_extractor.eval()  # SSL models always frozen

        epoch_metrics = {
            "total_loss": 0, "proto_loss": 0, "cls_loss": 0,
            "atk_loss": 0, "proto_acc": 0, "cls_acc": 0,
            "ewc_penalty": 0,
        }
        num_episodes = 0

        for episode in sampler:
            support_audio, support_labels, support_atk = collate_episode(
                dataset, episode["support_indices"], self.device
            )
            query_audio, query_labels, query_atk = collate_episode(
                dataset, episode["query_indices"], self.device
            )

            # Extract features
            with torch.no_grad():
                support_features = feature_extractor(support_audio)
                query_features = feature_extractor(query_audio)

            # Episodic step
            optimizer.zero_grad()
            losses = model.episodic_step(
                support_features, support_labels,
                query_features, query_labels,
                support_attack_types=support_atk,
                query_attack_types=query_atk,
                cls_weight=self.cfg.train.classification_weight,
                atk_weight=self.cfg.train.attack_type_weight,
                proto_weight=self.cfg.train.prototypical_weight,
            )

            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in epoch_metrics:
                epoch_metrics[k] += losses[k].item()
            num_episodes += 1

        # Average
        for k in epoch_metrics:
            epoch_metrics[k] /= max(num_episodes, 1)

        return epoch_metrics

    @torch.no_grad()
    def evaluate_epoch(
        self,
        model: DeepfakeDetector,
        feature_extractor: DualStreamFeatureExtractor,
        dataset: ASVspoofDataset,
        sampler: EpisodicSampler,
    ) -> Dict[str, float]:
        """Evaluate on dev set episodes."""
        model.eval()
        feature_extractor.eval()

        epoch_metrics = {
            "total_loss": 0, "proto_loss": 0, "cls_loss": 0,
            "proto_acc": 0, "cls_acc": 0,
        }
        num_episodes = 0

        for episode in sampler:
            support_audio, support_labels, support_atk = collate_episode(
                dataset, episode["support_indices"], self.device
            )
            query_audio, query_labels, query_atk = collate_episode(
                dataset, episode["query_indices"], self.device
            )

            support_features = feature_extractor(support_audio)
            query_features = feature_extractor(query_audio)

            losses = model.episodic_step(
                support_features, support_labels,
                query_features, query_labels,
                support_attack_types=support_atk,
                query_attack_types=query_atk,
            )

            epoch_metrics["total_loss"] += losses["total_loss"].item()
            epoch_metrics["proto_loss"] += losses["proto_loss"].item()
            epoch_metrics["cls_loss"] += losses["cls_loss"].item()
            epoch_metrics["proto_acc"] += losses["proto_acc"].item()
            epoch_metrics["cls_acc"] += losses["cls_acc"].item()
            num_episodes += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= max(num_episodes, 1)

        return epoch_metrics

    def train(self) -> DeepfakeDetector:
        """Full training loop."""
        set_seed(self.cfg.seed)
        print(f"Using device: {self.device}")

        # Setup
        train_dataset, dev_dataset, train_sampler, dev_sampler = self.setup_data()
        model, feature_extractor = self.setup_model()

        print(f"Training samples: {len(train_dataset)}")
        print(f"Dev samples: {len(dev_dataset)}")
        print(f"Feature dim: {self.cfg.features.fused_dim}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Optimizer and scheduler
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.cfg.train.scheduler_step,
            gamma=self.cfg.train.scheduler_gamma,
        )

        # Checkpointing
        os.makedirs(self.cfg.paths.checkpoint_dir, exist_ok=True)
        best_dev_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.cfg.train.num_epochs + 1):
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(
                model, feature_extractor, train_dataset, train_sampler, optimizer
            )

            # Evaluate
            dev_metrics = self.evaluate_epoch(
                model, feature_extractor, dev_dataset, dev_sampler
            )

            scheduler.step()
            elapsed = time.time() - t0

            # Print progress
            print(
                f"Epoch {epoch:03d}/{self.cfg.train.num_epochs} "
                f"[{elapsed:.1f}s] "
                f"Train Loss: {train_metrics['total_loss']:.4f} "
                f"Proto Acc: {train_metrics['proto_acc']:.4f} "
                f"| Dev Loss: {dev_metrics['total_loss']:.4f} "
                f"Dev Proto Acc: {dev_metrics['proto_acc']:.4f}"
            )

            # W&B logging
            if self.use_wandb:
                log_dict = {}
                for k, v in train_metrics.items():
                    log_dict[f"train/{k}"] = v
                for k, v in dev_metrics.items():
                    log_dict[f"dev/{k}"] = v
                log_dict["epoch"] = epoch
                log_dict["lr"] = optimizer.param_groups[0]["lr"]
                self.wandb.log(log_dict)

            # Checkpoint
            if dev_metrics["total_loss"] < best_dev_loss:
                best_dev_loss = dev_metrics["total_loss"]
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "dev_loss": best_dev_loss,
                    },
                    os.path.join(self.cfg.paths.checkpoint_dir, "best_model.pt"),
                )
                print(f"  -> Saved best model (dev loss: {best_dev_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.train.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        # Compute Fisher Information for EWC after training on known attacks
        print("Computing Fisher Information for EWC...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=0,  # safe for Fisher computation
        )
        model.compute_and_store_fisher(
            train_loader, feature_extractor, self.device,
            num_samples=self.cfg.model.ewc_sample_size,
        )

        # Save final model with EWC state
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "ewc_fisher": model.ewc.fisher_info,
                "ewc_params": model.ewc.saved_params,
            },
            os.path.join(self.cfg.paths.checkpoint_dir, "final_model_with_ewc.pt"),
        )
        print("Training complete. Final model saved.")

        if self.use_wandb:
            self.wandb.finish()

        return model


def train_pipeline(cfg: Config) -> DeepfakeDetector:
    """Entry point for training."""
    trainer = Trainer(cfg)
    return trainer.train()
