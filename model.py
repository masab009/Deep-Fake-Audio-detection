"""
Stage 3: Model Architecture Module
- Prototypical Network backbone with embedding space
- Elastic Weight Consolidation (EWC) for continual learning
- Genuine/fake classification head
- Auxiliary attack-type classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from config import ModelConfig, FeatureConfig


# ---------------------------------------------------------------------------
# Embedding Network (backbone)
# ---------------------------------------------------------------------------

class EmbeddingNetwork(nn.Module):
    """
    MLP-based embedding network that maps fused features to a
    lower-dimensional embedding space for prototypical classification.
    """

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) fused features
        Returns:
            embeddings: (batch, embedding_dim)
        """
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Classification Heads
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Binary genuine/fake classification head."""

    def __init__(self, embedding_dim: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 2),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns logits for [genuine, fake]."""
        return self.classifier(embeddings)


class AttackTypeClassifier(nn.Module):
    """Auxiliary attack-type classifier to encourage attack-discriminative embeddings."""

    def __init__(self, embedding_dim: int, num_attack_types: int, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_attack_types),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns logits over attack types."""
        return self.classifier(embeddings)


# ---------------------------------------------------------------------------
# Prototypical Network
# ---------------------------------------------------------------------------

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network with classification heads.

    During episodic training:
    - Computes class prototypes from support set embeddings
    - Classifies query samples based on distance to prototypes

    Also includes:
    - Binary classification head (genuine/fake)
    - Auxiliary attack-type classifier
    """

    def __init__(
        self,
        input_dim: int,
        model_cfg: ModelConfig,
    ):
        super().__init__()
        self.model_cfg = model_cfg

        # Embedding backbone
        self.embedding_net = EmbeddingNetwork(
            input_dim=input_dim,
            hidden_dim=model_cfg.hidden_dim,
            embedding_dim=model_cfg.embedding_dim,
            dropout=model_cfg.dropout,
        )

        # Classification heads
        self.classification_head = ClassificationHead(
            embedding_dim=model_cfg.embedding_dim,
            dropout=model_cfg.dropout,
        )
        self.attack_classifier = AttackTypeClassifier(
            embedding_dim=model_cfg.embedding_dim,
            num_attack_types=model_cfg.num_known_attacks + 1,  # +1 for bonafide
            dropout=model_cfg.dropout,
        )

        # Stored prototypes for inference
        self.register_buffer(
            "class_prototypes",
            torch.zeros(2, model_cfg.embedding_dim),  # genuine & fake
        )

    def embed(self, features: torch.Tensor) -> torch.Tensor:
        """Compute embeddings from fused features."""
        return self.embedding_net(features)

    def compute_prototypes(
        self, support_features: torch.Tensor, support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class prototypes (centroids) from support set.

        Args:
            support_features: (n_support, input_dim)
            support_labels: (n_support,)

        Returns:
            prototypes: (n_classes, embedding_dim)
        """
        embeddings = self.embed(support_features)
        classes = torch.unique(support_labels)
        prototypes = []
        for c in classes:
            mask = support_labels == c
            prototype = embeddings[mask].mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)  # (n_classes, embedding_dim)

    def prototypical_loss(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prototypical loss and accuracy for an episode.

        Returns:        Returns:

            loss: scalar
            accuracy: scalar
        """
        prototypes = self.compute_prototypes(support_features, support_labels)
        query_embeddings = self.embed(query_features)

        # Euclidean distance from each query to each prototype
        # query_embeddings: (Q, D), prototypes: (C, D)
        dists = torch.cdist(query_embeddings, prototypes)  # (Q, C)

        # Prototypical network uses negative distance as logits
        logits = -dists  # (Q, C)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, query_labels)

        # Accuracy
        preds = logits.argmax(dim=-1)
        accuracy = (preds == query_labels).float().mean()

        return loss, accuracy

    def forward(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for standard classification.

        Args:
            features: (batch, input_dim) fused features

        Returns:
            dict with 'embeddings', 'classification_logits', 'attack_logits'
        """
        embeddings = self.embed(features)
        cls_logits = self.classification_head(embeddings)
        atk_logits = self.attack_classifier(embeddings)

        return {
            "embeddings": embeddings,
            "classification_logits": cls_logits,
            "attack_logits": atk_logits,
        }

    def predict_with_prototypes(
        self, features: torch.Tensor, prototypes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict using stored or provided prototypes.

        Returns:
            deepfake_scores: (batch,) probability of being fake
        """
        if prototypes is None:
            prototypes = self.class_prototypes

        embeddings = self.embed(features)
        dists = torch.cdist(embeddings, prototypes)  # (B, 2)
        probs = F.softmax(-dists, dim=-1)  # (B, 2)
        # Return P(fake) — assuming class 1 is fake
        return probs[:, 1]

    def update_prototypes(self, prototypes: torch.Tensor):
        """Update stored class prototypes."""
        self.class_prototypes.copy_(prototypes)


# ---------------------------------------------------------------------------
# Elastic Weight Consolidation (EWC)
# ---------------------------------------------------------------------------

class EWC:
    """
    Elastic Weight Consolidation for preventing catastrophic forgetting.

    Computes Fisher Information Matrix diagonal and penalizes changes
    to parameters important for previously learned tasks.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 5000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda

        # Store snapshots of parameters and Fisher information
        self.saved_params: Dict[str, torch.Tensor] = {}
        self.fisher_info: Dict[str, torch.Tensor] = {}

    def compute_fisher(
        self,
        data_loader,
        feature_extractor,
        device: str = "cpu",
        num_samples: int = 200,
    ):
        """
        Compute diagonal Fisher Information Matrix using empirical Fisher.

        Args:
            data_loader: DataLoader providing audio samples
            feature_extractor: DualStreamFeatureExtractor to compute features
            device: computation device
            num_samples: number of samples for Fisher estimation
        """
        self.model.eval()
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        count = 0
        for batch in data_loader:
            if count >= num_samples:
                break

            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                features = feature_extractor(audio)

            self.model.zero_grad()
            output = self.model(features)
            log_probs = F.log_softmax(output["classification_logits"], dim=-1)

            # Use labels for empirical Fisher
            loss = F.nll_loss(log_probs, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

            count += audio.shape[0]

        # Average
        for n in fisher:
            fisher[n] /= max(count, 1)

        # Save current parameters and Fisher
        self.fisher_info = fisher
        self.saved_params = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty term.

        Returns:
            penalty: scalar tensor — sum of Fisher-weighted parameter changes
        """
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        if not self.fisher_info:
            return loss

        for n, p in self.model.named_parameters():
            if n in self.fisher_info and p.requires_grad:
                loss += (
                    self.fisher_info[n] * (p - self.saved_params[n]).pow(2)
                ).sum()

        return self.ewc_lambda * loss


# ---------------------------------------------------------------------------
# Full Model Wrapper
# ---------------------------------------------------------------------------

class DeepfakeDetector(nn.Module):
    """
    Complete model combining:
    - Prototypical Network backbone
    - EWC regularization
    - Classification + attack-type heads
    """

    def __init__(self, feature_dim: int, model_cfg: ModelConfig):
        super().__init__()
        self.proto_net = PrototypicalNetwork(
            input_dim=feature_dim,
            model_cfg=model_cfg,
        )
        self.ewc = EWC(self.proto_net, ewc_lambda=model_cfg.ewc_lambda)
        self.model_cfg = model_cfg

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.proto_net(features)

    def episodic_step(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        query_labels: torch.Tensor,
        support_attack_types: Optional[torch.Tensor] = None,
        query_attack_types: Optional[torch.Tensor] = None,
        cls_weight: float = 1.0,
        atk_weight: float = 0.5,
        proto_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for one episodic training step.

        Returns dict with:
            'total_loss', 'proto_loss', 'cls_loss', 'atk_loss',
            'proto_acc', 'cls_acc', 'ewc_penalty'
        """
        # 1. Prototypical loss
        proto_loss, proto_acc = self.proto_net.prototypical_loss(
            support_features, support_labels,
            query_features, query_labels,
        )

        # 2. Classification loss on query set
        query_output = self.proto_net(query_features)
        cls_loss = F.cross_entropy(
            query_output["classification_logits"], query_labels
        )
        cls_preds = query_output["classification_logits"].argmax(dim=-1)
        cls_acc = (cls_preds == query_labels).float().mean()

        # 3. Attack-type loss (only on spoof samples)
        atk_loss = torch.tensor(0.0, device=query_features.device)
        if query_attack_types is not None:
            spoof_mask = query_labels == 1
            if spoof_mask.any():
                spoof_atk_logits = query_output["attack_logits"][spoof_mask]
                spoof_atk_labels = query_attack_types[spoof_mask]
                # Ensure labels are valid
                valid_mask = spoof_atk_labels >= 0
                if valid_mask.any():
                    atk_loss = F.cross_entropy(
                        spoof_atk_logits[valid_mask],
                        spoof_atk_labels[valid_mask],
                    )

        # 4. EWC penalty
        ewc_penalty = self.ewc.penalty()

        # Total loss
        total_loss = (
            proto_weight * proto_loss
            + cls_weight * cls_loss
            + atk_weight * atk_loss
            + ewc_penalty
        )

        return {
            "total_loss": total_loss,
            "proto_loss": proto_loss,
            "cls_loss": cls_loss,
            "atk_loss": atk_loss,
            "proto_acc": proto_acc,
            "cls_acc": cls_acc,
            "ewc_penalty": ewc_penalty,
        }

    def compute_and_store_fisher(self, data_loader, feature_extractor, device, num_samples=200):
        """Compute Fisher information for EWC after training on a task."""
        self.ewc.compute_fisher(data_loader, feature_extractor, device, num_samples)

    def update_prototypes(self, support_features: torch.Tensor, support_labels: torch.Tensor):
        """Update stored prototypes from support set."""
        prototypes = self.proto_net.compute_prototypes(support_features, support_labels)
        self.proto_net.update_prototypes(prototypes)

    def predict(self, features: torch.Tensor, prototypes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get deepfake probability scores."""
        return self.proto_net.predict_with_prototypes(features, prototypes)
