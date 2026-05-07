"""
Evaluation Metrics Module
- Equal Error Rate (EER)
- Accuracy, Precision, Recall, F1
- Backward Transfer (BWT) for continual learning
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).

    The EER is the point where FAR == FRR.

    Args:
        y_true: ground truth binary labels (0=genuine, 1=spoof)
        y_scores: predicted scores (higher = more likely spoof)

    Returns:
        eer: Equal Error Rate
        threshold: operating threshold at EER
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # Find the point where FPR ≈ FNR
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

    return float(eer), float(eer_threshold)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard classification metrics.

    Args:
        y_true: ground truth binary labels
        y_pred: predicted binary labels

    Returns:
        dict with accuracy, precision, recall, f1
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def compute_backward_transfer(
    performance_matrix: np.ndarray,
) -> float:
    """
    Compute Backward Transfer (BWT) for continual learning.

    BWT = (1/T-1) * Σ_{i=1}^{T-1} (R_{T,i} - R_{i,i})

    Where R_{j,i} is the performance on task i after training on task j.

    Args:
        performance_matrix: (T, T) matrix where entry [j, i] is the
            performance on task i after learning task j.

    Returns:
        bwt: Backward Transfer score. Negative = catastrophic forgetting.
    """
    T = performance_matrix.shape[0]
    if T < 2:
        return 0.0

    bwt = 0.0
    for i in range(T - 1):
        bwt += performance_matrix[T - 1, i] - performance_matrix[i, i]

    return float(bwt / (T - 1))


def compute_all_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: ground truth binary labels
        y_scores: predicted deepfake scores
        threshold: classification threshold (if None, uses EER threshold)

    Returns:
        dict with eer, eer_threshold, accuracy, precision, recall, f1
    """
    eer, eer_threshold = compute_eer(y_true, y_scores)

    if threshold is None:
        threshold = eer_threshold

    y_pred = (y_scores >= threshold).astype(int)
    cls_metrics = compute_classification_metrics(y_true, y_pred)

    return {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "threshold_used": threshold,
        **cls_metrics,
    }


def _chunk_indices(indices: List[int], batch_size: int):
    """Yield index chunks for batched preprocessing and feature extraction."""
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


class _IndexedDataset(Dataset):
    """Wrap a base dataset so DataLoader can parallelize indexed fetches."""

    def __init__(self, base_dataset, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        dataset_idx = self.indices[idx]
        item = self.base_dataset[dataset_idx]
        return dataset_idx, item["audio"], item["label"]


def _default_eval_num_workers() -> int:
    """Choose a conservative worker count for eval preprocessing."""
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


def _default_feature_batch_size(device: str) -> int:
    """Use larger batches on GPU to amortize SSL inference overhead."""
    return 96 if device.startswith("cuda") else 16


@torch.inference_mode()
def _precompute_feature_cache(
    feature_extractor,
    eval_dataset,
    planned_episodes: Dict[str, Dict[str, List[Dict[str, List[int]]]]],
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Precompute fused features once for every eval sample touched by the
    planned episodes. This avoids repeated audio preprocessing and SSL
    feature extraction for the same utterance across attacks, shots, and
    episodes.
    """
    unique_indices = set()
    for attack_plan in planned_episodes.values():
        for episodes in attack_plan.values():
            for episode in episodes:
                unique_indices.update(episode["support_indices"])
                unique_indices.update(episode["query_indices"])

    if not unique_indices:
        return {}, {}

    feature_cache: Dict[int, torch.Tensor] = {}
    label_cache: Dict[int, torch.Tensor] = {}
    subset = _IndexedDataset(eval_dataset, sorted(unique_indices))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
    )
    total_items = len(subset)
    processed = 0

    print(
        f"Precomputing features for {total_items} unique evaluation samples "
        f"(batch_size={batch_size}, workers={num_workers})..."
    )

    for batch_num, (batch_indices, batch_audio, batch_labels) in enumerate(loader, start=1):
        batch_audio = batch_audio.to(device, non_blocking=True)
        batch_features = feature_extractor(batch_audio).cpu()

        for idx, label, feature in zip(batch_indices.tolist(), batch_labels, batch_features):
            feature_cache[idx] = feature
            label_cache[idx] = label.clone()

        processed += len(batch_indices)
        if batch_num == 1 or batch_num % 25 == 0 or processed == total_items:
            print(f"  Cached features: {processed}/{total_items}")

    return feature_cache, label_cache


def evaluate_fewshot_adaptation(
    model,
    feature_extractor,
    eval_dataset,
    attack_types: List[str],
    k_shots: List[int],
    device: str = "cpu",
    num_episodes: int = 10,
    q_query: int = 50,
    feature_batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Dict:
    """
    Evaluate few-shot adaptation performance across different attack types
    and shot counts.

    Returns:
        results dict with per-attack and per-k metrics
    """
    from data_preprocessing import EpisodicSampler

    model.eval()
    feature_extractor.eval()

    if feature_batch_size is None:
        feature_batch_size = _default_feature_batch_size(device)
    if num_workers is None:
        num_workers = _default_eval_num_workers()

    results = {}
    planned_episodes: Dict[str, Dict[str, List[Dict[str, List[int]]]]] = {}

    print(
        f"Planning evaluation episodes for {len(attack_types)} attacks, "
        f"shots={k_shots}, num_episodes={num_episodes}, q_query={q_query}"
    )

    for attack in attack_types:
        planned_episodes[attack] = {}
        for k in k_shots:
            sampler = EpisodicSampler(
                dataset=eval_dataset,
                n_way=2,
                k_shot=k,
                q_query=q_query,
                num_episodes=num_episodes,
            )

            episodes = []
            for _ in range(num_episodes):
                try:
                    episodes.append(sampler.sample_fewshot_episode(attack, k=k))
                except Exception:
                    continue

            planned_episodes[attack][f"{k}-shot"] = episodes

    total_episodes = sum(
        len(episodes)
        for attack_plan in planned_episodes.values()
        for episodes in attack_plan.values()
    )
    print(f"Planned {total_episodes} evaluation episodes.")

    feature_cache, label_cache = _precompute_feature_cache(
        feature_extractor,
        eval_dataset,
        planned_episodes,
        device=device,
        batch_size=feature_batch_size,
        num_workers=num_workers,
    )

    for attack in attack_types:
        results[attack] = {}
        for k in k_shots:
            episodes = planned_episodes.get(attack, {}).get(f"{k}-shot", [])
            all_scores = []
            all_labels = []

            for episode in episodes:
                support_indices = episode["support_indices"]
                query_indices = episode["query_indices"]
                if not support_indices or not query_indices:
                    continue

                s_features = torch.stack(
                    [feature_cache[idx] for idx in support_indices]
                ).to(device)
                q_features = torch.stack(
                    [feature_cache[idx] for idx in query_indices]
                ).to(device)
                s_labels = torch.stack(
                    [label_cache[idx] for idx in support_indices]
                ).to(device)
                q_labels = torch.stack(
                    [label_cache[idx] for idx in query_indices]
                )

                # Compute prototypes from support
                prototypes = model.proto_net.compute_prototypes(s_features, s_labels)

                # Score query samples
                scores = model.predict(q_features, prototypes)

                all_scores.extend(scores.cpu().detach().numpy())
                all_labels.extend(q_labels.numpy().tolist())

            if all_scores:
                metrics = compute_all_metrics(
                    np.array(all_labels), np.array(all_scores)
                )
                results[attack][f"{k}-shot"] = metrics

    return results


def evaluate_continual_learning(
    performance_records: List[Dict],
) -> Dict[str, float]:
    """
    Evaluate continual learning performance using Backward Transfer.

    Args:
        performance_records: list of dicts, each with
            'task_id', 'evaluated_on_task', 'eer'

    Returns:
        dict with bwt and per-task metrics
    """
    if not performance_records:
        return {"bwt": 0.0}

    # Build performance matrix
    tasks = sorted(set(r["task_id"] for r in performance_records))
    task_idx = {t: i for i, t in enumerate(tasks)}
    T = len(tasks)

    perf_matrix = np.zeros((T, T))
    for record in performance_records:
        i = task_idx[record["task_id"]]
        j = task_idx[record["evaluated_on_task"]]
        # Use 1 - EER as performance (higher is better)
        perf_matrix[i, j] = 1 - record["eer"]

    bwt = compute_backward_transfer(perf_matrix)

    return {
        "bwt": bwt,
        "performance_matrix": perf_matrix.tolist(),
        "tasks": tasks,
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    if prefix:
        print(f"  {prefix}")
        print(f"{'='*50}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        else:
            print(f"  {key:20s}: {value}")
    print(f"{'='*50}\n")
