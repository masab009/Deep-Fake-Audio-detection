"""
Configuration module for Few-Shot Continual Learning Audio Deepfake Detection.
Central place for all hyperparameters, paths, and settings.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathConfig:
    """Paths for data and outputs."""
    dataset_root: str = os.environ.get("ASVSPOOF_ROOT", "./data/LA")
    train_protocol: str = "ASVspoof2019.LA.cm.train.trn.txt"
    dev_protocol: str = "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_protocol: str = "ASVspoof2019.LA.cm.eval.trl.txt"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./outputs/checkpoints"
    log_dir: str = "./outputs/logs"

    @property
    def train_flac_dir(self):
        return os.path.join(self.dataset_root, "ASVspoof2019_LA_train", "flac")

    @property
    def dev_flac_dir(self):
        return os.path.join(self.dataset_root, "ASVspoof2019_LA_dev", "flac")

    @property
    def eval_flac_dir(self):
        return os.path.join(self.dataset_root, "ASVspoof2019_LA_eval", "flac")

    @property
    def train_protocol_path(self):
        return os.path.join(
            self.dataset_root,
            "ASVspoof2019_LA_cm_protocols",
            self.train_protocol,
        )

    @property
    def dev_protocol_path(self):
        return os.path.join(
            self.dataset_root,
            "ASVspoof2019_LA_cm_protocols",
            self.dev_protocol,
        )

    @property
    def eval_protocol_path(self):
        return os.path.join(
            self.dataset_root,
            "ASVspoof2019_LA_cm_protocols",
            self.eval_protocol,
        )


@dataclass
class AudioConfig:
    """Audio preprocessing parameters."""
    sample_rate: int = 16000
    segment_duration: float = 3.0  # seconds
    segment_overlap: float = 1.5   # seconds
    vad_aggressiveness: int = 2    # webrtcvad aggressiveness 0-3
    vad_frame_ms: int = 30         # VAD frame duration in ms

    # Codec simulation
    codecs: List[str] = field(default_factory=lambda: ["g711", "amr_nb", "opus"])
    codec_probability: float = 0.5  # probability of applying codec during training

    # Augmentation
    noise_snr_range: tuple = (5, 20)  # SNR in dB
    augment_probability: float = 0.3


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    # SSL models
    wavlm_model: str = "microsoft/wavlm-base-plus"
    hubert_model: str = "facebook/hubert-base-ls960"
    ssl_embedding_dim: int = 768  # base models output 768-dim

    # Cepstral features
    n_mfcc: int = 40
    n_lfcc: int = 40
    n_cqcc: int = 40
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 80

    # Fused dimension = wavlm(768) + hubert(768) + mfcc(40) + lfcc(40) + cqcc(40)
    @property
    def fused_dim(self):
        return self.ssl_embedding_dim * 2 + self.n_mfcc + self.n_lfcc + self.n_cqcc


@dataclass
class ModelConfig:
    """Model architecture parameters."""
    embedding_dim: int = 256       # Prototypical network embedding dimension
    hidden_dim: int = 512          # Hidden layer size
    num_known_attacks: int = 6     # A01-A06 in ASVspoof 2019 training
    dropout: float = 0.3

    # EWC parameters
    ewc_lambda: float = 5000.0    # EWC regularization strength
    ewc_sample_size: int = 200    # Number of samples for Fisher information


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Episodic training
    n_way: int = 2                 # genuine vs fake (binary)
    k_shot: int = 5               # support examples per class
    q_query: int = 15              # query examples per class
    num_episodes_train: int = 1000 # episodes per epoch
    num_episodes_eval: int = 200   # episodes for evaluation
    num_epochs: int = 50
    patience: int = 10             # early stopping patience

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler_step: int = 20
    scheduler_gamma: float = 0.5

    # Loss weights
    classification_weight: float = 1.0
    attack_type_weight: float = 0.5
    prototypical_weight: float = 1.0

    # Few-shot held-out attacks for evaluation
    held_out_attacks: List[str] = field(
        default_factory=lambda: ["A07", "A08", "A09", "A10", "A11", "A12",
                                  "A13", "A14", "A15", "A16", "A17", "A18", "A19"]
    )
    known_attacks: List[str] = field(
        default_factory=lambda: ["A01", "A02", "A03", "A04", "A05", "A06"]
    )
    few_shot_k: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Batch size for non-episodic operations
    batch_size: int = 32
    num_workers: int = 4

    # Device
    device: str = "cuda"  # will fall back to cpu if unavailable

    # W&B
    wandb_project: str = "audio-deepfake-fscl"
    wandb_entity: Optional[str] = None
    use_wandb: bool = True


@dataclass
class InferenceConfig:
    """Inference and adaptation parameters."""
    deepfake_threshold: float = 0.5
    adaptation_support_size: int = 10  # examples for prototype update
    centroid_momentum: float = 0.9     # momentum for centroid update


@dataclass
class Config:
    """Master config aggregating all sub-configs."""
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    seed: int = 42


def get_config(**overrides) -> Config:
    """Create config with optional overrides."""
    cfg = Config()
    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return cfg
