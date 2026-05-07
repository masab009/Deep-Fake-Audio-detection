"""
Stage 1: Data Preprocessing Module
- Resample to 16kHz
- Peak normalization
- Codec simulation (G.711, AMR-NB, Opus)
- VAD-based silence trimming
- Sliding window segmentation (3s windows, 1.5s overlap)
"""

import os
import struct
import io
import random
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torchaudio
import soundfile as sf
import webrtcvad
import pandas as pd
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from config import Config, AudioConfig


# ---------------------------------------------------------------------------
# Codec simulation helpers
# ---------------------------------------------------------------------------

def _mu_law_encode(samples: np.ndarray, mu: int = 255) -> np.ndarray:
    """μ-law companding (G.711 μ-law)."""
    samples = np.clip(samples, -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(samples)) / np.log1p(mu)
    return np.sign(samples) * magnitude


def _mu_law_decode(samples: np.ndarray, mu: int = 255) -> np.ndarray:
    """Inverse μ-law."""
    magnitude = (np.power(1 + mu, np.abs(samples)) - 1) / mu
    return np.sign(samples) * magnitude


def apply_g711(waveform: np.ndarray) -> np.ndarray:
    """Simulate G.711 μ-law codec (PSTN)."""
    # Quantize to 8-bit via mu-law companding
    encoded = _mu_law_encode(waveform)
    quantized = np.round(encoded * 127) / 127  # 8-bit quantization
    decoded = _mu_law_decode(quantized)
    return decoded


def apply_amr_nb(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulate AMR-NB codec by downsampling to 8kHz and back."""
    # AMR-NB operates at 8kHz — simulate by down/up sampling + quantization
    if sr != 16000:
        raise ValueError("Expected 16kHz input")
    # Downsample to 8kHz
    down = resample_poly(waveform, up=1, down=2)
    # Simulate lossy quantization
    down = np.round(down * 8192) / 8192
    # Upsample back to 16kHz
    up = resample_poly(down, up=2, down=1)
    # Match original length
    if len(up) > len(waveform):
        up = up[:len(waveform)]
    elif len(up) < len(waveform):
        up = np.pad(up, (0, len(waveform) - len(up)))
    return up


def apply_opus(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulate Opus codec via low-pass filtering + quantization noise."""
    # Opus is hard to simulate without an actual encoder; approximate with
    # band-limiting + additive quantization noise at typical Opus bitrates
    from scipy.signal import butter, filtfilt
    # Low-pass at 7.5kHz (Opus narrowband-like)
    nyq = sr / 2
    cutoff = 7500 / nyq
    if cutoff >= 1.0:
        cutoff = 0.99
    b, a = butter(5, cutoff, btype="low")
    filtered = filtfilt(b, a, waveform)
    # Add small quantization noise
    noise_level = 1e-4
    noise = np.random.randn(len(filtered)) * noise_level
    return filtered + noise


CODEC_FUNCTIONS = {
    "g711": apply_g711,
    "amr_nb": apply_amr_nb,
    "opus": apply_opus,
}


def apply_codec_simulation(
    waveform: np.ndarray,
    sr: int,
    codec_name: str,
) -> np.ndarray:
    """Apply a specific codec simulation to the waveform."""
    fn = CODEC_FUNCTIONS.get(codec_name)
    if fn is None:
        raise ValueError(f"Unknown codec: {codec_name}")
    return fn(waveform, sr) if codec_name != "g711" else fn(waveform)


# ---------------------------------------------------------------------------
# Peak normalization
# ---------------------------------------------------------------------------

def peak_normalize(waveform: np.ndarray) -> np.ndarray:
    """Normalize waveform to [-1, 1] using peak normalization."""
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    return waveform


# ---------------------------------------------------------------------------
# VAD-based silence trimming
# ---------------------------------------------------------------------------

def vad_trim(
    waveform: np.ndarray,
    sr: int = 16000,
    aggressiveness: int = 2,
    frame_duration_ms: int = 30,
) -> np.ndarray:
    """Remove non-speech segments using WebRTC VAD."""
    vad = webrtcvad.Vad(aggressiveness)
    # Convert to 16-bit PCM
    pcm = (waveform * 32767).astype(np.int16)
    frame_size = int(sr * frame_duration_ms / 1000)
    frames = []

    for start in range(0, len(pcm) - frame_size + 1, frame_size):
        frame = pcm[start : start + frame_size]
        frame_bytes = frame.tobytes()
        try:
            if vad.is_speech(frame_bytes, sr):
                frames.append(waveform[start : start + frame_size])
        except Exception:
            # If VAD fails on a frame, keep it
            frames.append(waveform[start : start + frame_size])

    if len(frames) == 0:
        return waveform  # fallback: return original if all trimmed

    return np.concatenate(frames)


# ---------------------------------------------------------------------------
# Sliding window segmentation
# ---------------------------------------------------------------------------

def segment_audio(
    waveform: np.ndarray,
    sr: int = 16000,
    segment_duration: float = 3.0,
    overlap: float = 1.5,
) -> List[np.ndarray]:
    """Extract fixed-length segments using sliding window."""
    segment_samples = int(segment_duration * sr)
    hop_samples = int((segment_duration - overlap) * sr)
    segments = []

    if len(waveform) < segment_samples:
        # Pad short audio
        padded = np.zeros(segment_samples)
        padded[: len(waveform)] = waveform
        segments.append(padded)
    else:
        for start in range(0, len(waveform) - segment_samples + 1, hop_samples):
            segments.append(waveform[start : start + segment_samples])

    return segments


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_audio(
    filepath: str,
    cfg: AudioConfig,
    apply_codecs: bool = False,
    codec_name: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Full preprocessing pipeline for a single audio file.
    Returns list of preprocessed segments.
    """
    # Load audio
    waveform, sr = sf.read(filepath, dtype="float32")

    # Ensure mono
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    # Resample to 16kHz if needed
    if sr != cfg.sample_rate:
        waveform = resample_poly(
            waveform,
            up=cfg.sample_rate,
            down=sr,
        ).astype(np.float32)

    # Peak normalization
    waveform = peak_normalize(waveform)

    # Codec simulation (stochastic during training)
    if apply_codecs:
        if codec_name is None:
            codec_name = random.choice(cfg.codecs)
        waveform = apply_codec_simulation(waveform, cfg.sample_rate, codec_name)
        waveform = waveform.astype(np.float32)

    # VAD-based silence trimming
    waveform = vad_trim(
        waveform,
        sr=cfg.sample_rate,
        aggressiveness=cfg.vad_aggressiveness,
        frame_duration_ms=cfg.vad_frame_ms,
    )

    # Sliding window segmentation
    segments = segment_audio(
        waveform,
        sr=cfg.sample_rate,
        segment_duration=cfg.segment_duration,
        overlap=cfg.segment_overlap,
    )

    return segments


# ---------------------------------------------------------------------------
# ASVspoof 2019 LA Protocol Parser
# ---------------------------------------------------------------------------

def parse_protocol(protocol_path: str) -> pd.DataFrame:
    """
    Parse ASVspoof 2019 LA protocol file.
    Format: SPEAKER_ID AUDIO_FILE_ID - ATTACK_TYPE LABEL
    """
    records = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                records.append({
                    "speaker_id": parts[0],
                    "utterance_id": parts[1],
                    "attack_type": parts[3],
                    "label": parts[4],  # bonafide or spoof
                })
            elif len(parts) >= 4:
                # Some protocol formats differ
                records.append({
                    "speaker_id": parts[0],
                    "utterance_id": parts[1],
                    "attack_type": parts[2],
                    "label": parts[3],
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ASVspoofDataset(Dataset):
    """
    ASVspoof 2019 LA dataset with full preprocessing pipeline.
    Each item returns preprocessed audio segments and metadata.
    """

    def __init__(
        self,
        protocol_path: str,
        flac_dir: str,
        cfg: AudioConfig,
        attack_filter: Optional[List[str]] = None,
        apply_codecs: bool = False,
        max_segments_per_utterance: int = 1,
    ):
        self.flac_dir = flac_dir
        self.cfg = cfg
        self.apply_codecs = apply_codecs
        self.max_segments = max_segments_per_utterance

        # Parse protocol
        self.metadata = parse_protocol(protocol_path)

        # Filter by attack type if specified
        if attack_filter is not None:
            mask = (
                self.metadata["attack_type"].isin(attack_filter)
                | (self.metadata["label"] == "bonafide")
            )
            self.metadata = self.metadata[mask].reset_index(drop=True)

        # Build label maps
        self.label_map = {"bonafide": 0, "spoof": 1}
        all_attacks = sorted(self.metadata["attack_type"].unique().tolist())
        self.attack_map = {a: i for i, a in enumerate(all_attacks)}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx) -> Dict:
        row = self.metadata.iloc[idx]
        filepath = os.path.join(self.flac_dir, f"{row['utterance_id']}.flac")

        # Preprocess
        segments = preprocess_audio(
            filepath, self.cfg, apply_codecs=self.apply_codecs
        )

        # Take up to max_segments
        if len(segments) > self.max_segments:
            segments = segments[: self.max_segments]

        # Convert to tensor
        segment_tensors = [torch.tensor(s, dtype=torch.float32) for s in segments]

        # Use first segment for simplicity in episodic training
        audio = segment_tensors[0]

        label = self.label_map.get(row["label"], 1)
        attack_type = self.attack_map.get(row["attack_type"], -1)

        return {
            "audio": audio,                          # (segment_samples,)
            "label": torch.tensor(label, dtype=torch.long),
            "attack_type": torch.tensor(attack_type, dtype=torch.long),
            "utterance_id": row["utterance_id"],
            "attack_name": row["attack_type"],
        }


class EpisodicSampler:
    """
    Episodic sampler for few-shot training.
    Each episode samples n_way classes with k_shot support + q_query queries.
    """

    def __init__(
        self,
        dataset: ASVspoofDataset,
        n_way: int = 2,
        k_shot: int = 5,
        q_query: int = 15,
        num_episodes: int = 1000,
        held_out_attacks: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes

        # Build index: label -> list of dataset indices
        self.class_indices = {}  # label -> [indices]
        self.attack_indices = {}  # attack_name -> [indices]

        for i in range(len(dataset)):
            row = dataset.metadata.iloc[i]
            label = dataset.label_map.get(row["label"], 1)
            attack = row["attack_type"]

            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(i)

            if attack not in self.attack_indices:
                self.attack_indices[attack] = []
            self.attack_indices[attack].append(i)

        self.held_out_attacks = held_out_attacks or []

    def sample_episode(self) -> Dict:
        """
        Sample one episodic task.
        Returns support set and query set indices.
        """
        support_indices = []
        query_indices = []

        for cls in range(self.n_way):
            indices = self.class_indices.get(cls, [])
            if len(indices) < self.k_shot + self.q_query:
                # Sample with replacement if not enough
                selected = random.choices(indices, k=self.k_shot + self.q_query)
            else:
                selected = random.sample(indices, self.k_shot + self.q_query)

            support_indices.extend(selected[: self.k_shot])
            query_indices.extend(selected[self.k_shot :])

        return {
            "support_indices": support_indices,
            "query_indices": query_indices,
        }

    def sample_fewshot_episode(self, novel_attack: str, k: int = 5) -> Dict:
        """
        Sample a few-shot episode for a specific novel attack type.
        Support set: k examples from novel attack + k bonafide.
        Query set: remaining examples.
        """
        novel_indices = self.attack_indices.get(novel_attack, [])
        bonafide_indices = self.class_indices.get(0, [])

        if len(novel_indices) < k:
            support_novel = random.choices(novel_indices, k=k)
        else:
            support_novel = random.sample(novel_indices, k)

        if len(bonafide_indices) < k:
            support_bonafide = random.choices(bonafide_indices, k=k)
        else:
            support_bonafide = random.sample(bonafide_indices, k)

        # Query: sample from remaining
        remaining_novel = [i for i in novel_indices if i not in support_novel]
        remaining_bonafide = [i for i in bonafide_indices if i not in support_bonafide]

        q = min(self.q_query, len(remaining_novel), len(remaining_bonafide))
        q = max(q, 1)

        query_novel = random.sample(remaining_novel, min(q, len(remaining_novel))) if remaining_novel else support_novel[:1]
        query_bonafide = random.sample(remaining_bonafide, min(q, len(remaining_bonafide))) if remaining_bonafide else support_bonafide[:1]

        return {
            "support_indices": support_bonafide + support_novel,
            "query_indices": query_bonafide + query_novel,
        }

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        for _ in range(self.num_episodes):
            yield self.sample_episode()
