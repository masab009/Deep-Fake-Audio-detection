"""
Stage 2: Feature Engineering Module
Dual-stream feature extraction:
  Stream 1: SSL embeddings (WavLM + HuBERT via Hugging Face Transformers)
  Stream 2: Cepstral features (MFCC, LFCC, CQCC via torchaudio/librosa)
Outputs concatenated fused representation.
"""

import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from transformers import WavLMModel, HubertModel, AutoProcessor
from typing import Optional

from config import FeatureConfig


# ---------------------------------------------------------------------------
# SSL Feature Extractor (Stream 1)
# ---------------------------------------------------------------------------

class SSLFeatureExtractor(nn.Module):
    """
    Extracts embeddings from pretrained WavLM and HuBERT models.
    Returns concatenated SSL features: [wavlm_emb; hubert_emb]
    """

    def __init__(self, cfg: FeatureConfig, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.use_cuda_amp = device.startswith("cuda") and torch.cuda.is_available()

        if self.use_cuda_amp:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Load pretrained models (frozen)
        self.wavlm = WavLMModel.from_pretrained(cfg.wavlm_model)
        self.hubert = HubertModel.from_pretrained(cfg.hubert_model)

        # Freeze SSL models — we only use them as feature extractors
        for param in self.wavlm.parameters():
            param.requires_grad = False
        for param in self.hubert.parameters():
            param.requires_grad = False

        self.wavlm.eval()
        self.hubert.eval()

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (batch, num_samples) raw audio at 16kHz

        Returns:
            ssl_features: (batch, ssl_dim * 2) where ssl_dim=768 for base models
        """
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.use_cuda_amp,
        ):
            # WavLM embeddings: mean pool over time
            wavlm_out = self.wavlm(waveforms).last_hidden_state  # (B, T, 768)
            wavlm_emb = wavlm_out.mean(dim=1)  # (B, 768)

            # HuBERT embeddings: mean pool over time
            hubert_out = self.hubert(waveforms).last_hidden_state  # (B, T, 768)
            hubert_emb = hubert_out.mean(dim=1)  # (B, 768)

        # Concatenate
        ssl_features = torch.cat([wavlm_emb, hubert_emb], dim=-1).float()  # (B, 1536)
        return ssl_features


# ---------------------------------------------------------------------------
# Cepstral Feature Extractor (Stream 2)
# ---------------------------------------------------------------------------

def compute_mfcc(
    waveform: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
) -> np.ndarray:
    """Compute MFCCs using librosa."""
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    return mfcc.mean(axis=1)  # (n_mfcc,) — average over time


def compute_lfcc(
    waveform: np.ndarray,
    sr: int = 16000,
    n_lfcc: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
) -> np.ndarray:
    # STFT — unchanged
    stft = np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_length))

    n_filters = n_lfcc * 2
    freq_bins = np.linspace(0, sr / 2, stft.shape[0])      # (257,)
    center_freqs = np.linspace(0, sr / 2, n_filters + 2)   # (82,)

    # --- Vectorized filterbank ---
    low    = center_freqs[:-2, None]   # (80, 1)  broadcast over freq axis
    center = center_freqs[1:-1, None]  # (80, 1)
    high   = center_freqs[2:,  None]   # (80, 1)
    f      = freq_bins[None, :]        # (1, 257) broadcast over filter axis

    rising  = (f - low)    / (center - low  + 1e-8)   # (80, 257)
    falling = (high - f)   / (high   - center + 1e-8) # (80, 257)

    fbank = np.where((f >= low)    & (f <= center), rising,
            np.where((f > center) & (f <= high),   falling, 0.0))
    # fbank shape: (80, 257) — computed in one shot, no loops

    # Rest unchanged
    filter_energies = np.dot(fbank, stft)
    log_energies = np.log(filter_energies + 1e-8)

    from scipy.fft import dct
    lfcc = dct(log_energies, type=2, axis=0, norm="ortho")[:n_lfcc]

    return lfcc.mean(axis=1)  # (n_lfcc,)


def compute_cqcc(
    waveform: np.ndarray,
    sr: int = 16000,
    n_cqcc: int = 40,
    hop_length: int = 160,
) -> np.ndarray:
    """
    Compute Constant-Q Cepstral Coefficients (CQCCs).
    Uses constant-Q transform instead of FFT.
    """
    # Constant-Q Transform
    # Limit bins so max freq stays well below Nyquist (sr/2 = 8000 Hz for 16kHz)
    cqt = np.abs(librosa.cqt(
        y=waveform,
        sr=sr,
        hop_length=hop_length,
        n_bins=72,
        bins_per_octave=12,
        fmin=librosa.note_to_hz("C2"),
    ))

    # Log power
    log_cqt = np.log(cqt + 1e-8)

    # DCT to get cepstral coefficients
    from scipy.fft import dct
    cqcc = dct(log_cqt, type=2, axis=0, norm="ortho")[:n_cqcc]

    return cqcc.mean(axis=1)  # (n_cqcc,)


class CepstralFeatureExtractor:
    """Extracts and concatenates MFCC, LFCC, and CQCC features."""

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg

    def extract(self, waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Args:
            waveform: (num_samples,) raw audio

        Returns:
            features: (n_mfcc + n_lfcc + n_cqcc,) concatenated cepstral features
        """
        mfcc = compute_mfcc(
            waveform, sr,
            n_mfcc=self.cfg.n_mfcc,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels,
        )
        lfcc = compute_lfcc(
            waveform, sr,
            n_lfcc=self.cfg.n_lfcc,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
        )
        cqcc = compute_cqcc(
            waveform, sr,
            n_cqcc=self.cfg.n_cqcc,
            hop_length=self.cfg.hop_length,
        )

        return np.concatenate([mfcc, lfcc, cqcc])

    def extract_batch(self, waveforms: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract cepstral features for a batch of waveforms."""
        return np.stack([self.extract(w, sr) for w in waveforms])


# ---------------------------------------------------------------------------
# Fused Feature Extractor (combining both streams)
# ---------------------------------------------------------------------------

class DualStreamFeatureExtractor(nn.Module):
    """
    Combines SSL and cepstral features into a fused representation.
    Output: (batch, fused_dim) where fused_dim = 1536 + 120 = 1656 (default)
    """

    def __init__(self, cfg: FeatureConfig, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.ssl_extractor = SSLFeatureExtractor(cfg, device)
        self.cepstral_extractor = CepstralFeatureExtractor(cfg)

    def forward(
        self,
        waveforms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            waveforms: (batch, num_samples) raw 16kHz audio

        Returns:
            fused: (batch, fused_dim)
        """
        # Stream 1: SSL features (on device)
        ssl_features = self.ssl_extractor(
            waveforms.to(self.device, non_blocking=True)
        )  # (B, 1536)

        # Stream 2: Cepstral features (CPU/numpy)
        waveforms_np = waveforms.cpu().numpy()
        cepstral_features = self.cepstral_extractor.extract_batch(waveforms_np)
        cepstral_tensor = torch.tensor(
            cepstral_features, dtype=torch.float32
        ).to(self.device)  # (B, 120)

        # Concatenate along feature dimension
        fused = torch.cat([ssl_features, cepstral_tensor], dim=-1)  # (B, 1656)
        return fused

    def get_output_dim(self) -> int:
        return self.cfg.fused_dim
