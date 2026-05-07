"""
SIP Call Listener — pyVoIP-based server
Registers as a local user on the miniSIP server and listens for incoming calls.
Captured audio is saved as WAV and optionally scored by the deepfake detector.

miniSIP server : 10.95.159.135:5060
Listening user : 100  (configure PASSWORD below)
Caller (mobile) : 101 (Zoiper)

Usage:
    python sip_listener.py
    python sip_listener.py --model outputs/checkpoints/best_model.pt  # with inference
"""

import argparse
import io
import logging
import os
import sys
import time
import wave
from datetime import datetime
from threading import Event, Thread

import pyVoIP
from pyVoIP.VoIP import VoIPPhone, CallState, InvalidStateError

# ---------------------------------------------------------------------------
# Configuration — edit these to match your miniSIP user credentials
# ---------------------------------------------------------------------------
SIP_SERVER_IP   = "10.95.159.135"
SIP_SERVER_PORT = 5060

SIP_USERNAME    = "100"          # The user this script registers as
SIP_PASSWORD    = "100"          # Password set for user 100 in miniSIP

# IP of THIS machine (must be reachable by the miniSIP server)
MY_IP           = "10.95.159.135"   # Change if running on a different host
MY_SIP_PORT     = 5090              # Local SIP port (avoid clashing with 5060)
MY_RTP_PORT_MIN = 10000
MY_RTP_PORT_MAX = 10100

# Where to save captured audio files
RECORDINGS_DIR  = "./recordings"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("sip_listener")

# ---------------------------------------------------------------------------
# Optional: deepfake inference
# ---------------------------------------------------------------------------
_inference_pipeline = None

def load_inference_pipeline(model_path: str):
    """Load the deepfake detection model if a checkpoint is provided."""
    global _inference_pipeline
    try:
        import torch
        from config import get_config
        from feature_engineering import DualStreamFeatureExtractor
        from model import DeepfakeDetector
        from inference import InferencePipeline

        cfg = get_config()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        feature_extractor = DualStreamFeatureExtractor(cfg).to(device)
        model = DeepfakeDetector(cfg).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        feature_extractor.eval()
        model.eval()

        _inference_pipeline = InferencePipeline(model, feature_extractor, cfg, device)
        log.info("Deepfake detection model loaded from %s", model_path)
    except Exception as exc:
        log.warning("Could not load inference pipeline: %s — running in capture-only mode.", exc)


def score_recording(wav_path: str):
    """Run deepfake scoring on a saved WAV file if the pipeline is available."""
    if _inference_pipeline is None:
        return
    try:
        result = _inference_pipeline.score_audio(wav_path)
        verdict = "FAKE" if result["is_fake"] else "REAL"
        log.info(
            "Deepfake score: %.4f  →  %s  (%d segment(s))",
            result["deepfake_score"],
            verdict,
            result["num_segments"],
        )
    except Exception as exc:
        log.warning("Scoring failed: %s", exc)


# ---------------------------------------------------------------------------
# Audio capture helper
# ---------------------------------------------------------------------------
def capture_audio(call, wav_path: str, sample_rate: int = 8000, sample_width: int = 1):
    """
    Read RTP audio from an active call and write it to a WAV file.
    Runs until the call ends or an error occurs.

    pyVoIP delivers raw PCM frames via call.readAudio().
    PCMU (G.711 µ-law) — 8 kHz, 8-bit, mono.
    """
    os.makedirs(os.path.dirname(wav_path) if os.path.dirname(wav_path) else ".", exist_ok=True)
    frames = []

    log.info("Recording started → %s", wav_path)
    try:
        while call.state == CallState.ANSWERED:
            chunk = call.readAudio()
            if chunk:
                frames.append(chunk)
            else:
                time.sleep(0.02)   # 20 ms sleep when buffer is empty
    except Exception as exc:
        log.debug("Audio capture loop ended: %s", exc)

    if not frames:
        log.warning("No audio captured for this call.")
        return

    # Write WAV
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

    duration = sum(len(f) for f in frames) / (sample_rate * sample_width)
    log.info("Recording saved: %s  (%.1f s)", wav_path, duration)


# ---------------------------------------------------------------------------
# Call handler — called by pyVoIP in a new thread for each incoming call
# ---------------------------------------------------------------------------
def on_call(call):
    """Handle one incoming SIP call."""
    caller = call.request.headers.get("From", {}).get("number", "unknown")
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(RECORDINGS_DIR, f"call_{caller}_{ts}.wav")

    log.info("Incoming call from %s", caller)

    try:
        call.answer()
        log.info("Call answered — state: %s", call.state)

        # Capture audio in this same thread (blocks until call ends)
        capture_audio(call, wav_path)

    except InvalidStateError as exc:
        log.warning("Could not answer call: %s", exc)
        return
    except Exception as exc:
        log.error("Unexpected error during call: %s", exc)
    finally:
        try:
            if call.state == CallState.ANSWERED:
                call.hangup()
                log.info("Call with %s ended.", caller)
        except Exception:
            pass

    # Score the recording after the call
    score_recording(wav_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="pyVoIP SIP listener for deepfake research")
    parser.add_argument(
        "--model",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a trained DeepfakeDetector checkpoint for live scoring",
    )
    parser.add_argument("--server",   default=SIP_SERVER_IP,   help="miniSIP server IP")
    parser.add_argument("--port",     default=SIP_SERVER_PORT,  type=int, help="miniSIP SIP port")
    parser.add_argument("--user",     default=SIP_USERNAME,     help="SIP username to register as")
    parser.add_argument("--password", default=SIP_PASSWORD,     help="SIP password")
    parser.add_argument("--my-ip",    default=MY_IP,            help="IP of this machine")
    parser.add_argument("--sip-port", default=MY_SIP_PORT,      type=int, help="Local SIP port")
    args = parser.parse_args()

    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    if args.model:
        load_inference_pipeline(args.model)

    log.info("Starting SIP listener …")
    log.info("  Registering as sip:%s@%s", args.user, args.server)
    log.info("  Local SIP port : %d", args.sip_port)
    log.info("  Recordings dir : %s", os.path.abspath(RECORDINGS_DIR))

    phone = VoIPPhone(
        server=args.server,
        port=args.port,
        username=args.user,
        password=args.password,
        callCallback=on_call,
        myIP=args.my_ip,
        sipPort=args.sip_port,
        rtpPortLow=MY_RTP_PORT_MIN,
        rtpPortHigh=MY_RTP_PORT_MAX,
    )

    phone.start()
    log.info("SIP phone registered. Waiting for calls from user 101 …  (Ctrl-C to stop)")

    stop = Event()
    try:
        stop.wait()          # Block until KeyboardInterrupt
    except KeyboardInterrupt:
        log.info("Shutting down …")
    finally:
        phone.stop()
        log.info("SIP phone stopped.")


if __name__ == "__main__":
    main()