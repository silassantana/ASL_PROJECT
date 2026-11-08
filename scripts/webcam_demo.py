#!/usr/bin/env python3
"""EXPERIMENTAL webcam demo: webcam -> MediaPipe landmarks -> seq2seq English.

.. warning::

   The live camera path is **untested on the development machine**: that box has
   no camera and deliberately does not install MediaPipe (heavy / version
   hostile).  Only ``--self-test`` is verified here.  Treat the live loop as a
   scaffold with the right wiring, not as a working translator.  With the smoke
   checkpoint behind it the decoded text is meaningless (it was trained on
   random synthetic poses), and there is no checkpoint trained on real sign
   data in this repository.

Pipeline (live path)
--------------------
::

    OpenCV frame
      -> MediaPipe Holistic :: extract_frame_landmarks() -> [543]
      -> rolling ~48-frame buffer
      -> add_motion_features() -> [T <= 48, 1629]
      -> ASLSeq2Seq.greedy_decode() every --infer-every frames (default: 16)
      -> cv2.putText overlay, quit on q / ESC

Per-frame landmark extraction and the model forward pass share the exact code
path used by ``scripts/infer_landmarks.py`` (the real ``ASLSeq2Seq`` class from
``train_seq2seq_diagnostic.py``), so the demo cannot silently diverge from the
CLI.  The decoder/model-inference part is factored into :func:`run_inference`,
which needs no camera, no OpenCV and no MediaPipe.

Optional dependencies
---------------------
The live path needs ``mediapipe`` and OpenCV (the project's ``[mediapipe]``
extra; ``opencv-python-headless`` also works).  They are imported lazily: if
missing, the demo prints a one-line install hint and exits with code ``2``
instead of a traceback.

Usage
-----
::

    pip install -e ".[mediapipe]"
    python scripts/webcam_demo.py --checkpoint runs/smoke/model.pt --camera 0

    # No camera / no MediaPipe needed -- one forward pass on a synthetic
    # [48, 1629] buffer, prints the decoded (meaningless) text, exits 0:
    python scripts/webcam_demo.py --self-test --checkpoint runs/smoke/model.pt

Exit codes: ``0`` success (including a clean quit and ``--self-test``),
``1`` user error (missing/bad checkpoint, camera unavailable), ``2`` missing
optional dependency (MediaPipe/OpenCV).

No feature normalization is applied: the checkpoint contract carries no
training mean/std, so this demo feeds raw poses like the smoke training run.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import textwrap
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Reuse the inference CLI's model rebuilding / decoding (single code path).
from infer_landmarks import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_MAX_LEN,
    POSE_DIM,
    InferenceError,
    build_model,
    decode_poses,
    load_checkpoint,
    resolve_device,
)

DEFAULT_WINDOW = "ASL landmark demo (experimental)"
BUFFER_FRAMES = 48  # rolling landmark buffer, matches training segment length
MIN_BUFFER_FRAMES = 4  # need a little temporal context before decoding
INFER_EVERY = 16  # decode every N frames (CPU forward pass is not free)
LANDMARK_DIM = 543  # zero-padded MediaPipe Holistic frame (pre-motion)
MAX_OVERLAY_LINES = 2  # cap wrapped prediction lines on screen


# ---------------------------------------------------------------------------
# Testable core: model inference, no camera / OpenCV / MediaPipe involved
# ---------------------------------------------------------------------------


def run_inference(
    model,
    vocab: dict[str, int],
    motion_frames: np.ndarray,
    device: torch.device,
    max_len: int = DEFAULT_MAX_LEN,
) -> str:
    """Greedily decode one sentence from a ``[T, 1629]`` motion-feature buffer.

    ``--self-test`` and the live loop both call this, so the decoder path can be
    exercised on CPU without a webcam or MediaPipe installed.
    """
    frames = np.asarray(motion_frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != POSE_DIM:
        raise InferenceError(
            f"webcam buffer must be [T, {POSE_DIM}] motion features, got shape {frames.shape}"
        )
    return decode_poses(model, frames, vocab, device=device, max_len=max_len)


def _wrap_text(text: str, width: int = 46) -> list[str]:
    """Wrap prediction text into a bounded number of overlay lines."""
    if not text:
        return ["(no prediction)"]
    return textwrap.wrap(text, width=width)[:MAX_OVERLAY_LINES] or ["(no prediction)"]


# ---------------------------------------------------------------------------
# Optional-dependency help
# ---------------------------------------------------------------------------


def _missing_dependency(name: str, package_hint: str, exc: Exception | None = None) -> int:
    """Print a one-line install hint (never a traceback) and return exit code 2."""
    message = (
        f"error: webcam demo needs {name}, which is not usable here. "
        f'Install it with: pip install -e ".[mediapipe]" (or {package_hint})'
    )
    if exc is not None:
        message += f" [{type(exc).__name__}: {exc}]"
    print(message, file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Self-test: one forward pass on a synthetic buffer, exits 0
# ---------------------------------------------------------------------------


def _run_self_test(model, vocab: dict[str, int], device: torch.device, max_len: int) -> int:
    rng = np.random.default_rng(0)
    synthetic = rng.standard_normal((BUFFER_FRAMES, POSE_DIM)).astype(np.float32)
    t0 = time.perf_counter()
    prediction = run_inference(model, vocab, synthetic, device, max_len=max_len)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(
        f"[self-test] device={device}  buffer={BUFFER_FRAMES}x{POSE_DIM}  "
        f"greedy decode (max_len={max_len}) in {elapsed_ms:.1f} ms"
    )
    print(
        f"[self-test] prediction (meaningless with the smoke checkpoint): {prediction or '(empty)'}"
    )
    print("[self-test] OK: model rebuilt from checkpoint and ran one forward pass")
    return 0


# ---------------------------------------------------------------------------
# Live webcam loop
# ---------------------------------------------------------------------------


def _resolve_camera_args(args: argparse.Namespace) -> None:
    """Validate CLI values that argparse cannot express tersely."""
    if args.max_frames is not None and args.max_frames < 1:
        raise InferenceError("--max-frames must be >= 1 (omit it for unlimited)")
    if args.infer_every < 1:
        raise InferenceError("--infer-every must be >= 1")
    if not 1 <= args.max_len <= 510:
        raise InferenceError("--max-len must be between 1 and 510")


def _run_live(args: argparse.Namespace, device: torch.device, model, vocab: dict[str, int]) -> int:
    # Optional dependencies, lazy so --self-test works without them.
    try:
        import cv2
    except ImportError as exc:
        return _missing_dependency("OpenCV (cv2)", "pip install opencv-python-headless", exc)

    if importlib.util.find_spec("mediapipe") is None:
        return _missing_dependency("MediaPipe", "pip install mediapipe")

    # scrape_youtube_asl imports cv2 at module import time, hence the lazy import
    # after the cv2 check above.  It gives us the exact preprocessing functions.
    try:
        from scrape_youtube_asl import (
            add_motion_features,
            create_holistic,
            extract_frame_landmarks,
        )
    except Exception as exc:
        return _missing_dependency(
            "MediaPipe/OpenCV preprocessing (scrape_youtube_asl)", "pip install mediapipe", exc
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        print(
            f"error: could not open camera index {args.camera}. "
            "Check --camera or that no other app is using it.",
            file=sys.stderr,
        )
        return 1

    try:
        holistic = create_holistic()
    except Exception as exc:
        cap.release()
        return _missing_dependency("MediaPipe Holistic", "pip install mediapipe", exc)

    print("webcam demo: EXPERIMENTAL, live path not validated in this environment.")
    print("No trained translator is included; overlay text is meaningless.")
    print("Press q or ESC to quit.")
    print(
        f"camera={args.camera}  window={args.window_name!r}  "
        f"buffer<={BUFFER_FRAMES} frames  infer every {args.infer_every} frames",
        flush=True,
    )

    landmark_buffer: deque[np.ndarray] = deque(maxlen=BUFFER_FRAMES)
    frames_seen = 0
    prediction = ""
    latency_ms = 0.0
    rc = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("error: camera stream ended unexpectedly", file=sys.stderr)
                rc = 1
                break
            frames_seen += 1
            if args.max_frames is not None and frames_seen > args.max_frames:
                break

            landmark_buffer.append(extract_frame_landmarks(frame, holistic))

            if len(landmark_buffer) >= MIN_BUFFER_FRAMES and frames_seen % args.infer_every == 0:
                motion = add_motion_features(np.stack(tuple(landmark_buffer), axis=0))
                t0 = time.perf_counter()
                try:
                    prediction = run_inference(model, vocab, motion, device, max_len=args.max_len)
                except InferenceError as exc:
                    print(f"warning: inference skipped: {exc}", file=sys.stderr)
                latency_ms = (time.perf_counter() - t0) * 1000.0

            lines = [
                "ASL -> English demo (EXPERIMENTAL / untrained model)",
                f"pred: {prediction or '(listening...)'}",
            ]
            if len(lines[1]) > 46:
                lines[1:] = [f"pred: {chunk}" for chunk in _wrap_text(prediction)]
            lines.append(f"infer: {latency_ms:.1f} ms | frames: {frames_seen}")
            for i, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (10, 28 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(args.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()

    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXPERIMENTAL webcam demo for the ASL landmark->text "
        "seq2seq model. Requires a camera and MediaPipe for the "
        "live path; --self-test needs neither.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint .pt (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--window-name",
        default=DEFAULT_WINDOW,
        help=f"OpenCV window title (default: {DEFAULT_WINDOW!r})",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Stop after N frames (default: unlimited)"
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=INFER_EVERY,
        help=f"Decode every N frames (default: {INFER_EVERY})",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help=f"Max decoded tokens (default: {DEFAULT_MAX_LEN})",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run one forward pass on a synthetic buffer and exit 0; needs no camera or MediaPipe",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _resolve_camera_args(args)
        device = resolve_device(args.device)
        checkpoint = load_checkpoint(args.checkpoint)
        model = build_model(checkpoint, device)
    except InferenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.self_test:
        return _run_self_test(model, checkpoint["vocab"], device, args.max_len)

    return _run_live(args, device, model, checkpoint["vocab"])


if __name__ == "__main__":
    raise SystemExit(main())
