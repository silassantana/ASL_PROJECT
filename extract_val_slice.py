"""
Extract a small slice of the mediapipe H5 (val split) plus gloss vocab and
normalization stats, so the recognizer diagnostic can run fully locally
without downloading the full 25GB file.

Usage:
  modal run extract_val_slice.py --n-samples 100
Produces: val_slice.h5 (downloaded to current directory)
"""

import modal

app = modal.App("extract-val-slice")
volume = modal.Volume.from_name("asl-training-data", create_if_missing=False)
VOLUME_PATH = "/data"

image = modal.Image.debian_slim(python_version="3.11").pip_install("h5py==3.11.0", "numpy==1.26.4")


@app.function(image=image, volumes={VOLUME_PATH: volume}, memory=8192, timeout=600)
def extract_slice(n_samples: int = 100):
    import h5py
    import numpy as np
    import os

    src_path = os.path.join(VOLUME_PATH, "how2sign_mediapipe_clip_1000vocab.h5")
    stats_path = src_path + ".train_stats.npz"
    out_path = "/tmp/val_slice.h5"

    with h5py.File(src_path, "r") as fin, h5py.File(out_path, "w") as fout:
        n = min(n_samples, fin["val_sequences"].shape[0])

        seq_lens = fin["val_sequence_lengths"][:n]
        max_seq_len_in_slice = int(seq_lens.max())

        fout.create_dataset("val_sequences", data=fin["val_sequences"][:n, :max_seq_len_in_slice, :])
        fout.create_dataset("val_sequence_lengths", data=seq_lens)
        fout.create_dataset("val_labels", data=fin["val_labels"][:n])
        fout.create_dataset("val_label_lengths", data=fin["val_label_lengths"][:n])
        fout.create_dataset("gloss_names", data=fin["gloss_names"][:])

        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        print(f"Wrote {n} samples, max_seq_len={max_seq_len_in_slice} to {out_path}")
        print(f"sequences shape: {fout['val_sequences'].shape}")
        print(f"labels shape: {fout['val_labels'].shape}")

    # Bundle stats file alongside, if present.
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            stats_bytes = f.read()
        with open("/tmp/val_slice.h5.train_stats.npz", "wb") as f:
            f.write(stats_bytes)
        print(f"Copied stats file ({len(stats_bytes)} bytes)")
    else:
        print("WARNING: no stats file found")

    with open(out_path, "rb") as f:
        h5_bytes = f.read()

    stats_bytes_out = None
    stats_local = "/tmp/val_slice.h5.train_stats.npz"
    if os.path.exists(stats_local):
        with open(stats_local, "rb") as f:
            stats_bytes_out = f.read()

    return h5_bytes, stats_bytes_out


@app.local_entrypoint()
def main(n_samples: int = 100):
    h5_bytes, stats_bytes = extract_slice.remote(n_samples=n_samples)

    with open("val_slice.h5", "wb") as f:
        f.write(h5_bytes)
    print(f"Saved val_slice.h5 ({len(h5_bytes)} bytes)")

    if stats_bytes is not None:
        with open("val_slice.h5.train_stats.npz", "wb") as f:
            f.write(stats_bytes)
        print(f"Saved val_slice.h5.train_stats.npz ({len(stats_bytes)} bytes)")
