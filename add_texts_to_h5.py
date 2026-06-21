"""
Align How2Sign CSV sentences with the mediapipe H5 and add *_texts datasets.

extract_mediapipe_dataset.py walks CSV rows in order and skips rows where
the video file is missing or all sentence words are OOV. The H5 samples are
therefore an ordered subset of the CSV rows. We recover the mapping via a
two-pointer walk: for each CSV row, tokenize the sentence → label sequence
using the stored gloss_names vocabulary, then compare against the next H5
label. If they match, the sentence belongs to this H5 sample; if not, the
CSV row was a failed/skipped extraction.

Result: a new *_texts variable-length string dataset in the H5, aligned 1-1
with *_sequences / *_labels.
"""

import modal

app = modal.App("add-texts-to-h5")
volume = modal.Volume.from_name("asl-training-data", create_if_missing=False)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("h5py==3.11.0", "numpy==1.26.4", "pandas==2.2.2")
    .add_local_file(
        "/home/silass/Code/how-to-sign/how2sign_realigned_train.csv",
        remote_path="/root/how2sign_realigned_train.csv",
    )
    .add_local_file(
        "/home/silass/Code/how-to-sign/how2sign_realigned_val.csv",
        remote_path="/root/how2sign_realigned_val.csv",
    )
    .add_local_file(
        "/home/silass/Code/how-to-sign/how2sign_realigned_test.csv",
        remote_path="/root/how2sign_realigned_test.csv",
    )
)

CSV_PATHS = {
    "train": "/root/how2sign_realigned_train.csv",
    "val":   "/root/how2sign_realigned_val.csv",
    "test":  "/root/how2sign_realigned_test.csv",
}


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=1800, memory=16384)
def add_texts(dry_run: bool = False):
    import os
    import re

    import h5py
    import numpy as np
    import pandas as pd

    h5_path = os.path.join(VOLUME_PATH, "how2sign_mediapipe_clip_1000vocab.h5")

    # Reconstruct word_to_idx from the stored gloss_names.
    with h5py.File(h5_path, "r") as f:
        gloss_names = [
            g.decode("utf-8") if isinstance(g, bytes) else str(g)
            for g in f["gloss_names"][:]
        ]
    word_to_idx = {w: i for i, w in enumerate(gloss_names)}

    def clean_tokenize(sentence):
        s = str(sentence).lower()
        s = re.sub(r"[^a-z0-9'\- ]", " ", s)
        return s.split()

    def sentence_to_label(sentence):
        return [word_to_idx[w] for w in clean_tokenize(sentence) if w in word_to_idx]

    with h5py.File(h5_path, "a") as f:
        for split in ("train", "val", "test"):
            texts_key = f"{split}_texts"
            labels_key = f"{split}_labels"
            label_len_key = f"{split}_label_lengths"

            print(f"\n=== {split} ===")

            if texts_key in f:
                print(f"  {texts_key} already exists — skipping.")
                continue

            h5_labels = f[labels_key][:]        # [N, max_label_len] int32
            h5_label_lens = f[label_len_key][:] # [N] int32
            n_h5 = len(h5_labels)

            df = pd.read_csv(CSV_PATHS[split], sep="\t")
            print(f"  CSV rows: {len(df)}, H5 samples: {n_h5}")

            aligned_texts = []
            h5_ptr = 0
            csv_matched = 0
            csv_skipped = 0

            for _, row in df.iterrows():
                if h5_ptr >= n_h5:
                    break

                sentence = str(row["SENTENCE"]).strip()
                expected = sentence_to_label(sentence)

                if len(expected) == 0:
                    # All OOV → would have been skipped during extraction.
                    csv_skipped += 1
                    continue

                actual_len = int(h5_label_lens[h5_ptr])
                actual = h5_labels[h5_ptr, :actual_len].tolist()

                if actual == expected:
                    aligned_texts.append(sentence)
                    h5_ptr += 1
                    csv_matched += 1
                else:
                    # Label mismatch → this CSV row was a failed video extraction.
                    csv_skipped += 1

            unmatched = n_h5 - h5_ptr
            print(f"  Matched: {csv_matched}  |  CSV skipped: {csv_skipped}  |  H5 unmatched: {unmatched}")

            if unmatched > 0:
                print(f"  WARNING: {unmatched} H5 samples have no matching CSV row.")
                print("  Falling back to label-reconstructed text for unmatched samples.")

            if dry_run:
                print(f"  DRY RUN — would write {len(aligned_texts)} texts (+ {unmatched} reconstructed)")
                for i in range(min(5, len(aligned_texts))):
                    reconstructed = " ".join(
                        gloss_names[j]
                        for j in h5_labels[i, : int(h5_label_lens[i])].tolist()
                    )
                    print(f"    [{i}] CSV:    {aligned_texts[i][:90]}")
                    print(f"    [{i}] LABELS: {reconstructed[:90]}")
                continue

            # Build full text list, filling unmatched with label reconstruction.
            full_texts = list(aligned_texts)
            for i in range(len(aligned_texts), n_h5):
                label_seq = h5_labels[i, : int(h5_label_lens[i])].tolist()
                full_texts.append(" ".join(gloss_names[j] for j in label_seq))

            # Write variable-length string dataset.
            dt = h5py.string_dtype(encoding="utf-8")
            ds = f.create_dataset(texts_key, shape=(n_h5,), dtype=dt)
            for i, txt in enumerate(full_texts):
                ds[i] = txt

            f.flush()
            print(f"  Written {texts_key}: {n_h5} entries")

        volume.commit()

    print("\nDone.")


@app.local_entrypoint()
def main(dry_run: bool = False):
    add_texts.remote(dry_run=dry_run)
