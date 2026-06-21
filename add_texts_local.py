#!/usr/bin/env python3
"""
Add *_texts datasets to the local how2sign_mediapipe_clip_1000vocab.h5.

Same two-pointer alignment as add_texts_to_h5.py (Modal version),
but runs locally where both the H5 and CSVs are present.

Usage:
    python add_texts_local.py [--h5 data/how2sign_mediapipe_clip_1000vocab.h5] [--dry-run]
"""

import argparse
import re
import sys

import h5py
import numpy as np
import pandas as pd

CSV_PATHS = {
    "train": "/home/silass/Code/how-to-sign/how2sign_realigned_train.csv",
    "val":   "/home/silass/Code/how-to-sign/how2sign_realigned_val.csv",
    "test":  "/home/silass/Code/how-to-sign/how2sign_realigned_test.csv",
}


def clean_tokenize(sentence):
    s = str(sentence).lower()
    s = re.sub(r"[^a-z0-9'\- ]", " ", s)
    return s.split()


def sentence_to_label(sentence, word_to_idx):
    return [word_to_idx[w] for w in clean_tokenize(sentence) if w in word_to_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="data/how2sign_mediapipe_clip_1000vocab.h5")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with h5py.File(args.h5, "a") as f:
        gloss_names = [
            g.decode("utf-8") if isinstance(g, bytes) else str(g)
            for g in f["gloss_names"][:]
        ]
        word_to_idx = {w: i for i, w in enumerate(gloss_names)}

        for split in ("train", "val", "test"):
            texts_key = f"{split}_texts"
            labels_key = f"{split}_labels"
            label_len_key = f"{split}_label_lengths"

            print(f"\n=== {split} ===")

            if texts_key in f:
                print(f"  {texts_key} already exists — skipping.")
                continue

            h5_labels = f[labels_key][:]
            h5_label_lens = f[label_len_key][:]
            n_h5 = len(h5_labels)

            df = pd.read_csv(CSV_PATHS[split], sep="\t")
            print(f"  CSV rows: {len(df)}, H5 samples: {n_h5}")

            aligned_texts = []
            h5_ptr = 0
            csv_skipped = 0

            for _, row in df.iterrows():
                if h5_ptr >= n_h5:
                    break
                sentence = str(row["SENTENCE"]).strip()
                expected = sentence_to_label(sentence, word_to_idx)
                if len(expected) == 0:
                    csv_skipped += 1
                    continue
                actual_len = int(h5_label_lens[h5_ptr])
                actual = h5_labels[h5_ptr, :actual_len].tolist()
                if actual == expected:
                    aligned_texts.append(sentence)
                    h5_ptr += 1
                else:
                    csv_skipped += 1

            unmatched = n_h5 - h5_ptr
            print(f"  Matched: {h5_ptr}  |  CSV skipped: {csv_skipped}  |  H5 unmatched: {unmatched}")

            if unmatched > 0:
                print(f"  WARNING: {unmatched} H5 samples have no matching CSV row — filling with label reconstruction.")

            if args.dry_run:
                for i in range(min(3, len(aligned_texts))):
                    print(f"    [{i}] {aligned_texts[i][:80]}")
                continue

            full_texts = list(aligned_texts)
            for i in range(len(aligned_texts), n_h5):
                label_seq = h5_labels[i, : int(h5_label_lens[i])].tolist()
                full_texts.append(" ".join(gloss_names[j] for j in label_seq))

            dt = h5py.string_dtype(encoding="utf-8")
            ds = f.create_dataset(texts_key, shape=(n_h5,), dtype=dt)
            for i, txt in enumerate(full_texts):
                ds[i] = txt
            f.flush()
            print(f"  Written {texts_key}: {n_h5} entries")

    print("\nDone.")


if __name__ == "__main__":
    main()
