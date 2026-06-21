import modal

app = modal.App("inspect-text-h5")
volume = modal.Volume.from_name("asl-training-data", create_if_missing=False)
VOLUME_PATH = "/data"

image = modal.Image.debian_slim(python_version="3.11").pip_install("h5py", "numpy")


@app.function(image=image, volumes={VOLUME_PATH: volume}, memory=8192)
def inspect():
    import h5py
    import os
    import numpy as np
    from collections import Counter

    path = os.path.join(VOLUME_PATH, "how2sign_clip_text_realigned.h5")
    with h5py.File(path, "r") as f:
        print("Keys:", list(f.keys()))
        print("Attrs:", dict(f.attrs))
        print()

        for split in ["train", "val", "test"]:
            seq_key = f"{split}_sequences"
            txt_key = f"{split}_texts"
            len_key = f"{split}_sequence_lengths"

            if seq_key not in f:
                print(f"{split}: missing")
                continue

            n = f[seq_key].shape[0]
            seq_lens = f[len_key][:]

            texts = f[txt_key][:]
            texts = [
                (t.decode("utf-8") if isinstance(t, bytes) else str(t)).strip()
                for t in texts
            ]

            unique_texts = set(texts)
            text_counts = Counter(texts)
            most_common = text_counts.most_common(10)

            word_counts = [len(t.split()) for t in texts]

            print(f"=== {split} ===")
            print(f"  N samples:           {n}")
            print(f"  Unique texts:        {len(unique_texts)} ({100*len(unique_texts)/max(n,1):.1f}%)")
            print(f"  Avg seq_len (frames):{seq_lens.mean():.1f} (min={seq_lens.min()}, max={seq_lens.max()})")
            print(f"  Avg text len (words):{np.mean(word_counts):.1f} (min={min(word_counts)}, max={max(word_counts)})")
            print(f"  Empty texts:         {sum(1 for t in texts if len(t) == 0)}")
            print(f"  Top 10 most common texts:")
            for txt, cnt in most_common:
                pct = 100 * cnt / n
                display = txt if len(txt) <= 70 else txt[:67] + "..."
                print(f"    {cnt:5d} ({pct:5.2f}%)  {display!r}")
            print()

            # Top-10 cumulative coverage
            top10_total = sum(c for _, c in most_common)
            print(f"  Top-10 texts cover {100*top10_total/n:.2f}% of {split} samples")

            # Vocabulary size over all words
            vocab = Counter()
            for t in texts:
                vocab.update(t.split())
            print(f"  Vocabulary size:     {len(vocab)} unique words")
            print(f"  Top 15 words: {vocab.most_common(15)}")
            print()


@app.local_entrypoint()
def main():
    inspect.remote()
