"""
Retrieval probe: does mean-pooled CLIP frame embedding carry any signal
that's predictive of the paired sentence, using CLIP's own joint
image-text embedding space?

For a sample of N (video, text) pairs from a split:
  - video_emb = mean over frames of stored CLIP image features (already L2-normalized per-frame)
  - text_emb  = CLIP text encoder applied to the sentence, L2-normalized

Then compute cosine similarity matrix [N, N] and report:
  - video->text top-1 / top-5 retrieval accuracy (diagonal should be highest)
  - text->video top-1 / top-5 retrieval accuracy
  - mean diagonal similarity vs mean off-diagonal similarity
  - chance baseline = 1/N

If retrieval is near chance, mean-pooled CLIP frame features carry
essentially no information about sentence content/semantics in CLIP's
own embedding space -- a strong signal that this feature representation
is the bottleneck, independent of the translator architecture.
"""

import modal

app = modal.App("clip-retrieval-probe")
volume = modal.Volume.from_name("asl-training-data", create_if_missing=False)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install("h5py==3.11.0", "numpy==1.26.4", "transformers==4.41.2")
)


@app.function(image=image, volumes={VOLUME_PATH: volume}, gpu="t4", timeout=1800, memory=16384)
def run_probe(split: str = "val", n_samples: int = 500, seed: int = 42):
    import os
    import h5py
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    h5_path = os.path.join(VOLUME_PATH, "how2sign_clip_text_realigned.h5")

    print("Loading CLIP model (text + image towers)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    print(f"Reading {n_samples} samples from split={split}...")
    with h5py.File(h5_path, "r") as f:
        seq_key = f"{split}_sequences"
        len_key = f"{split}_sequence_lengths"
        txt_key = f"{split}_texts"

        n_total = f[seq_key].shape[0]
        rng = np.random.RandomState(seed)
        n_samples = min(n_samples, n_total)
        indices = rng.choice(n_total, size=n_samples, replace=False)
        indices.sort()

        seq_lens = f[len_key][:][indices]
        texts_raw = f[txt_key][:][indices]
        texts = [
            (t.decode("utf-8") if isinstance(t, bytes) else str(t)).strip()
            for t in texts_raw
        ]

        # Mean-pool video features over valid frames (already L2-normalized per-frame
        # as produced by prepare_how2sign_text_h5.py's _encode_frames_clip).
        video_embs = np.zeros((n_samples, 512), dtype=np.float32)
        for i, idx in enumerate(indices):
            slen = int(seq_lens[i])
            if slen <= 0:
                slen = 1
            feats = f[seq_key][int(idx), :slen, :].astype(np.float32)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            video_embs[i] = feats.mean(axis=0)

    # Re-normalize the mean-pooled video embedding (mean of unit vectors isn't unit norm).
    video_embs = video_embs / np.clip(
        np.linalg.norm(video_embs, axis=1, keepdims=True), 1e-8, None
    )

    print("Encoding text with CLIP text tower...")
    text_embs = []
    batch_size = 64
    for i in range(0, n_samples, batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = processor(
            text=batch_texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        text_embs.append(feats.detach().cpu().numpy().astype(np.float32))
    text_embs = np.concatenate(text_embs, axis=0)

    # Cosine similarity matrix [N, N] (both sides already unit-normalized).
    video_t = torch.from_numpy(video_embs).to(device)
    text_t = torch.from_numpy(text_embs).to(device)
    sim = (video_t @ text_t.T).cpu().numpy()  # [N, N], rows=video, cols=text

    n = sim.shape[0]
    diag = np.diag(sim)
    off_diag_mean = (sim.sum() - diag.sum()) / (n * n - n)

    def topk_acc(matrix, k):
        # For each row, check if the diagonal index is within the top-k columns.
        topk_idx = np.argpartition(-matrix, kth=min(k, n - 1) - 1, axis=1)[:, :k]
        hits = sum(1 for i in range(n) if i in topk_idx[i])
        return hits / n

    v2t_top1 = topk_acc(sim, 1)
    v2t_top5 = topk_acc(sim, 5)
    t2v_top1 = topk_acc(sim.T, 1)
    t2v_top5 = topk_acc(sim.T, 5)

    chance_top1 = 1.0 / n
    chance_top5 = min(5, n) / n

    print()
    print("=" * 60)
    print(f"CLIP RETRIEVAL PROBE  (split={split}, N={n})")
    print("=" * 60)
    print(f"Mean diagonal similarity (matched pairs):     {diag.mean():.4f}")
    print(f"Mean off-diagonal similarity (random pairs):  {off_diag_mean:.4f}")
    print(f"Diagonal - off-diagonal gap:                  {diag.mean() - off_diag_mean:.4f}")
    print()
    print(f"Video -> Text  top-1 acc: {v2t_top1:.4f}  (chance: {chance_top1:.4f})")
    print(f"Video -> Text  top-5 acc: {v2t_top5:.4f}  (chance: {chance_top5:.4f})")
    print(f"Text  -> Video top-1 acc: {t2v_top1:.4f}  (chance: {chance_top1:.4f})")
    print(f"Text  -> Video top-5 acc: {t2v_top5:.4f}  (chance: {chance_top5:.4f})")
    print()

    # A few qualitative examples: for the first 5 videos, show top-3 retrieved texts.
    print("Qualitative examples (video -> top-3 nearest texts):")
    for i in range(min(5, n)):
        order = np.argsort(-sim[i])[:3]
        print(f"\n  TRUE: {texts[i][:90]}")
        for rank, j in enumerate(order):
            marker = "  <-- correct" if j == i else ""
            print(f"    #{rank+1} (sim={sim[i, j]:.3f}): {texts[j][:90]}{marker}")


@app.local_entrypoint()
def main(split: str = "val", n_samples: int = 500, seed: int = 42):
    run_probe.remote(split=split, n_samples=n_samples, seed=seed)
