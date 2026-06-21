#!/usr/bin/env python3
"""
Re-tokenize an existing H5 label space into SentencePiece BPE tokens.

This keeps video features unchanged and only rewrites labels/gloss_names.
The output H5 is compatible with train_transformer_encdec.py.
"""

import argparse
import os
import tempfile

import h5py
import numpy as np


def _decode_name(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _labels_to_texts(labels, lengths, id_to_token):
    texts = []
    for i in range(len(lengths)):
        L = int(lengths[i])
        toks = [id_to_token[int(t)] for t in labels[i, :L]]
        texts.append(" ".join(toks))
    return texts


def _encode_texts(sp, texts):
    encoded = [sp.encode(t, out_type=int) for t in texts]
    max_len = max((len(x) for x in encoded), default=1)
    out = np.zeros((len(encoded), max_len), dtype=np.int32)
    out_len = np.zeros((len(encoded),), dtype=np.int32)
    for i, row in enumerate(encoded):
        out_len[i] = len(row)
        if len(row) > 0:
            out[i, : len(row)] = np.asarray(row, dtype=np.int32)
    return out, out_len


def main():
    parser = argparse.ArgumentParser(description="Build BPE-tokenized H5 from existing H5")
    parser.add_argument("--input", required=True, help="Input H5 path")
    parser.add_argument("--output", required=True, help="Output H5 path")
    parser.add_argument("--vocab-size", type=int, default=1500, help="SentencePiece vocab size")
    parser.add_argument("--character-coverage", type=float, default=1.0, help="SentencePiece character coverage")
    parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"], help="SentencePiece model type")
    parser.add_argument("--seed-sentencepiece-size", type=int, default=1000000, help="SentencePiece seed_sentencepiece_size")
    args = parser.parse_args()

    try:
        import sentencepiece as spm
    except Exception as exc:
        raise SystemExit(
            "sentencepiece is required. Install with: pip install sentencepiece\n"
            f"Import error: {exc}"
        )

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise SystemExit("--input and --output must be different files")

    with h5py.File(args.input, "r") as src:
        if "gloss_names" not in src:
            raise SystemExit("Input H5 missing gloss_names dataset")

        id_to_token = [_decode_name(x) for x in src["gloss_names"][:]]

        # Build corpus from train labels.
        train_labels = src["train_labels"][:]
        train_lengths = src["train_label_lengths"][:]
        train_texts = _labels_to_texts(train_labels, train_lengths, id_to_token)

        with tempfile.TemporaryDirectory(prefix="bpe_h5_") as td:
            corpus_path = os.path.join(td, "corpus.txt")
            model_prefix = os.path.join(td, "spm")

            with open(corpus_path, "w", encoding="utf-8") as f:
                for line in train_texts:
                    f.write(line + "\n")

            spm.SentencePieceTrainer.Train(
                input=corpus_path,
                model_prefix=model_prefix,
                vocab_size=int(args.vocab_size),
                character_coverage=float(args.character_coverage),
                model_type=args.model_type,
                seed_sentencepiece_size=int(args.seed_sentencepiece_size),
                input_sentence_size=0,
                shuffle_input_sentence=True,
                normalization_rule_name="identity",
            )

            sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
            vocab_size = int(sp.get_piece_size())
            pieces = [sp.id_to_piece(i) for i in range(vocab_size)]

            with h5py.File(args.output, "w") as dst:
                # Copy attributes, then override label-space metadata.
                for k, v in src.attrs.items():
                    dst.attrs[k] = v
                dst.attrs["num_classes"] = vocab_size
                dst.attrs["tokenization"] = "sentencepiece"
                dst.attrs["sentencepiece_model_type"] = args.model_type
                dst.attrs["sentencepiece_vocab_size"] = vocab_size

                # Store piece vocabulary in gloss_names to keep trainer compatibility.
                dt = h5py.string_dtype("utf-8")
                gloss_ds = dst.create_dataset("gloss_names", (vocab_size,), dtype=dt)
                for i, p in enumerate(pieces):
                    gloss_ds[i] = p

                for split in ["train", "val", "test"]:
                    seq_key = f"{split}_sequences"
                    seq_len_key = f"{split}_sequence_lengths"
                    label_key = f"{split}_labels"
                    label_len_key = f"{split}_label_lengths"

                    if seq_key not in src:
                        continue

                    # Copy feature tensors unchanged.
                    src.copy(seq_key, dst)
                    src.copy(seq_len_key, dst)

                    labels = src[label_key][:]
                    lengths = src[label_len_key][:]
                    texts = _labels_to_texts(labels, lengths, id_to_token)
                    new_labels, new_lengths = _encode_texts(sp, texts)

                    dst.create_dataset(label_key, data=new_labels)
                    dst.create_dataset(label_len_key, data=new_lengths)

            # Persist model near output H5 for reproducibility.
            out_prefix = os.path.splitext(args.output)[0] + ".spm"
            with open(model_prefix + ".model", "rb") as rf, open(out_prefix + ".model", "wb") as wf:
                wf.write(rf.read())
            with open(model_prefix + ".vocab", "rb") as rf, open(out_prefix + ".vocab", "wb") as wf:
                wf.write(rf.read())

    print(f"Wrote BPE H5: {args.output}")
    print(f"Saved SentencePiece model: {os.path.splitext(args.output)[0] + '.spm.model'}")


if __name__ == "__main__":
    main()
