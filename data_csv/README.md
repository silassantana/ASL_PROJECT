# `data_csv/`

This directory holds **label mappings only** — no video, pose, or feature data.

| File | What it is |
|------|------------|
| `gloss_dict.csv` | Gloss ↔ English word mapping used by the How2Sign / ASL Citizen experiments. |

## Provenance

`gloss_dict.csv` is derived from the annotation vocabulary of the
[ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) dataset
(Microsoft Research). It is kept here only because the historical training
scripts reference a gloss vocabulary. It carries no per-video data.

The large per-video split files (≈5 MB of ASL Citizen metadata) that used to be
committed here were **removed from version control** — the dataset has its own
terms of use, and redistributing its metadata is not this repository's job.
Download ASL Citizen (and its split CSVs) from the official source if you need them.

## Other datasets used by this project

- **YouTube-ASL** — Google Research, CC BY 4.0. See `scrape_youtube_asl.py`.
- **How2Sign** — CC BY-NC 4.0 (non-commercial). See `how2sign_prepare_dataset.py`
  and `extract_how2sign_*.py`. Because of the NC clause, models trained on
  How2Sign should not be used commercially without checking the terms.

See [`../THIRD_PARTY.md`](../THIRD_PARTY.md) for full attribution.
