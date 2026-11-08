# Third-party attribution and data terms

This project's **code** is MIT-licensed; see [LICENSE](LICENSE). The
datasets it uses are **not** covered by that license, are not redistributed
here, and have their own terms. This file records what came from where and how
to obtain the data.

## Code

- **This repository** — MIT License, Copyright © 2026 Silas Santana.
- **Microsoft ASL Citizen code scaffolding** — earlier revisions of this
  repository copied `LICENSE` and `CODE_OF_CONDUCT.md` verbatim from
  [microsoft/ASL-citizen-code](https://github.com/microsoft/ASL-citizen-code)
  (MIT-licensed, © Microsoft Corporation), and used the ASL Citizen baseline
  setup as scaffolding. Both copied files were replaced in the 2026-11 revival.
  Any remaining code derived from that project remains under the MIT license
  with Microsoft's copyright notice.
- **Python dependencies** (PyTorch, MediaPipe, h5py, NumPy, T5 via
  `transformers`, OpenCV, yt-dlp, etc.) keep their own licenses. Optional
  dependencies are declared as extras in [pyproject.toml](pyproject.toml).

## Datasets

| Dataset | Source | Used for | Terms |
|---|---|---|---|
| **YouTube-ASL** | Google Research — Uthus, Tanzer & Georg, *YouTube-ASL: A Large-Scale, Open-Domain American Sign Language–English Parallel Corpus* (2023) | Landmark trajectories + English captions; the primary dataset of the current pipeline | CC BY 4.0. Videos stay on YouTube and remain subject to YouTube's Terms of Service |
| **How2Sign** | Duarte et al., *How2Sign: A Large-Scale Multimodal Dataset for Continuous American Sign Language* (CVPR 2021) | Earlier continuous signing experiments (landmarks, CLIP features, gloss recognition) | CC BY-NC 4.0 — non-commercial, **research use only** |
| **ASL Citizen** | Desai et al., *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition* (2023), Microsoft Research | Earlier isolated-sign and gloss-vocabulary experiments | Community-sourced dataset released for research; see the official project page and dataset license |

### YouTube-ASL

The official dataset release is a list of **YouTube video IDs** plus English
captions: 11,093 videos and 984 hours of footage (per the
[official dataset README](https://github.com/google-research/google-research/tree/master/youtube_asl)).
The ID list is distributed in the public Google Cloud bucket
[`gresearch/youtube-asl`](https://console.cloud.google.com/storage/browser/gresearch/youtube-asl).
This repository carries a copy of that ID list (`youtube_asl_video_ids.txt`;
the official README describes 11,093 videos and the tracked list has 11,095
entries) and uses `yt-dlp` to download each video for pose extraction.
No video content, landmark arrays, or captions are redistributed here.

If you use YouTube-ASL, cite:

```bibtex
@misc{uthus2023youtubeasl,
  author = {Uthus, David and Tanzer, Garrett and Georg, Manfred},
  title  = {YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus},
  year   = {2023},
  eprint = {2306.15162},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2306.15162}
}
```

### How2Sign

80+ hours of continuous ASL with English translations, glosses, and 2D/3D
skeletons. It is available for research purposes only under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); download it
through the [official site](https://how2sign.github.io/), which is also where
the current terms live. Do not use How2Sign-derived models or artifacts
commercially without checking those terms. No How2Sign video, feature, or H5
file is included in this repository.

If you use How2Sign, cite:

```bibtex
@inproceedings{Duarte_CVPR2021,
  title     = {{How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language}},
  author    = {Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deepti and DeHaan, Kenneth and Metze, Florian and Torres, Jordi and Giro-i-Nieto, Xavier},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021}
}
```

### ASL Citizen

83,399 isolated sign videos covering 2,731 distinct signs, filmed by 52
signers and collected with consent. It is **not** a continuous-translation
dataset; the pre-revival experiments used it for isolated-sign and gloss
vocabulary work. Obtain it from the
[Microsoft Research project page](https://www.microsoft.com/en-us/research/project/asl-citizen/),
read the dataset license there, and cite the paper if you use it.

`data_csv/gloss_dict.csv` in this repository is a gloss ↔ English-word mapping
derived from the ASL Citizen annotation vocabulary and retained because the
historical training scripts reference a gloss vocabulary. It contains no
per-video data. The large per-video ASL Citizen split CSVs that were previously
committed were removed from version control in the revival.

If you use ASL Citizen, cite:

```bibtex
@misc{desai2023aslcitizen,
  title  = {ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition},
  author = {Desai, Aashaka and Berger, Lauren and Minakov, Fyodor O. and Milan, Vanessa and Singh, Chinmay and Pumphrey, Kriston and Ladner, Richard E. and Daum{\'e} III, Hal and Lu, Alex X. and Caselli, Naomi and Bragg, Danielle},
  year   = {2023},
  eprint = {2304.05934},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2304.05934}
}
```

## What this repository does not contain

- No videos, audio, or raw frames from any dataset.
- No `.npz` pose files, `.h5` feature/dataset files, or model checkpoints.
- No dataset split CSVs from ASL Citizen or How2Sign.

These are excluded by `.gitignore` and by project policy: datasets must be
obtained from their official sources under their own terms. If you add a
dataset-derived artifact, document its provenance in this file and check that
the dataset's license permits redistribution.

## How to obtain the data

1. **YouTube-ASL** — download the video-ID list from the
   [official bucket](https://console.cloud.google.com/storage/browser/gresearch/youtube-asl),
   then run the scraper (long-running; requires network access and `yt-dlp`):

   ```bash
   python scrape_youtube_asl.py \
     --ids youtube_asl_video_ids.txt \
     --output-dir ~/youtube_asl_poses \
     --resume
   ```

   Respect YouTube's Terms of Service and rate limits; some videos will have
   been removed or changed since the dataset was released.

2. **How2Sign** — request and download from <https://how2sign.github.io/>,
   following its research-only CC BY-NC 4.0 terms. The old How2Sign scripts are
   kept for reference but are not part of the current pipeline.

3. **ASL Citizen** — request it from the
   [Microsoft Research project page](https://www.microsoft.com/en-us/research/project/asl-citizen/)
   and accept the dataset license before downloading.
