# Contributing

This project is pre-alpha research code that is being revived. Contributions are
welcome, especially from Deaf and hard-of-hearing contributors, sign-language
researchers, and people who care about accessibility. Read
[ROADMAP.md](ROADMAP.md) before starting on a large change, and
[docs/FAILED_APPROACHES.md](docs/FAILED_APPROACHES.md) before re-trying a
representation that has already failed here.

## Development setup

Requirements: Python 3.10+ and a CPU is enough for tests and the smoke run.

```bash
git clone https://github.com/silassantana/ASL_PROJECT.git
cd ASL_PROJECT
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras, only if you are working on those parts:

```bash
pip install -e ".[mediapipe]"    # MediaPipe + OpenCV extraction/inference
pip install -e ".[train]"        # transformers, sentencepiece, scipy, matplotlib
```

No dataset or checkpoint is needed for the test suite or the smoke run;
they build synthetic tensors.

## Tests, lint, and smoke run

```bash
pytest -q          # or: make test
ruff check .       # or: make lint
make smoke         # CPU-only end-to-end training smoke run
```

The smoke run is the fastest way to check that the model path works. It is
deterministic for a fixed seed (apart from timing) and writes small artifacts
under `runs/smoke/`; do not commit them. If you change code on the smoke path, re-run it and paste the resulting
numbers into [RESULTS.md](RESULTS.md) if they change.

CI runs pytest and ruff on every push and pull request. Keep both green.

## Commit style

This revival uses [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: load pretrained temporal encoder in seq2seq fine-tuning
fix: drop 351 always-zero input dims before normalization
docs: correct landmark count in architecture section
test: assert smoke run writes metrics.json
chore: bump ruff baseline
```

Write small, focused commits. Explain *why* in the body when the change is not
obvious. Do not mix formatting-only changes with behaviour changes in the same
commit.

## Pull requests

- Keep the diff scoped to one change; mention which roadmap milestone it moves.
- Add or update tests for behaviour changes. A bug fix should come with a test
  that fails before it.
- Update the docs you invalidate: `README.md`, `RESULTS.md`, `ROADMAP.md`, or
  `THIRD_PARTY.md` as appropriate. Accuracy matters more than polish here.
- Do not commit datasets, model checkpoints, `.h5`/`.npz` files, or generated
  artifacts. See [THIRD_PARTY.md](THIRD_PARTY.md) for dataset terms and how to
  obtain data.
- Do not report a BLEU/WER number without the command, split, seed, and saved
  metric artifact that produced it. Historical numbers belong in
  [RESULTS.md](RESULTS.md) with the "historical" label, not in new claims.

## Where to start

- **Docs and clarity**: read `README.md` and the docs; if a claim does not
  match the code, open an issue or fix it.
- **Tests**: the legacy scripts are under-tested. Small CPU tests around the
  landmark representation, H5 schemas, and metric functions are high-value.
- **Roadmap**: milestone 1 (wiring the pretrained encoder) and milestone 3
  (trimming the 1,629-dim input to the real 426 coordinates) are the most
  useful code contributions right now.
- **Data pipeline**: improving the scraper's robustness and the H5 builders'
  validation is useful and does not need a GPU.

If you are unsure where to begin, open an issue describing what you want to
work on. For anything involving sign-language data or Deaf community
representation, please involve the community rather than assuming.
