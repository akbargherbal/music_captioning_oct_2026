# Changelog

## [1.0.0] - 2026-10-09

First stable release.

### Added

- Pipeline: audio → ACE-Step captioner → prose caption → deterministic lexicon
  tag extractor → Suno prompt string.
- 4-bit `transformers` backend (nf4, fp16 compute, SDPA) for the Colab T4
  (`src/captioner_hf.py`) — no llama.cpp build required.
- llama.cpp GGUF backend kept as an alternative (`src/captioner.py`,
  `scripts/build_llamacpp.sh`).
- `scripts/demo.py` for a single track, with `--out` to save JSON.
- `scripts/benchmark.py` for a file or a whole album: per-track chunk timings,
  peak VRAM, totals, and `realtime_factor`.
- 27 CPU-only tests (`pytest tests/`).
- `notebooks/colab_setup.ipynb`, `README.md`, and measured numbers in
  `docs/RESULTS.md`.

### Measured on a Colab T4

- Single track, 317 s: 68.5 s captioning, 8.0 GiB peak VRAM.
- Album `Elissa - 2008 Ayami Beek`, 11 tracks, 3034 s: 617 s captioning
  (RTF 0.203, ≈4.9× faster than real time), 8.0 GiB peak VRAM.
