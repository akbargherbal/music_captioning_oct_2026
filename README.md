# Music Captioning

Given **any music track**, produce Suno-style prompt tags describing its
**instruments**, **vocals**, and **production**.

Pipeline:

```
audio ─▶ ACE-Step Captioner ─▶ prose caption
      ─▶ deterministic lexicon tag extractor ─▶ Suno prompt string
```

The captioner is [`ACE-Step/acestep-captioner`](https://huggingface.co/ACE-Step/acestep-captioner)
(MIT, based on Qwen2.5-Omni-7B). It runs through **HuggingFace `transformers` in
4-bit (nf4)** on the T4, using the reference audio pipeline. Tag extraction is
plain Python — **no hosted LLM**.

A llama.cpp GGUF backend is also supported; see *Alternative backend* below.

## Requirements

- Google Colab with a **T4 GPU** (sm_75). No bf16, no FlashAttention.
- `ffmpeg` / `ffprobe` on `PATH` (used to split long tracks).
- Python 3.10+.

## Quickstart (fresh Colab runtime, start to finish)

Every step is idempotent — re-running skips what already exists. Colab already
provides `torch`, `ffmpeg` and `ffprobe`; only the extra packages are installed.
Both Hugging Face repos are public, so **no HF token is needed**.

```bash
# 0. Get the code (skip if already on this runtime / on Drive).
git clone https://github.com/akbargherbal/music_captioning_oct_2026.git
cd music_captioning_oct_2026

# 1. Dependencies (torch itself is preinstalled on Colab).
pip install -q -r requirements.txt

# 2. Weights: 4-bit captioner (~6.4 GB) + spk_dict.pt (skipped if cached).
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Urabewe/Ace-Step-Captioner-4bit", local_dir="models/Ace-Step-Captioner-4bit",
                  allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
hf_hub_download("ACE-Step/acestep-captioner", "spk_dict.pt", local_dir="models/Ace-Step-Captioner-4bit")
PY

# 3. Run the pipeline on a track (upload one, or point at a path).
python scripts/demo.py /content/elisa_maktooba_leek.mp3

# 4. Optional: CPU-only tests, no model or GPU needed.
python -m pytest tests/ -q
```

Or open `notebooks/colab_setup.ipynb` and **Run all** — it does exactly the
same thing, cell by cell.

> **The runtime is ephemeral.** `models/` lives on the Colab VM and is lost on
> restart, so a fresh runtime re-downloads ~6.4 GB. To avoid that, persist it on
> Drive before step 2, e.g.
> `ln -s /content/drive/MyDrive/music_captioning_models models` (mount Drive
> first). Otherwise just re-run step 2 — it is safe and resumable.

## Example

Input: `elisa_maktooba_leek.mp3` (317 s).

<!-- EXAMPLE_START -->
Measured on a Colab T4: model load **36.2 s**, six 60 s chunks captioned in
**68.5 s total**, peak VRAM **8.0 GiB**. Full numbers in `docs/RESULTS.md`.

A representative chunk caption:

> An emotional Arabic pop ballad driven by a powerful female vocal performance.
> The arrangement blends traditional and modern elements, featuring a prominent
> oud playing melodic lines over a bed of lush, cinematic synth strings and pads.
> A steady, mid-tempo beat combines electronic drum machine elements with the
> distinct sound of a darbuka, while a subtle synth bass provides the foundation.
> The vocalist delivers the lyrics with significant passion and vibrato, her
> voice treated with a spacious reverb that enhances the track's dramatic and
> longing mood.

Resulting tags and Suno prompt:

```json
{
  "instruments": ["nylon-string guitar", "string section", "percussion", "oud",
                  "synth strings", "drum machine", "darbuka", "synth bass",
                  "tabla", "synth pads", "kick", "violin", "accordion"],
  "vocals": ["female vocal", "singing", "vibrato", "melismatic", "melisma", "vocals"],
  "production": ["arpeggiated", "melody", "lush", "cinematic", "bassline",
                 "world-fusion", "melodic", "spacious", "reverb", "clean",
                 "punchy", "dynamic", "lo-fi", "vintage", "sample"],
  "suno_prompt": "nylon-string guitar, string section, percussion, oud, synth strings, drum machine, darbuka, synth bass, tabla, synth pads, kick, violin, accordion, female vocal, singing, vibrato, melismatic, melisma, vocals, arpeggiated, melody, lush, cinematic, bassline, world-fusion, melodic, spacious, reverb, clean, punchy, dynamic, lo-fi, vintage, sample"
}
```
<!-- EXAMPLE_END -->

## Output format

`scripts/demo.py --json` prints:

```json
{
  "instruments": ["..."],
  "vocals": ["..."],
  "production": ["..."],
  "caption": "the raw prose caption",
  "suno_prompt": "instruments, vocals, production"
}
```

Only tags the caption actually mentions are emitted (no guessing, e.g. no vocal
gender unless stated). Long tracks are split into ~60 s chunks, captioned
separately, and merged with order-preserving de-duplication.

## Alternative backend: llama.cpp GGUF

The original pinned GGUF runtime is kept in `src/captioner.py` +
`scripts/build_llamacpp.sh`. It needs a **CUDA build** of the patched fork
(`tpsjr7/llama.cpp`, branch `ted/fix-qwen-audio-cleanup-merge`) because stock
llama.cpp mishandles this model's audio input. There is **no Linux CUDA
prebuilt**, so this path compiles for `sm_75`.

```bash
bash scripts/build_llamacpp.sh
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["acestep-captioner-Q4_K_M.gguf", "acestep-captioner-mmproj-Q8_0.gguf"]:
    print(hf_hub_download("dernet/acestep-captioner-GGUF", f, local_dir="models"))
PY
python scripts/demo.py SONG.mp3 --backend llamacpp
```

## Layout

```
src/captioner_hf.py  # transformers 4-bit backend (current)
src/captioner.py     # llama.cpp GGUF wrapper (alternative)
src/tags.py          # lexicon + to_suno()
src/lexicon.json     # instruments / vocals / production terms
scripts/demo.py      # end-to-end CLI
scripts/build_llamacpp.sh
tests/               # pure-Python, no GPU/model needed
docs/RESULTS.md      # real measured VRAM + timing
```

## Testing

```bash
python -m pytest tests/ -q
```

Runs on CPU with no model present (uses saved caption fixtures).

## Notes & limitations

- The captioner is deterministic (greedy decoding) and capped at 400 tokens.
- VRAM/time are hardware-specific; see `docs/RESULTS.md` for measured numbers.
- The 4-bit checkpoint is a community conversion (`Urabewe`); the base model and
  prompt are unchanged.
- Rejected for this T4 (do not switch silently): MOSS-Music-8B (bf16 ~18 GB) and
  the original ACE-Step F32/BF16 11B weights.

## Attribution

- ACE-Step Captioner and its GGUF conversion: MIT (`ACE-Step`, `dernet`).
- 4-bit conversion: `Urabewe`.
- Underlying architecture: Qwen2.5-Omni.
- Alternative patched runtime: `tpsjr7/llama.cpp`, `koda-dernet/acestep-captioner`.
