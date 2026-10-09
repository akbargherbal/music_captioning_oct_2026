# Results — real runs

> Rule from `AGENTS.md`: do not claim the pipeline works until it has been run
> end to end on a real T4. This file records actual commands and measurements.

## Environment

| Item | Value |
|---|---|
| Platform | Google Colab |
| GPU | Tesla T4, 15360 MiB VRAM, compute capability 7.5 (sm_75) |
| CUDA driver / toolkit | 13.0 (driver 580.82.07) |
| Python | 3.13 |
| Runtime | `transformers` 5.18 + `bitsandbytes` 0.50.2 (`device_map="auto"`, fp16 compute, SDPA) |
| Model | `Urabewe/Ace-Step-Captioner-4bit` (nf4, ~6.4 GB) + `spk_dict.pt` from `ACE-Step/acestep-captioner` |
| Prompt | `*Task* Describe this audio in detail` |
| Decoding | greedy (`do_sample=False`), `thinker_max_new_tokens=400`, `generation_mode="text"` |

### Runtime change (2026-10-09, user-approved)

The original plan was the llama.cpp GGUF runtime. We built the patched fork
(`tpsjr7/llama.cpp`, branch `ted/fix-qwen-audio-cleanup-merge`) but stopped at
~32% because the CUDA build is slow on Colab's 2 vCPUs and there is no shortcut
for this GPU:

- llama.cpp `b10796` publishes **no Linux CUDA prebuilt** (Windows CUDA only).
- The upstream **Vulkan** Linux prebuilt is unusable here: the Colab image ships
  `libvulkan.so.1` but **no NVIDIA Vulkan ICD**.
- The GGUF author's fix is a 1.88 MB `libmtmd.so` overlay for `b10796`, which
  still needs a base llama.cpp build for CUDA.

The user approved switching to the transformers 4-bit route, which uses the
reference audio pipeline (no audio patch needed).

## Run 1 — `elisa_maktooba_leek.mp3`

- Source file: `/content/elisa_maktooba_leek.mp3`
- Duration: 317.0 s (5:17)
- Chunking: 60 s segments → 6 chunks

### Measurements

| Metric | Value |
|---|---|
| Model load | 36.2 s |
| Chunk caption times | 16.7, 9.7, 11.5, 10.6, 10.2, 9.8 s |
| Total caption time (6 chunks) | 68.5 s |
| Peak VRAM | **8229 MiB (~8.0 GiB)** |
| Status | ✅ end-to-end success, no errors |

The first chunk is slower (CUDA kernel warmup); steady-state is ~10 s per 60 s
of audio.

### Model output (chunk 2 of 6)

> An emotional Arabic pop ballad driven by a powerful female vocal performance.
> The arrangement blends traditional and modern elements, featuring a prominent
> oud playing melodic lines over a bed of lush, cinematic synth strings and
> pads. A steady, mid-tempo beat combines electronic drum machine elements with
> the distinct sound of a darbuka, while a subtle synth bass provides the
> foundation. The vocalist delivers the lyrics with significant passion and
> vibrato, her voice treated with a spacious reverb that enhances the track's
> dramatic and longing mood.

### Extracted tags (merged across 6 chunks)

```json
{
  "instruments": ["nylon-string guitar", "string section", "percussion", "oud",
                  "synth strings", "drum machine", "darbuka", "synth bass",
                  "tabla", "synth pads", "kick", "violin", "accordion"],
  "vocals": ["female vocal", "singing", "vibrato", "melismatic", "melisma", "vocals"],
  "production": ["arpeggiated", "melody", "lush", "cinematic", "bassline",
                 "world-fusion", "melodic", "spacious", "reverb", "clean",
                 "punchy", "dynamic", "lo-fi", "vintage", "sample"]
}
```

Suno prompt:

```
nylon-string guitar, string section, percussion, oud, synth strings, drum machine,
darbuka, synth bass, tabla, synth pads, kick, violin, accordion, female vocal,
singing, vibrato, melismatic, melisma, vocals, arpeggiated, melody, lush, cinematic,
bassline, world-fusion, melodic, spacious, reverb, clean, punchy, dynamic, lo-fi,
vintage, sample
```

### Notes / errors encountered

- `from_pretrained` **always** loads `spk_dict.pt` (speaker map) even when the
  talker is unused; `Urabewe/Ace-Step-Captioner-4bit` omits it. Fetch it from the
  base repo into the model dir.
- The checkpoint ships a `quantization_config` with **bf16 compute**, which the
  T4 cannot do natively, and it overrides any config passed to `from_pretrained`.
  We patch `config.quantization_config["bnb_4bit_compute_dtype"] = "float16"`
  before loading.
- `generate` renamed `return_audio` → `generation_mode`; use
  `generation_mode="text"` to skip the talker/vocoder entirely.
- The reference pipeline needs no 30-second-boundary workaround (that bug was
  specific to stock llama.cpp).

## GPU smoke test

Kept out of `pytest` (manual, needs GPU + model). Reproduce with:

```bash
python scripts/demo.py /content/elisa_maktooba_leek.mp3 --json
```

CPU-only tag-extraction tests: `python -m pytest tests/ -q` (27 passed).
