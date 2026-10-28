# Full ABC transcription — measured results

Only numbers actually measured on this runtime appear here. Anything not run is
marked **not measured**. Spec: `Elissa_Fadhl_Full_Transcription_Plan.md`.

## Environment (2026-10-28, Colab T4)

| Item | Measured |
|---|---|
| GPU | Tesla T4, 15,360 MiB total |
| Driver | 580.82.07 |
| Python (venv) | 3.11.17 |
| torch | 2.8.0+cu126 |
| transformers | 4.45.2 |
| huggingface_hub | 0.36.0 |
| `--doctor` | exit 0, `cuda: True`, dtype `fp32` |
| Free disk at start | ~53 GB |
| abc2midi | 4.88 (2024-02-22) |
| Model | `m-a-p/SheetSage2@b83c3d79a6d98bcffaf8f9ed112c90aa887887bc` |

## Manifest vs plan §2

`work/manifest.jsonl`, 236 rows, 236 unique `track_id`.

| Check | Plan §2 | Measured | Diff |
|---|---:|---:|---|
| Tracks | 236 | 236 | — |
| Albums | 8 ELISSA + 15 FADL = 23 | 8 + 15 = 23 | — |
| Audio | 19.19 h | 19.188 h | −0.002 h |
| Bytes | 2.78 GB | 2.782 GB | +0.002 GB |
| Captioned | 230 | 230 | — |
| `needs_caption` | 6 | 6 | — |
| Tracks >300 s | 21 / 51 / 72 | 21 / **57** / **78** | **+6 FADL, +6 total** |
| Tracks >480 s | 0 / 6 / 6 | 0 / 6 / 6 | — |
| Instrumental `(Music)` | 2 | 2 | — |
| Duplicate titles | 11 titles / 25 rows | 11 / 25 | — |
| Unique `audio_sha256` | 236 (at download) | **not measured** | hashes are null in the manifest until download |

**Plan arithmetic discrepancy:** the §2 value “51 FADL tracks >300 s” is the
*captioned-only* count. The 6 skipped tracks are all >300 s, so the full
inventory is 57 (78 total). The 72 figure is caption-only. The manifest is
right; the plan's row mixes the two populations.

The 6 skipped durations were measured with ffprobe (not the rounded skip
reason): 521.042, 493.195, 717.352, 522.344, 561.293, 524.829 s. Sum 3,340.055 s
= 0.928 h; captioned audio is 65,738.0 s = 18.261 h.

## Pilot (§7)

3 tracks, one `--full` run, `--device cuda`, fp32. Peak VRAM was sampled
independently every 2 s (see the sampling method below); the transcriber's own
`meta.json` does not record it.

| Track | Duration | Status | abc2midi | Vocal / Ins notes | Chords | Coverage | GPU s | RTF |
|---|---:|---|---|---:|---|---:|---:|---:|
| 01. Baddy Doub | 236.8 s | ok | exit 0 | 504 / 394 | yes (130) | 0.996 | 45.8 | 5.17× |
| 03. Ajmal Ihssas | 355.0 s | ok | exit 0 | 632 / 297 | yes (161) | 0.995 | 55.4 | 6.41× |
| 09.Inta 3omri | 717.4 s | ok | exit 0 | 868 / 827 | yes (349) | 1.000 | 138.0 | 5.20× |

- **Peak VRAM:** 3,855 MiB (3.76 GiB), run-level peak. Per-track peaks were not
  sampled.
- **Total:** 1,309.13 s audio / 239.2 s GPU = **5.47× RTF** (per-track seconds).
  Including the 16.3 s model load and 240.0 s wall: 5.45×.
- **Gate verdict:** `≥ 4×` → **2 sessions**. Projected pure decode for 19.188 h:
  **3.51 h**, or **~4.0 h** after 23 per-album model loads, downloads, packing
  and uploads. S1 ELISSA ≈1.7 h, S2 FADL_SHAKER ≈2.6 h; a buffer remains.
- **Caveat:** track 3's 868 Vocal notes are above the §7 inspection band
  (150–700). It is 3× longer than track 1 and has 1.21 notes/s, so this is
  duration scaling, not a new failure mode. The plan only acts on `<50`.
- No §9 QC flags on any pilot track.

## Melody-only equivalence (§9 last row, §15.3)

Three real `--melody-only` runs were made on the same audio, then compared with
`strip_chords(full)`.

| Track | Exact text equal | Note events equal | Unified diff | What differs |
|---|---|---|---|---|
| 01. Baddy Doub | **no** | **yes** | 67 lines | `Z` whole-bar rest compression vs repeated `z32` rests; no pitch/duration change |
| 03. Ajmal Ihssas | **no** | **yes** | 82 lines | same rest compression |
| 09.Inta 3omri | **no** | **yes** | 274 lines | rest compression + 7 tie-split notes in Vocal (868 raw → 861 raw; both merge to 842) |

“Note events equal” means the merged (pitch, duration) sequence of every voiced
note is identical after tying; it is not a claim of byte equality.

**Verdict:** the derived `abc_melody` is **musically valid** and is kept in every
album JSON (§15.3 yes). It is **not textually identical** to a real melody-only
run. If a downstream consumer needs exact melody-only bytes, it must either run
`--melody-only` or normalize rests/ties first; do not treat
`sha256(abc_melody)` as interchangeable with a melody-only run.

## Dataset split projection

This is a **projection from manifest titles**, not a packed-corpus measurement.

| Rule | Val rows | Share |
|---|---:|---:|
| Fixed val albums only (`ELISSA/…2012 As3ad Wa7da`, `FADL_SHAKER/15.The Best Of`) | 30 | 12.7 % |
| Album split **and** duplicate groups kept whole (current code) | **73** | **30.9 %** |

The second row is the strict reading of §10.2. FADL album 15 shares 11 titles
with six other albums (plus live/studio copies), so keeping duplicate groups in
one split pulls a 59-row component into val. This is a real conflict between
“hold out ~10 %” and “keep groups in one split”, not a bug. Correction
proposal at the gate: make the audio hash authoritative for grouping (drop the
normalized-title fallback) and accept whatever val size results; or accept 31 %
val. Do not let duplicates straddle.

## Bulk run 1 — ELISSA (2026-10-28)

8/8 albums committed; 96 tracks: **95 ok, 0 flagged, 1 failed**.

| Metric | Measured |
|---|---:|
| Audio transcribed | 26,374.3 s (7.33 h) |
| GPU seconds (per-track sum) | 4,614.9 s (1.28 h) |
| RTF | **5.715×** |
| Wall, second half (4 albums) | 42.7 min |
| Archives / packed JSONs in GCS | 8 / 8, size + md5 verified |

Per album: Baddy Doub 10 ok (5.29×), Wakherta Maak 11 (4.99×), Aayshalak 10
(5.75×), Ahla Donya 14 (5.74×), Bastannak 13 (5.94×), Ayami Beek 11 (6.05×),
Tesada'a Bemeen 13 (6.13×), As3ad Wa7da 13 ok + 1 failed (5.79×).

**The one failure:** `ELISSA/…/2012  As3ad Wa7da/05. Te3ebt Minnak.mp3` —
`transcribe.py` wrote every artifact except `score.abc` (`meta.status=no-abc`,
45.2 s). The album was packed `partial` and committed; the other 13 tracks were
not lost. Re-run that album with `--retry-failed` to fill the gap.

**Operational note:** one track per album can now fail to download without
stopping the session (see the runbook). During this run the same class of
failure did stop it: `gcloud storage cp` treats `[Remix]` as a wildcard, which
aborted album 5. Both behaviours are now covered by tests.

## Not measured yet

- Bulk FADL_SHAKER (15 albums) and corpus QC histograms.
- Unique `audio_sha256` across all 236 tracks (filled at download).
- Per-track peak VRAM.
- The §9.2 listening check (12 tracks).
- Lyrics coverage (§11, deferred).

## Reproduction notes

- Pilot log: the run's `_driver.log` / `meta.json`; VRAM sampler wrote
  `nvidia-smi --query-gpu=memory.used` every 2 s and reported the maximum.
- `pytest tests/` → **67 passed** on CPU (ABC parse, chord stripping, split
  integrity, dedupe, quarantine, caption join, download skip/systemic policy).
- Raw artifacts are stored as one `tar.gz` per album; the pilot's 5.0 MB raw
  tree compressed to 472 KB (≈10.6×).
