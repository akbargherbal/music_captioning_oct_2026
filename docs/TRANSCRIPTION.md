# Full ABC transcription — runbook

Operational guide for the SheetSage2 `--full` pass over `ELISSA` and
`FADL_SHAKER`. The spec is
`Elissa_Fadhl_Full_Transcription_Plan.md`; this file is how to run it and what
the scripts actually do. Measured numbers live in
[`TRANSCRIPTION_RESULTS.md`](TRANSCRIPTION_RESULTS.md) — nothing is claimed here
that was not measured there.

## Scope

- **One `--full` transcript per track**: Vocal + Ins voices, chord symbols, key,
  tempo, meters, sections.
- **No lyrics.** SheetSage2 emits no words. `lyrics` stays `null` until the §11
  coverage phase is approved.
- **No maqam labels and no microtones.** Western-tonal model; half-flats are
  rounded. Recorded per track, never "fixed".
- Model pinned at `m-a-p/SheetSage2@b83c3d79a6d98bcffaf8f9ed112c90aa887887bc`.

## Storage contract

| What | Where | Format |
|---|---|---|
| Raw SheetSage2 tree (ABC, MIDI, `.lab`, events, tokens, notation) | `gs://akbar-december-2024-backup/TRANSCRIPTS/<ARTIST>/<album>.tar.gz` | **one archive per album** |
| ABC JSON + QC (`outputs/<ARTIST>/abc/<album>.json`) | local repo + `gs://…/TRANSCRIPTS/<ARTIST>/abc/<album>.json` | JSON |
| Resume index | git: `outputs/abc_index.json` | JSON |
| Code + this runbook | git: `scripts/`, `docs/` | source |
| Audio cache, archives, raw output | `work/` (gitignored) | ephemeral |

**Nothing binary is committed.** In particular the batch driver never runs
`git add -A`; it stages only `outputs/abc_index.json scripts docs`. ABC text in
git is still an open policy decision (§15.1), so the packed album JSONs are
written locally and backed up to GCS, but are **not** committed.

### Why one tar.gz per album

A track produces ~27 files (~1–2.6 MB raw). 236 tracks would be ~6,400 GCS
objects and ~470 MB; the pilot compressed ~10×. Per-album archives give:

- ~23 objects, ~50 MB projected;
- an atomic object per album (no half-uploaded trees);
- an album-level checkpoint that matches the resume unit;
- one size + md5 verification per album instead of thousands of listings.

The trade-off: individual artifacts are not browsable in the GCS console. They
are re-packable from the archive; the ABC text also exists separately in the
packed JSON. `abc_batch.py --no-archive` keeps the old file-by-file mode for
debugging only.

## Phase 0 — manifest (CPU, no GPU)

```bash
cd /content/mc
python3 scripts/abc_manifest.py \
  --artists ELISSA FADL_SHAKER \
  --root gs://akbar-december-2024-backup/DISCOGRAPHY \
  --outputs outputs \
  --out work/manifest.jsonl
```

The script joins `gcloud storage ls -l -r`, the existing caption JSONs, and
`ffprobe` for the 6 tracks the captioner skipped. It exits non-zero unless all
§5 acceptance checks pass. `audio_sha256` is filled at download time, not here.

## Phase 1 — environment (once per Colab VM)

```bash
git clone https://github.com/akbargherbal/arabmaqamrock.git /content/amr
cd /content/amr && pip install -q uv
./plumbing/setup_transcribe.sh --cuda --warm      # ~2.8 GB weights
apt-get -qq install -y abcmidi
./transcribe.sh --doctor
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /content
```

Expected doctor: Python 3.11, torch `2.8.0+cu126`, transformers `4.45.2`,
`cuda: True`, exit 0, dtype `fp32`. The system-Python doctor failing is the
expected wrong-command case.

## Phase 2 — pilot (3 tracks)

```bash
mkdir -p /content/work/audio/pilot /content/work/out/pilot
gcloud storage cp \
  "gs://akbar-december-2024-backup/DISCOGRAPHY/ELISSA/Elissa - 1998 Baddy Doub/01. Baddy Doub.mp3" \
  "gs://akbar-december-2024-backup/DISCOGRAPHY/ELISSA/Elissa - 2002 Aayshalak/03. Ajmal Ihssas.mp3" \
  "gs://akbar-december-2024-backup/DISCOGRAPHY/FADL_SHAKER/Fadl Shaker.Full Discography.BY.RoMaNTiCPoET/04.Fouk El Shouk.2000/09.Inta 3omri.mp3" \
  /content/work/audio/pilot/

cd /content/amr
./transcribe.sh /content/work/audio/pilot \
  --device cuda --full --out-dir /content/work/out/pilot
```

Sample peak VRAM from a second detached shell:

```bash
while pgrep -f "transcribe\.py .*out/pilot" >/dev/null; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  sleep 2
done | sort -n | tail -1
```

Gate: §7 table. If the measured RTF is `< 1.5×`, stop. The recorded verdict is
in `TRANSCRIPTION_RESULTS.md`.

## Phase 3 — bulk (one album = one commit)

Pending albums:

```bash
cd /content/mc
python3 scripts/abc_batch.py --list
```

Launch one artist, bounded, detached through the supervisor:

```bash
cd /content/mc && setsid nohup bash -c \
  'python3 -u scripts/supervise.py --log /content/abc_batch.log \
     --script scripts/abc_batch.py -- --artist ELISSA --max-minutes 170; \
   echo $? > /content/logs/bulk_ELISSA.exit' \
  > /content/logs/bulk_ELISSA.out 2>&1 & disown
```

Then `FADL_SHAKER`, then the compilation last (`--album "15.The Best Of (Al Rawae3).2010"`).
One album at a time is the default and is the right granularity.

Per album the driver:

1. downloads missing MP3s (skip-if-present) and verifies the count;
2. runs `transcribe.sh <album_dir> --device cuda --full --on-collision suffix
   --time-budget <remaining> --out-dir work/out/<ARTIST>/<album>`;
3. runs the §9 QC gate per track (`ok` / `flagged` / `failed`);
4. builds `work/archives/<ARTIST>/<album>.tar.gz`, uploads it, and verifies its
   **size and md5** in GCS;
5. runs `abc_pack.py` → `outputs/<ARTIST>/abc/<album>.json` +
   `outputs/abc_index.json`, uploads the JSON to
   `TRANSCRIPTS/<ARTIST>/abc/<album>.json` and verifies it;
6. commits only `outputs/abc_index.json scripts docs` and pushes, then checks
   `HEAD == @{u}`.

A failed upload or an unlanded commit **exits 2**; the supervisor treats 2 as a
hard stop and does not advance to the next album. Transient failures can be
retried with the same command after inspection.

### Resume after a disconnect

`work/` is gone; GCS and git are not. Redo Phase 1, `git pull`, then rerun the
same command. Albums whose index entry is `done` are skipped. `--force` is not
needed and must not be used (it bypasses the hash guards).

### Rehearsal

`--no-push` commits locally but does not verify origin. `--dry-run` prints the
album plan without downloading or loading the model.

## Scripts

| Script | Role |
|---|---|
| `scripts/abc_manifest.py` | GCS + captions + ffprobe → `work/manifest.jsonl` |
| `scripts/abc_batch.py` | download → transcribe → QC → archive/upload/verify → pack → commit |
| `scripts/abc_pack.py` | raw transcripts → per-album ABC JSON + `outputs/abc_index.json` |
| `scripts/build_abc_dataset.py` | ABC + captions → `datasets/elissa_fadl/{train,val,quarantine}.jsonl` |
| `scripts/abcscore.py` | shared pure-Python ABC parser, chord stripper, QC |
| `scripts/supervise.py` | restart supervisor; `--script` selects the batch, rc=2 is a hard stop |

## QC gate

Applied by `abc_pack.py` and stored per track under `qc`:

| Check | Pass | Failure |
|---|---|---|
| Parse | `abc2midi` exit 0, no "not complete" | `qc.parse=fail` → quarantine downstream |
| Voices | `V: Vocal` and `V: Ins` present | flag |
| Vocal presence | ≥50 pitched notes (instrumentals exempt) | flag |
| Coverage | last beat ≥0.80 × audio duration | flag |
| Octave | no phrase median 12±0.5 semitones from track median | `octave=suspect`, never corrected |
| Range | vocal median inside E3–A5 | flag |
| Key/chord | `key.lab` and `chord.lab` non-empty; tempo 40–200 BPM | flag |
| Melody-only | chord-stripped full vs a real `--melody-only` run | measure; drop derived artifact only if musically different |

`flagged ≠ failed`. Flags travel with the row; nothing is edited. The ABC is
evidence. The maqam-edit pass is a separate human-driven step.

## Dataset

```bash
python3 scripts/build_abc_dataset.py \
  --outputs outputs --manifest work/manifest.jsonl \
  --out datasets/elissa_fadl
```

- Album-level split, never track-level.
- Albums connected by a duplicate (same hash, else `duplicate_of`) are kept in
  one split; a component touching a val album becomes val. This is stricter than
  the plan's ~10 % target and is called out in the results file.
- `qc.parse=fail` rows are written to `quarantine.jsonl`, never dropped.
- `caption_status=skipped_long` rows ship with `caption: null`.
- `lyrics` is `null` until §11.

## Decisions still open

- **§15.1 (ABC text in git): unresolved.** Current behaviour: local + GCS only,
  never `git add`.
- **§15.2 (re-caption the 6 long tracks): deferred.**
- **§15.3 (derived melody-only ABC): accepted**, measured against 3 real
  melody-only runs.
- **§15.4 (lyrics): deferred.**

## Don't

- Don't `git add -A`; it would commit `work/`, audio, or ABC text.
- Don't hand-edit an ABC to "fix" it.
- Don't write to `outputs/<ARTIST>/<album>.json` (caption files).
- Don't delete a track directory to retry; use `--force`.
- Don't claim a number that is not in `TRANSCRIPTION_RESULTS.md`.
