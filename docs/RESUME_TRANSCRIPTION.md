# Resume plan — Elissa + Fadl Shaker full ABC transcription

_Last updated: 2026-10-28 08:48 UTC._
_Status at that moment: **22 / 23 albums committed**. Only
`FADL_SHAKER/15.The Best Of (Al Rawae3).2010` may still be running._

> **بالعربي:** هذا ملف المواصلة. أي شيء داخل `/content` يُمسح عند فصل كولاب.
> الحالة الدائمة في GitHub (`music_captioning_oct_2026`) وفي GCS
> (`TRANSCRIPTS/`). بعد إعادة التشغيل اتبع الخطوات بالترتيب، وخصوصًا
> **استرجاع ملفات ABC من GCS** قبل إكمال أي ألبوم. لا تعِد نسخ الألبومات
> المكتملة ولا تستخدم `--force`.

The authoritative spec is `Elissa_Fadhl_Full_Transcription_Plan.md`; the
runbook is `docs/TRANSCRIPTION.md`; measured numbers are in
`docs/TRANSCRIPTION_RESULTS.md`. This file is only "what do I do next".

---

## 1. What survives a disconnect

| Item | Durable location |
|---|---|
| `outputs/abc_index.json` (which albums are done) | **git** `music_captioning_oct_2026` |
| `scripts/`, `docs/` | **git** |
| Packed album JSON (`outputs/<ARTIST>/abc/<album>.json`) | **GCS** `TRANSCRIPTS/<ARTIST>/abc/<album>.json` — **not in git** |
| Raw SheetSage2 artifacts | **GCS** `TRANSCRIPTS/<ARTIST>/<album>.tar.gz` |
| Caption JSONs (`outputs/<ARTIST>/<album>.json`) | git (already there) |

Lost with `/content`: `work/manifest.jsonl`, `work/audio`, `work/out`,
`work/archives`, the venv, the model weights, all local logs.

A disconnect mid-album loses **only that album**; the last committed album is
safe. There is nothing to clean up.

---

## 2. Rebuild the runtime (≈15–20 min)

```bash
# transcribe repo + venv + weights + abc2midi
git clone https://github.com/akbargherbal/arabmaqamrock.git /content/amr
cd /content/amr && pip install -q uv
./plumbing/setup_transcribe.sh --cuda --warm
apt-get -qq install -y abcmidi
./transcribe.sh --doctor
source /content/amr/.venv-sheetsage2/bin/activate
```

Expected doctor: Python 3.11, torch `2.8.0+cu126`, transformers `4.45.2`,
`cuda: True`, dtype `fp32`. Then check the GPU and disk:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /content
```

Auth checks (if either fails, it needs one interactive login):

```bash
gcloud storage ls gs://akbar-december-2024-backup/ >/dev/null && echo GCS_OK
gh auth status
```

If git pushes fail, run `gh auth setup-git`.

---

## 3. Clone the data repo and rebuild the manifest

```bash
git clone https://github.com/akbargherbal/music_captioning_oct_2026.git /content/mc
cd /content/mc && mkdir -p work

python3 scripts/abc_manifest.py \
  --artists ELISSA FADL_SHAKER \
  --root gs://akbar-december-2024-backup/DISCOGRAPHY \
  --outputs outputs \
  --out work/manifest.jsonl
```

Expect **236 rows / 23 albums**. The manifest is not in git and must be
rebuilt every session.

---

## 4. Restore the packed ABC JSONs from GCS — do not skip

`abc_pack.py` computes `duplicate_of` from **local** album JSONs, and the
dataset builder reads them. Without this step, album 15 would not see the
earlier albums and duplicates would be assigned differently.

```bash
cd /content/mc
mkdir -p outputs/ELISSA/abc outputs/FADL_SHAKER/abc
gcloud storage rsync -r "gs://akbar-december-2024-backup/TRANSCRIPTS/ELISSA/abc" outputs/ELISSA/abc
gcloud storage rsync -r "gs://akbar-december-2024-backup/TRANSCRIPTS/FADL_SHAKER/abc" outputs/FADL_SHAKER/abc
```

Verify every index entry has a local file:

```bash
python3 - <<'PY'
import json, pathlib
idx = json.load(open('outputs/abc_index.json'))['albums']
missing = [v['path'] for v in idx.values()
           if not (pathlib.Path('outputs') / v['path']).is_file()]
print('albums in index:', len(idx))
print('missing local JSONs:', missing or 'none')
PY
```

Both lines must be clean before proceeding.

---

## 5. See what is left

```bash
cd /content/mc
python3 scripts/abc_batch.py --artist FADL_SHAKER --list
python3 scripts/abc_batch.py --artist ELISSA --list
```

- If nothing is pending, go to **§7** (dataset).
- If album 15 is pending, go to **§6**.
- `partial` albums are skipped by design (the `no-abc` rows); see **§8**.

---

## 6. Resume the batch

```bash
cd /content/mc && setsid nohup bash -c 'python3 -u scripts/supervise.py --log /content/abc_batch_resume.log --script scripts/abc_batch.py -- --artist FADL_SHAKER --max-minutes 120; echo $? > /content/logs/bulk_resume.exit' > /content/logs/bulk_resume.out 2>&1 & disown
```

- Album 15 is the big one: 16 tracks, 1.23 h audio, ~14–16 min.
- `--max-minutes 120` is plenty for one album; raise it only if more is pending.
- It only stops **before** an album, never mid-album.
- Watch with:

```bash
tail -20 /content/abc_batch_resume.log
cat /content/logs/bulk_resume.exit      # written only when the supervisor exits
python3 scripts/abc_batch.py --artist FADL_SHAKER --list
```

`exit=0` = clean; `exit=2` = hard stop (upload/commit/systemic download) — read
the last `ABORT:` line before rerunning.

---

## 7. After the last album

```bash
cd /content/mc
python3 scripts/build_abc_dataset.py \
  --outputs outputs --manifest work/manifest.jsonl \
  --out datasets/elissa_fadl
wc -l datasets/elissa_fadl/*.jsonl
```

Expected: 236 rows across `train/val/quarantine` (minus `no-abc` rows, which
go to quarantine). Then update `docs/TRANSCRIPTION_RESULTS.md` with the final
FADL numbers and commit only:

```bash
git add -- outputs/abc_index.json scripts docs
git commit -m "abc: FADL complete — final index and results"
git push
```

**Do not commit** `datasets/` JSONL (it contains ABC text; §15.1 is still
unresolved) or `outputs/<ARTIST>/abc/*.json`. Keep them local and, if a backup
is wanted, upload to GCS under `TRANSCRIPTS/_datasets/`.

---

## 8. Optional: retry the `no-abc` rows

These tracks produced every side artifact except `score.abc`. They are
recorded as `partial` and skipped automatically. `transcribe.py` re-runs a
`no-abc` track on retry and skips the `ok` ones.

| Track | Failure |
|---|---|
| `ELISSA/Elissa - 2012  As3ad Wa7da/05-te3ebt-minnak` | `no-abc` |
| `FADL_SHAKER/01.Walla Zaman.1998/03-adee-wala-tesalemsh` | `no-abc` |
| `FADL_SHAKER/01.Walla Zaman.1998/08-nazra-wahda` | `no-abc` |
| `FADL_SHAKER/02.Bayya'a El Qeloub.1999/04-zaid-el-malam` | `no-abc` |
| `FADL_SHAKER/05.Hobak Khayal.2001/03-hazzak-ya-albi` | `no-abc` |
| `FADL_SHAKER/08.Hafla 2004/05-hobbak-khayal` | `no-abc` |
| `FADL_SHAKER/08.Hafla 2004/09-la-telomuni` | `no-abc` |
| `FADL_SHAKER/10.Saharni El Shoq.2005/04-tadri-lieh` | `no-abc` |

Retry one album:

```bash
python3 scripts/abc_batch.py --artist ELISSA \
  --album "Elissa - 2012  As3ad Wa7da" --retry-failed --max-minutes 30
```

Retry every partial FADL album (re-runs only the `no-abc` tracks, then
repacks/uploads/commits each album):

```bash
python3 scripts/abc_batch.py --artist FADL_SHAKER --retry-failed --max-minutes 60
```

If a track returns `no-abc` again, record it as a model/audio limitation. **Do
not hand-edit an ABC.**

Two `octave_suspect` flags are expected and are never "fixed":
`FADL_SHAKER/07.Fadl Shaker.2003/02-el-oyoun-el-soud` and
`FADL_SHAKER/09.Arabian Night.2004/02-oulou-laeyn-el-shames`.

---

## 9. Known issues / decisions

- **ABC text in git (§15.1): unresolved.** Local + GCS only; never `git add`
  `outputs/<ARTIST>/abc/`.
- **Val split (§10.2): open decision.** Keeping duplicate groups whole projects
  **73 rows / 30.9 % val**, not ~10 %. Either accept it or make `audio_sha256`
  authoritative for grouping. Do not let duplicates straddle.
- **`no-abc` rate:** 8 tracks so far (1/96 ELISSA, 7/124 FADL before album 15).
  All isolated, none stopped the session.
- **Wildcard filenames:** `gcloud storage cp` cannot copy names containing
  `[ ] * ?` (e.g. `13.Kermalak [Remix].mp3`). `abc_batch.py` downloads those
  through the GCS JSON API by exact name. Do not "fix" the filename in GCS.
- **One tar.gz per album:** `TRANSCRIPTS/<ARTIST>/<album>.tar.gz`, verified by
  size + md5 before the checkpoint commit.

---

## 10. Don'ts

- No `git add -A` (would commit audio/ABC/artifacts).
- No `--force` on `abc_batch.py` (bypasses the hash guards).
- Do not write to `outputs/<ARTIST>/<album>.json` (caption files).
- Do not hand-edit an ABC.
- Do not run two batches concurrently on the same GPU/work dir.
- Do not delete a track directory to retry; `transcribe.py` already re-runs
  `no-abc` tracks.
