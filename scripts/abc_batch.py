#!/usr/bin/env python3
"""Per-album ABC transcription driver for ephemeral Colab runtimes (§8.1).

One album is the checkpoint. For each pending album:

1. download its MP3s from GCS (skip-if-present) and verify the count,
2. run ``transcribe.sh --device cuda --full`` into ``work/out/<artist>/<album>``,
3. run the §9 QC gate per track (ok / flagged / failed),
4. ``gcloud storage rsync -r`` the raw tree and **verify it landed**,
5. pack ``outputs/<artist>/abc/<album>.json`` + ``outputs/abc_index.json``,
   upload the packed JSON to GCS and **verify it landed**,
6. commit ``outputs/abc_index.json`` + ``scripts/`` + ``docs/`` and push,
   verifying the commit is not ahead of origin.

Hard rule: a failed upload or a commit that did not land aborts the session
before the next album starts. ABC text is never ``git add``-ed (§15.1 open).

Usage:
    python3 scripts/abc_batch.py --artist ELISSA --max-minutes 170
    python3 scripts/abc_batch.py --list
    python3 scripts/abc_batch.py --dry-run --artist FADL_SHAKER
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from abcscore import qc_track  # noqa: E402

DEFAULT_ROOT = "gs://akbar-december-2024-backup/DISCOGRAPHY"
DEFAULT_TRANSCRIPTS = "gs://akbar-december-2024-backup/TRANSCRIPTS"
FALLBACK_GIT_NAME = "akbargherbal"
FALLBACK_GIT_EMAIL = "akbargherbal@users.noreply.github.com"

TRANSCRIBE = "transcribe.py"


class BatchError(RuntimeError):
    """A session-stopping failure (bad upload, unpacked commit, no manifest)."""


def run(cmd: list[str], *, cwd: Path | None = None,
        log: Path | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    if log is None:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              check=False, timeout=timeout)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                              text=True, check=False, timeout=timeout)


def load_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise BatchError(f"manifest not found: {path} (run scripts/abc_manifest.py)")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_albums(rows: list[dict]) -> dict[str, list[dict]]:
    albums: dict[str, list[dict]] = {}
    for row in rows:
        albums.setdefault(f"{row['artist']}/{row['album']}", []).append(row)
    for tracks in albums.values():
        tracks.sort(key=lambda r: (r.get("track_no") or 0, r["file"]))
    return albums


def load_index(path: Path) -> dict:
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("version", 1)
        data.setdefault("albums", {})
        return data
    return {"version": 1, "albums": {}}


def album_pending(key: str, index: dict, outputs: Path,
                  force: bool, retry_failed: bool) -> bool:
    entry = index["albums"].get(key)
    if force or not entry:
        return True
    rel = entry.get("path")
    local = outputs / rel if rel else None
    if entry.get("status") == "done" and local and local.is_file():
        return False
    if retry_failed and entry.get("status") in ("partial", "failed"):
        return True
    # Index says done but the local packed file is absent: the album is safe
    # in GCS/git, so do not spend GPU on it; restore happens on demand.
    return not (entry.get("status") == "done" and rel)


def ensure_git_identity(repo: Path) -> None:
    for field, value in (("user.name", FALLBACK_GIT_NAME),
                         ("user.email", FALLBACK_GIT_EMAIL)):
        current = run(["git", "-C", str(repo), "config", field])
        if current.returncode != 0 or not current.stdout.strip():
            run(["git", "-C", str(repo), "config", field, value])


GCS_WILDCARDS = set("*?[]")


def download_object_api(uri: str, dest: Path) -> None:
    """Download one object by exact name via the GCS JSON API.

    The gcloud CLI treats ``[``/``]``/``*``/``?`` in *both* source and
    destination as URL wildcards, so ``13.Kermalak [Remix].mp3`` cannot be
    copied with it at all. The API has no wildcard language.
    """
    if not uri.startswith("gs://"):
        raise BatchError(f"not a GCS URI: {uri}")
    bucket, _, obj = uri[5:].partition("/")
    if not bucket or not obj:
        raise BatchError(f"cannot parse GCS URI: {uri}")
    token = run(["gcloud", "auth", "print-access-token"])
    if token.returncode != 0 or not token.stdout.strip():
        raise BatchError("cannot get a gcloud access token for the GCS API")
    url = (f"https://storage.googleapis.com/storage/v1/b/"
           f"{urllib.parse.quote(bucket, safe='')}/o/"
           f"{urllib.parse.quote(obj, safe='')}?alt=media")
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token.stdout.strip()}"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=300) as response, \
                tmp.open("wb") as fh:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                fh.write(block)
        tmp.replace(dest)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise BatchError(f"GCS API download failed for {uri}: {exc}") from exc


def download_album(rows: list[dict], album_dir: Path,
                   log: Path) -> None:
    album_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        dest = album_dir / row["file"]
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        if GCS_WILDCARDS & set(row["source_uri"] + str(dest)):
            download_object_api(row["source_uri"], dest)
        else:
            proc = run(["gcloud", "storage", "cp", row["source_uri"], str(dest)],
                       log=log)
            if proc.returncode != 0:
                raise BatchError(f"download failed: {row['source_uri']}")
    missing = [r["file"] for r in rows if not (album_dir / r["file"]).is_file()]
    if missing:
        raise BatchError(f"{len(missing)} track(s) missing after download: "
                         f"{missing[:3]}")


def gcs_inventory(uri: str) -> dict[str, int]:
    """``{relative_path: size}`` for everything under a GCS prefix."""
    prefix = uri.rstrip("/") + "/"
    proc = run(["gcloud", "storage", "ls", "-l", "-r", prefix + "**"])
    if proc.returncode != 0:
        raise BatchError(f"gcloud ls failed for {uri}: {proc.stderr.strip()}")
    out: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3 or not parts[2].startswith(prefix):
            continue
        try:
            out[parts[2][len(prefix):]] = int(parts[0])
        except ValueError:
            continue
    return out


def local_inventory(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_size
            for p in root.rglob("*") if p.is_file()}


def verify_upload(local: Path, remote_uri: str) -> None:
    expected = local_inventory(local)
    actual = gcs_inventory(remote_uri)
    missing = sorted(set(expected) - set(actual))
    wrong_size = sorted(k for k in expected if k in actual and actual[k] != expected[k])
    if missing or wrong_size:
        raise BatchError(
            f"upload verification failed for {remote_uri}: "
            f"{len(missing)} missing, {len(wrong_size)} size mismatch; "
            f"first missing={missing[:3]}")


def make_archive(out_dir: Path, archive: Path, log: Path) -> Path:
    """One deterministic-ish tar.gz per album (raw artifacts only)."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    proc = run(["tar", "-czf", str(archive), "-C", str(out_dir), "."], log=log)
    if proc.returncode != 0 or not archive.is_file():
        raise BatchError(f"archive creation failed: {archive}")
    return archive


def md5_b64(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def verify_archive(local: Path, remote_uri: str) -> None:
    """Verify the one archive object by size and GCS md5 (hard stop on mismatch)."""
    listing = run(["gcloud", "storage", "ls", "-l", remote_uri])
    if listing.returncode != 0:
        raise BatchError(f"archive not visible in GCS: {remote_uri}")
    try:
        remote_size = int(listing.stdout.split(None, 1)[0])
    except (ValueError, IndexError):
        raise BatchError(f"cannot read remote size for {remote_uri}")
    if remote_size != local.stat().st_size:
        raise BatchError(
            f"archive size mismatch: local={local.stat().st_size} "
            f"remote={remote_size} ({remote_uri})")

    remote = run(["gcloud", "storage", "hash", remote_uri])
    remote_md5 = ""
    for line in remote.stdout.splitlines():
        if line.startswith("md5_hash:"):
            remote_md5 = line.split(":", 1)[1].strip()
    local_md5 = md5_b64(local)
    if remote.returncode != 0 or remote_md5 != local_md5:
        raise BatchError(
            f"archive md5 mismatch for {remote_uri}: "
            f"local={local_md5} remote={remote_md5 or 'unreadable'}")


def run_transcribe(album_dir: Path, out_dir: Path, remaining_s: float,
                   amr_dir: Path, log: Path) -> int:
    cmd = [str(amr_dir / "transcribe.sh"), str(album_dir),
           "--device", "cuda", "--full", "--on-collision", "suffix",
           "--out-dir", str(out_dir)]
    if remaining_s > 0:
        cmd += ["--time-budget", str(int(remaining_s))]
    proc = run(cmd, cwd=amr_dir, log=log)
    return proc.returncode


def qc_album(out_dir: Path, rows: list[dict]) -> dict[str, str]:
    """Map stem/file -> ok|flagged|failed using the §9 gate."""
    by_stem = {Path(r["file"]).stem: r for r in rows}
    result: dict[str, str] = {}
    for track_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        row = by_stem.get(track_dir.name)
        if row is None:
            # --on-collision suffix appends _2, _3 to the stem.
            candidates = [r for r in rows
                          if track_dir.name.startswith(Path(r["file"]).stem)]
            row = candidates[0] if len(candidates) == 1 else None
        if row is None:
            continue
        meta_path = track_dir / "meta.json"
        if not meta_path.is_file():
            result[row["file"]] = "failed"
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "ok":
            result[row["file"]] = "failed"
            continue
        qc = qc_track(track_dir, meta.get("duration_s"), file=row["file"])
        result[row["file"]] = "ok" if not qc["flags"] else "flagged"
    return result


def pack_album(repo: Path, outputs: Path, manifest: Path, index: Path,
               artist: str, album: str, log: Path,
               raw_gcs: str | None = None) -> None:
    cmd = [sys.executable, str(repo / "scripts" / "abc_pack.py"),
           "--outputs", str(outputs), "--manifest", str(manifest),
           "--artist", artist, "--album", album, "--index", str(index)]
    if raw_gcs:
        cmd += ["--raw-gcs", raw_gcs]
    proc = run(cmd, log=log)
    if proc.returncode != 0:
        raise BatchError(f"abc_pack failed for {artist}/{album}")
    packed = outputs / artist / "abc" / f"{album}.json"
    if not packed.is_file():
        raise BatchError(f"packed JSON not written: {packed}")


def upload_packed(packed: Path, artist: str, album: str,
                  transcripts_root: str, log: Path) -> str:
    dest = f"{transcripts_root.rstrip('/')}/{artist}/abc/{album}.json"
    proc = run(["gcloud", "storage", "cp", str(packed), dest], log=log)
    if proc.returncode != 0:
        raise BatchError(f"packed JSON upload failed: {dest}")
    listing = run(["gcloud", "storage", "ls", "-l", dest])
    if listing.returncode != 0:
        raise BatchError(f"packed JSON not visible in GCS: {dest}")
    try:
        remote_size = int(listing.stdout.split(None, 1)[0])
    except (ValueError, IndexError):
        raise BatchError(f"cannot read remote size for {dest}")
    if remote_size != packed.stat().st_size:
        raise BatchError(f"packed JSON size mismatch at {dest}")
    return dest


def commit_checkpoint(repo: Path, message: str, no_push: bool,
                      log: Path, retries: int = 3) -> None:
    ensure_git_identity(repo)
    # IMPORTANT: never `git add -A` -- it would commit the ABC text (§15.1).
    proc = run(["git", "-C", str(repo), "add", "--",
                "outputs/abc_index.json", "scripts", "docs"], log=log)
    if proc.returncode != 0:
        raise BatchError("git add failed")
    staged = run(["git", "-C", str(repo), "diff", "--cached", "--quiet"])
    if staged.returncode != 0:
        proc = run(["git", "-C", str(repo), "commit", "-q", "-m", message],
                   log=log)
        if proc.returncode != 0:
            raise BatchError("git commit failed")
    tracked = run(["git", "-C", str(repo), "ls-files", "--error-unmatch",
                   "outputs/abc_index.json"])
    if tracked.returncode != 0:
        raise BatchError("outputs/abc_index.json is not tracked")
    if no_push:
        print("  (--no-push: commit kept local, not verified against origin)")
        return
    for attempt in range(1, retries + 1):
        proc = run(["git", "-C", str(repo), "push", "-u", "origin", "HEAD"],
                   log=log)
        if proc.returncode == 0:
            break
        print(f"  push failed ({attempt}/{retries})")
        time.sleep(2 * attempt)
    else:
        raise BatchError("git push failed; aborting before the next album")

    head = run(["git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip()
    upstream = run(["git", "-C", str(repo), "rev-parse", "@{u}"])
    if upstream.returncode != 0:
        raise BatchError("no upstream after push; commit did not land")
    if head != upstream.stdout.strip():
        raise BatchError("HEAD is ahead of origin after push; commit did not land")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artist", default=None)
    ap.add_argument("--album", default=None)
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop before an album that would overrun; 0 = no limit")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--amr-dir", type=Path, default=Path("/content/amr"))
    ap.add_argument("--repo", type=Path, default=REPO_ROOT)
    ap.add_argument("--work-dir", type=Path, default=Path("/content/work"))
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--outputs", type=Path, default=None)
    ap.add_argument("--index", type=Path, default=None)
    ap.add_argument("--gcs-root", default=DEFAULT_ROOT)
    ap.add_argument("--gcs-transcripts", default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--no-archive", action="store_true",
                    help="upload the raw tree file-by-file instead of one "
                         "per-album tar.gz (debugging only)")
    ap.add_argument("--assumed-rtf", type=float, default=2.5,
                    help="conservative audio-seconds per GPU-second for the "
                         "pre-album budget check")
    args = ap.parse_args(argv)

    outputs = args.outputs or (args.repo / "outputs")
    index_path = args.index or (outputs / "abc_index.json")
    manifest_path = args.manifest
    if manifest_path is None:
        for candidate in (args.repo / "work" / "manifest.jsonl",
                          args.work_dir / "manifest.jsonl",
                          Path("/content/manifest.jsonl")):
            if candidate.is_file():
                manifest_path = candidate
                break
    if manifest_path is None:
        raise BatchError("no manifest; pass --manifest")
    if not (args.amr_dir / "transcribe.sh").is_file():
        raise BatchError(f"transcribe.sh not found in {args.amr_dir}")

    rows = load_manifest(manifest_path)
    albums = group_albums(rows)
    index = load_index(index_path)
    pending = [k for k in sorted(albums)
               if album_pending(k, index, outputs, args.force, args.retry_failed)]
    if args.artist:
        pending = [k for k in pending if k.split("/", 1)[0] == args.artist]
    if args.album:
        pending = [k for k in pending
                   if k.split("/", 1)[1] == args.album]

    if args.list or args.dry_run:
        print(f"manifest: {manifest_path} ({len(rows)} tracks, {len(albums)} albums)")
        print(f"index:    {index_path} ({len(index['albums'])} packed)")
        for key in pending:
            audio_s = sum(r.get("duration_s") or 0 for r in albums[key])
            print(f"  {'PLAN' if args.dry_run else 'PENDING'}  {key}  "
                  f"{len(albums[key])} tracks  {audio_s/3600:.2f} h")
        if not pending:
            print("  (nothing pending)")
        return 0

    started = time.monotonic()
    budget_s = args.max_minutes * 60 if args.max_minutes else 0.0
    log_dir = args.work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    for key in pending:
        artist, album = key.split("/", 1)
        tracks = albums[key]
        audio_s = sum(r.get("duration_s") or 0 for r in tracks)
        elapsed = time.monotonic() - started
        remaining = budget_s - elapsed if budget_s else 0.0
        estimate = audio_s / max(args.assumed_rtf, 0.1) + 60
        if budget_s and remaining < estimate:
            print(f"stopping before {key}: {remaining/60:.1f} min left, "
                  f"estimated {estimate/60:.1f} min (never start mid-album)")
            break

        print(f"\n== {key} ({len(tracks)} tracks, {audio_s/3600:.2f} h) ==")
        log = log_dir / f"abc_{artist}_{album}.log".replace(" ", "_")
        album_dir = args.work_dir / "audio" / artist / album
        out_dir = args.work_dir / "out" / artist / album

        download_album(tracks, album_dir, log)

        t0 = time.monotonic()
        rc = run_transcribe(album_dir, out_dir, remaining, args.amr_dir, log)
        gpu_s = time.monotonic() - t0
        statuses = qc_album(out_dir, tracks)
        n_ok = sum(v == "ok" for v in statuses.values())
        n_flagged = sum(v == "flagged" for v in statuses.values())
        n_failed = sum(v == "failed" for v in statuses.values())
        print(f"  transcribe exit={rc}  wall={gpu_s/60:.1f} min  "
              f"qc: {n_ok} ok / {n_flagged} flagged / {n_failed} failed")
        if not statuses:
            raise BatchError(f"{key}: no track output; see {log}")

        # 1) raw artifacts land in GCS as ONE archive (hard stop on mismatch).
        raw_gcs = None
        if args.no_archive:
            raw_uri = f"{args.gcs_transcripts.rstrip('/')}/{artist}/{album}"
            proc = run(["gcloud", "storage", "rsync", "-r", str(out_dir), raw_uri],
                       log=log)
            if proc.returncode != 0:
                raise BatchError(f"raw rsync failed for {key}; aborting session")
            verify_upload(out_dir, raw_uri)
            raw_gcs = raw_uri
            print(f"  raw tree verified: {raw_uri}")
        else:
            archive = args.work_dir / "archives" / artist / f"{album}.tar.gz"
            make_archive(out_dir, archive, log)
            raw_uri = (f"{args.gcs_transcripts.rstrip('/')}/"
                       f"{artist}/{album}.tar.gz")
            proc = run(["gcloud", "storage", "cp", str(archive), raw_uri],
                       log=log)
            if proc.returncode != 0:
                raise BatchError(f"raw archive upload failed for {key}; "
                                 f"aborting session")
            verify_archive(archive, raw_uri)
            raw_gcs = raw_uri
            print(f"  raw archive verified ({archive.stat().st_size/1e6:.1f} MB): "
                  f"{raw_uri}")

        # 2) pack + index, then the packed JSON lands in GCS
        pack_album(args.repo, outputs, manifest_path, index_path,
                   artist, album, log, raw_gcs)
        packed = outputs / artist / "abc" / f"{album}.json"
        upload_packed(packed, artist, album, args.gcs_transcripts, log)
        print(f"  packed JSON verified in GCS")

        # 3) commit only the index/scripts/docs; ABC text stays out of git
        index = load_index(index_path)
        message = (f"abc({artist}/{album}): {n_ok + n_flagged} tracks "
                   f"({n_ok} ok, {n_flagged} flagged, {n_failed} failed)")
        commit_checkpoint(args.repo, message, args.no_push, log)
        print(f"  committed: {message}")

        if budget_s and time.monotonic() - started >= budget_s:
            print("time budget reached; stopping cleanly after a complete album")
            break

    print(f"\ndone in {(time.monotonic()-started)/60:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatchError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        # Exit 2 is a hard stop: the supervisor must NOT restart into the next
        # album after a failed upload or an unlanded commit.
        raise SystemExit(2)
