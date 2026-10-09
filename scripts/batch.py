#!/usr/bin/env python3
"""Resumable batch captioning across ephemeral Colab runtimes.

GitHub is the durable store. For every album we:
  1. download its tracks from GCS (skip files already cached),
  2. caption them,
  3. atomically write outputs/<ARTIST>/<album>.json,
  4. update outputs/index.json,
  5. git add/commit/push immediately.

On a fresh runtime, run the same command: it pulls, reads the manifest, and
skips every album already pushed. Nothing is redone — only the in-flight album
is lost if the runtime dies.

Usage:
    python scripts/batch.py --artist ELISSA --list          # show pending
    python scripts/batch.py --artist ELISSA --max-minutes 90
    python scripts/batch.py                                 # all artists
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.resume import (  # noqa: E402
    album_key,
    album_output_path,
    atomic_write_json,
    group_local_tracks,
    group_tracks,
    is_done,
    load_manifest,
    pending_albums,
)
from src.tags import CATEGORIES  # noqa: E402

DEFAULT_ROOT = "gs://akbar-december-2024-backup/DISCOGRAPHY"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def list_gs_mp3(root: str) -> list[str]:
    """List every .mp3 under ``root`` recursively via the gcloud CLI."""
    pattern = root.rstrip("/") + "/**/*.mp3"
    proc = _run(["gcloud", "storage", "ls", "-r", pattern])
    if proc.returncode != 0:
        raise RuntimeError(f"gcloud storage ls failed: {proc.stderr.strip()}")
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip().endswith(".mp3")]


def download_track(uri: str, dest: Path) -> None:
    """Copy one track to ``dest`` unless it is already cached."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(["gcloud", "storage", "cp", uri, str(dest)])
    if proc.returncode != 0:
        raise RuntimeError(f"gcloud cp failed for {uri}: {proc.stderr.strip()}")


def git_checkpoint(repo: Path, message: str, push: bool, retries: int = 3) -> None:
    """Commit everything and (optionally) push, retrying transient failures."""
    _run(["git", "-C", str(repo), "add", "-A"])
    if _run(["git", "-C", str(repo), "diff", "--cached", "--quiet"]).returncode == 0:
        print("  (no changes to commit)")
        return
    if _run(["git", "-C", str(repo), "commit", "-q", "-m", message]).returncode != 0:
        raise RuntimeError("git commit failed")
    print(f"  committed: {message}")
    if not push:
        return
    for attempt in range(1, retries + 1):
        proc = _run(["git", "-C", str(repo), "push", "origin", "HEAD"])
        if proc.returncode == 0:
            print("  pushed to origin")
            return
        print(f"  push failed (attempt {attempt}/{retries}): {proc.stderr.strip()[:200]}")
        time.sleep(2 * attempt)
    print("  WARNING: could not push; work is committed locally only")


def process_album(captioner, entry: dict, workdir: Path, outputs: Path) -> dict:
    """Download + caption one album, write its JSON, return the album record."""
    from src.captioner import audio_duration
    from src.captioner_hf import caption_and_tag
    from src.tags import merge_tags, to_suno

    artist, album = entry["artist"], entry["album"]
    album_dir = workdir / artist / album
    track_results: list[dict] = []
    total_audio = 0.0
    total_caption = 0.0

    for track in entry["tracks"]:
        filename = track["filename"]
        uri = track["uri"]
        if uri.startswith("gs://"):
            local = album_dir / filename
            download_track(uri, local)
        else:
            local = Path(uri)
        duration = audio_duration(local)
        t0 = time.time()
        result = caption_and_tag(captioner, local)
        caption_s = time.time() - t0
        result.pop("chunk_captions", None)
        total_audio += duration
        total_caption += caption_s
        track_results.append({
            "file": filename,
            "duration_s": round(duration, 1),
            "caption_seconds": round(caption_s, 1),
            "instruments": result["instruments"],
            "vocals": result["vocals"],
            "production": result["production"],
            "caption": result["caption"],
            "suno_prompt": result["suno_prompt"],
        })
        print(f"    {filename}: {caption_s:.1f}s", flush=True)

    merged = merge_tags([
        {cat: t[cat] for cat in CATEGORIES} for t in track_results
    ])
    album_obj = {
        "artist": artist,
        "album": album,
        "source_dir": entry["source_dir"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "track_count": len(track_results),
        "total_audio_seconds": round(total_audio, 1),
        "total_caption_seconds": round(total_caption, 1),
        "instruments": merged["instruments"],
        "vocals": merged["vocals"],
        "production": merged["production"],
        "suno_prompt": to_suno(merged),
        "tracks": track_results,
    }
    atomic_write_json(album_output_path(outputs, artist, album), album_obj)
    return album_obj


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="GCS prefix (gs://...) or a local directory to scan")
    parser.add_argument("--artist", action="append", default=None,
                        help="only this artist (repeatable); default: all")
    parser.add_argument("--workdir", type=Path, default=REPO_ROOT / "work")
    parser.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "outputs" / "index.json")
    parser.add_argument("--hf-model", type=Path, default=REPO_ROOT / "models" / "Ace-Step-Captioner-4bit")
    parser.add_argument("--chunk-seconds", type=float, default=60.0)
    parser.add_argument("--max-minutes", type=float, default=100.0,
                        help="stop cleanly between albums after this long (0 = no limit)")
    parser.add_argument("--max-albums", type=int, default=0, help="cap albums this run (0 = no cap)")
    parser.add_argument("--no-push", action="store_true", help="commit but do not push")
    parser.add_argument("--force", action="store_true", help="redo albums already in the manifest")
    parser.add_argument("--list", action="store_true", help="show pending albums and exit")
    args = parser.parse_args(argv)

    push = not args.no_push

    if push:
        proc = _run(["git", "-C", str(REPO_ROOT), "pull", "--ff-only"])
        if proc.returncode != 0:
            print(f"WARNING: git pull failed: {proc.stderr.strip()[:200]}")

    if args.root.startswith("gs://"):
        albums = group_tracks(list_gs_mp3(args.root), args.root)
    else:
        albums = group_local_tracks(args.root)
    if args.artist:
        wanted = set(args.artist)
        albums = {k: v for k, v in albums.items() if v["artist"] in wanted}

    manifest = load_manifest(args.manifest)
    pending = pending_albums(albums, manifest, args.outputs, force=args.force)
    done = [k for k in albums if k not in pending]

    print(f"Albums found: {len(albums)} | done: {len(done)} | pending: {len(pending)}")
    if args.list:
        for key in pending:
            print(f"  PENDING  {key}  ({len(albums[key]['tracks'])} tracks)")
        for key in done:
            print(f"  done     {key}")
        return 0
    if not pending:
        print("Nothing to do — all albums already captured.")
        return 0

    from src.captioner_hf import HFCaptioner

    captioner = HFCaptioner(model_dir=args.hf_model, chunk_seconds=args.chunk_seconds)
    t0 = time.time()
    captioner._load()
    print(f"MODEL LOAD {time.time() - t0:.1f}s", flush=True)

    started = time.time()
    completed = 0
    durations: list[float] = []
    for i, key in enumerate(pending, 1):
        if args.max_albums and completed >= args.max_albums:
            print(f"Reached --max-albums={args.max_albums}; stopping.")
            break
        elapsed_min = (time.time() - started) / 60.0
        est = (sum(durations) / len(durations)) if durations else 0.0
        if args.max_minutes and durations and elapsed_min + est > args.max_minutes:
            print(f"Time budget {args.max_minutes:.0f} min reached "
                  f"(elapsed {elapsed_min:.1f} min); stopping cleanly.")
            break

        entry = albums[key]
        print(f"\n[{i}/{len(pending)}] {key} ({len(entry['tracks'])} tracks)", flush=True)
        album_t0 = time.time()
        try:
            album_obj = process_album(captioner, entry, args.workdir, args.outputs)
        except Exception as exc:  # noqa: BLE001 - keep going
            print(f"  ERROR on {key}: {type(exc).__name__}: {exc}")
            continue

        manifest["albums"][key] = {
            "path": f"{entry['artist']}/{entry['album']}.json",
            "source_dir": entry["source_dir"],
            "track_count": album_obj["track_count"],
            "total_caption_seconds": album_obj["total_caption_seconds"],
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(args.manifest, manifest)

        album_min = (time.time() - album_t0) / 60.0
        durations.append(album_min)
        completed += 1
        print(f"  done in {album_min:.1f} min "
              f"({album_obj['total_caption_seconds']}s captioning)", flush=True)
        git_checkpoint(
            REPO_ROOT,
            f"captions({key}): {album_obj['track_count']} tracks "
            f"({len(done) + completed}/{len(albums)})",
            push=push,
        )

    remaining = len(pending) - completed
    print(f"\nRun complete: {completed} album(s) this session, ~{remaining} still pending.")
    print(f"Total wall time: {(time.time() - started) / 60.0:.1f} min")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
