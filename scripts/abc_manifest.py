#!/usr/bin/env python3
"""Build the track manifest for the ABC transcription run (no GPU needed).

The manifest is the single source of truth for every later phase: one row per
MP3, keyed by ``track_id`` (never by filename -- 11 titles recur across Fadl
Shaker albums). Three sources are joined:

1. ``gcloud storage ls -l -r``  -- every MP3 URI and its byte size,
2. ``outputs/<ARTIST>/<album>.json`` -- the existing caption rows,
3. ``ffprobe`` -- duration for the 6 tracks the captioner skipped (their
   duration is absent from the caption JSON; only the skip reason remains).

The FADL_SHAKER tree has an extra wrapper directory::

    FADL_SHAKER/<wrapper>/<album>/<file>.mp3

so the album is always the *second-to-last* path component, not the second.
A naive "first directory after the artist" split reports 140 missing tracks.

Usage:
    python3 scripts/abc_manifest.py \
      --artists ELISSA FADL_SHAKER \
      --root gs://akbar-december-2024-backup/DISCOGRAPHY \
      --outputs outputs \
      --out work/manifest.jsonl

Exit code is non-zero when an acceptance check fails, so this can gate a run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
SKIP_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*s\b")
TRACK_NO_RE = re.compile(r"^\s*(\d+)\s*[.\-_ ]*\s*(.*)$")


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def slugify(text: str) -> str:
    """Lowercase, non-alphanumeric -> ``-``, collapse runs, trim.

    ``str.isalnum`` is Unicode-aware, so Arabizi digits and non-Latin letters
    survive; only punctuation/whitespace becomes a separator.
    """
    out: list[str] = []
    for ch in text.strip().lower():
        out.append(ch if ch.isalnum() else "-")
    return re.sub(r"-+", "-", "".join(out)).strip("-")


def split_filename(filename: str) -> tuple[int | None, str]:
    """``"03. Ajmal Ihssas.mp3"`` -> ``(3, "Ajmal Ihssas")``."""
    stem = Path(filename).stem
    m = TRACK_NO_RE.match(stem)
    if not m:
        return None, stem.strip()
    return int(m.group(1)), (m.group(2).strip() or stem.strip())


def normalize_title(title: str) -> str:
    """Title key for duplicate detection.

    ``"(Music)"`` marks an instrumental *version of the same song*, not a new
    title, so it is stripped before grouping. That is what turns the 11 Fadl
    Shaker title groups from 23 rows into the measured 25 (§2).
    """
    base = re.sub(r"\s*[\(\[]\s*music\s*[\)\]]\s*$", "", title.strip(),
                  flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", base).strip().lower()


def list_gs_mp3(root: str, artist: str) -> list[dict[str, object]]:
    """List every MP3 under ``root/<artist>/`` with its byte size."""
    pattern = f"{root.rstrip('/')}/{artist}/**/*.mp3"
    proc = run(["gcloud", "storage", "ls", "-l", "-r", pattern])
    if proc.returncode != 0:
        raise RuntimeError(
            f"gcloud storage ls failed for {pattern}: {proc.stderr.strip()}")
    rows: list[dict[str, object]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or not line.endswith(".mp3"):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            raise RuntimeError(f"unexpected `gcloud storage ls -l` line: {line!r}")
        size_s, _modified, uri = parts
        try:
            size = int(size_s)
        except ValueError as exc:
            raise RuntimeError(f"bad size in listing line: {line!r}") from exc
        rows.append({"uri": uri, "size_bytes": size})
    return rows


def parse_gs_uri(uri: str, root: str) -> dict[str, str]:
    """``gs://…/<artist>/…/<album>/<file>.mp3`` -> its parts.

    ``album`` is the second-to-last component, which transparently strips
    FADL_SHAKER's wrapper directory.
    """
    root = root.rstrip("/")
    if not uri.startswith(root + "/"):
        raise ValueError(f"{uri!r} is not under {root!r}")
    parts = uri[len(root) + 1:].split("/")
    if len(parts) < 3 or Path(parts[-1]).suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"cannot parse track URI: {uri!r}")
    artist = parts[0]
    album = parts[-2]
    filename = parts[-1]
    return {
        "artist": artist,
        "album": album,
        "filename": filename,
        "source_dir": "/".join([root, *parts[:-1]]),
        "source_uri": uri,
    }


def load_captions(outputs: Path, artist: str, album: str) -> dict[str, dict]:
    """Return ``{file: {"status": "ok"|"skipped_long", ...}}`` for one album."""
    path = outputs / artist / f"{album}.json"
    result: dict[str, dict] = {}
    if not path.is_file():
        return result
    data = json.loads(path.read_text(encoding="utf-8"))
    for track in data.get("tracks", []):
        result[track["file"]] = {"status": "ok", "caption_row": track}
    for track in data.get("skipped_tracks", []):
        result[track["file"]] = {"status": "skipped_long", "skipped_row": track}
    return result


def ffprobe_duration(uri: str, cache_dir: Path) -> float | None:
    """Download one object (skip-if-cached) and read its duration with ffprobe."""
    dest = cache_dir / Path(uri).name
    if not dest.is_file() or dest.stat().st_size == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = run(["gcloud", "storage", "cp", uri, str(dest)])
        if proc.returncode != 0:
            raise RuntimeError(
                f"gcloud storage cp failed for {uri}: {proc.stderr.strip()}")
    proc = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(dest),
    ])
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {uri}: {proc.stderr.strip()}")
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def build_manifest(
    artists: list[str],
    root: str,
    outputs: Path,
    probe_dir: Path,
    use_ffprobe: bool = True,
) -> tuple[list[dict], list[str]]:
    """Join the three sources. Returns ``(rows, problems)``."""
    rows: list[dict] = []
    problems: list[str] = []
    for artist in artists:
        listed = list_gs_mp3(root, artist)
        albums: dict[str, list[dict[str, object]]] = {}
        for item in listed:
            info = parse_gs_uri(str(item["uri"]), root)
            albums.setdefault(info["album"], []).append({**info, **item})

        for album in sorted(albums):
            captions = load_captions(outputs, artist, album)
            if not captions:
                problems.append(f"{artist}/{album}: no caption JSON")
            album_rows: list[dict] = []
            for n, item in enumerate(
                    sorted(albums[album], key=lambda r: str(r["filename"])), start=1):
                filename = str(item["filename"])
                cap = captions.get(filename)
                track_no, title = split_filename(filename)
                if track_no is None:
                    track_no = n
                    problems.append(
                        f"{artist}/{album}/{filename}: no leading track number")
                status = cap["status"] if cap else "missing"
                needs_caption = status == "skipped_long"

                duration: float | None = None
                duration_source = "missing"
                if cap and status == "ok":
                    duration = cap["caption_row"].get("duration_s")
                    duration_source = "caption"
                elif cap and status == "skipped_long":
                    duration = cap["skipped_row"].get("duration_s")
                    if duration:
                        duration_source = "caption"
                    else:
                        # The skip reason carries a rounded duration, but §5
                        # says to measure it. Keep it as a fallback only.
                        m = SKIP_DURATION_RE.search(
                            str(cap["skipped_row"].get("reason", "")))
                        if m:
                            duration = float(m.group(1))
                            duration_source = "skip_reason"
                if status == "skipped_long" and use_ffprobe:
                    probed = ffprobe_duration(str(item["uri"]), probe_dir)
                    if probed is not None:
                        duration = probed
                        duration_source = "ffprobe"
                if duration is None and use_ffprobe:
                    duration = ffprobe_duration(str(item["uri"]), probe_dir)
                    duration_source = "ffprobe"
                if duration is None:
                    problems.append(f"{artist}/{album}/{filename}: unknown duration")

                track_id = f"{artist}/{album}/{track_no:02d}-{slugify(title)}"
                album_rows.append({
                    "track_id": track_id,
                    "artist": artist,
                    "album": album,
                    "file": filename,
                    "title": title,
                    "title_normalized": normalize_title(title),
                    "track_no": track_no,
                    "duration_s": round(float(duration), 3) if duration else None,
                    "duration_source": duration_source,
                    "size_bytes": item["size_bytes"],
                    "source_uri": item["uri"],
                    "source_dir": item["source_dir"],
                    "caption_status": status,
                    "needs_caption": needs_caption,
                    # Filled at download time by the batch driver.
                    "audio_sha256": None,
                })
                if cap is None:
                    problems.append(f"{artist}/{album}/{filename}: no caption row")
            rows.extend(album_rows)
    rows.sort(key=lambda r: (r["artist"], r["album"], r["track_no"], r["file"]))
    return rows, problems


def duplicate_titles(rows: list[dict]) -> dict[tuple[str, str], list[str]]:
    """Title groups (artist, normalized title) with more than one row."""
    groups: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (row["artist"], row.get("title_normalized") or normalize_title(row["title"]))
        groups.setdefault(key, []).append(row["track_id"])
    return {k: v for k, v in groups.items() if len(v) > 1}


def summarize(rows: list[dict]) -> dict[str, object]:
    by_artist: dict[str, dict[str, object]] = {}
    for artist in sorted({r["artist"] for r in rows}):
        subset = [r for r in rows if r["artist"] == artist]
        by_artist[artist] = {
            "albums": len({r["album"] for r in subset}),
            "tracks": len(subset),
            "audio_seconds": round(sum(r["duration_s"] or 0 for r in subset), 1),
            "with_caption": sum(r["caption_status"] == "ok" for r in subset),
            "needs_caption": sum(bool(r["needs_caption"]) for r in subset),
            "instrumental_music": sum(
                "(music)" in r["file"].lower() for r in subset),
        }
    dups = duplicate_titles(rows)
    return {
        "rows": len(rows),
        "unique_track_id": len({r["track_id"] for r in rows}),
        "artists": by_artist,
        "albums": sum(int(v["albums"]) for v in by_artist.values()),
        "audio_hours": round(sum(r["duration_s"] or 0 for r in rows) / 3600, 3),
        "caption_ok": sum(r["caption_status"] == "ok" for r in rows),
        "needs_caption": sum(bool(r["needs_caption"]) for r in rows),
        "instrumental_music": sum("(music)" in r["file"].lower() for r in rows),
        "duplicate_title_groups": len(dups),
        "duplicate_title_rows": sum(len(v) for v in dups.values()),
        "missing_caption_rows": sum(r["caption_status"] == "missing" for r in rows),
        "unknown_durations": sum(r["duration_s"] is None for r in rows),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artists", nargs="+", default=["ELISSA", "FADL_SHAKER"])
    ap.add_argument("--root", default="gs://akbar-december-2024-backup/DISCOGRAPHY")
    ap.add_argument("--outputs", type=Path, default=REPO_ROOT / "outputs")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "work" / "manifest.jsonl")
    ap.add_argument("--probe-dir", type=Path,
                    default=REPO_ROOT / "work" / "probe_audio",
                    help="where ffprobe-only downloads are cached")
    ap.add_argument("--no-ffprobe", action="store_true",
                    help="never download; leave unknown durations null")
    args = ap.parse_args(argv)

    rows, problems = build_manifest(
        args.artists, args.root, args.outputs, args.probe_dir,
        use_ffprobe=not args.no_ffprobe)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.out)

    summary = summarize(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)

    # Acceptance checks from plan section 5 (track cwd-independent).
    checks = {
        "236 rows": summary["rows"] == 236,
        "unique track_id": summary["unique_track_id"] == summary["rows"],
        "8 ELISSA / 15 FADL_SHAKER albums": (
            summary["artists"].get("ELISSA", {}).get("albums") == 8
            and summary["artists"].get("FADL_SHAKER", {}).get("albums") == 15),
        "230 captioned / 6 needs_caption": (
            summary["caption_ok"] == 230 and summary["needs_caption"] == 6),
        "19.19 h total": abs(float(summary["audio_hours"]) - 19.19) < 0.02,
        "11 duplicate titles / 25 rows": (
            summary["duplicate_title_groups"] == 11
            and summary["duplicate_title_rows"] == 25),
        "2 instrumental (Music) files": summary["instrumental_music"] == 2,
        "every duration known": summary["unknown_durations"] == 0,
        "no missing caption rows": summary["missing_caption_rows"] == 0,
    }
    print("\nacceptance:")
    for label, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")

    failed = [k for k, ok in checks.items() if not ok] + problems
    if failed:
        print(f"\n{len(failed)} acceptance problem(s); manifest written to "
              f"{args.out} but is NOT trusted", file=sys.stderr)
        return 1
    print(f"\nmanifest: {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
