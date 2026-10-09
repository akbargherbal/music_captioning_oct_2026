#!/usr/bin/env python3
"""Supervisor: keep the resumable batch runner alive across kills/crashes.

``scripts/batch.py`` already isolates per-track and per-album *failures*, but it
cannot survive the whole process being killed from outside Python (a host OOM
kill, a native crash in torch/ffmpeg/CUDA, or the notebook/session reaping the
background job). In those cases nothing inside the process runs, so nothing can
be logged and nothing continues.

This wrapper runs ``scripts/batch.py`` and, whenever it exits non-zero, simply
starts it again. Because the batch commits after every album and skips albums
already done, each restart resumes where the previous one stopped.

Run it detached so it also survives the launching shell::

    setsid nohup python3 -u scripts/supervise.py --log /content/batch.log -- \
        --root "/path/to/dir" --artist-name X >/content/supervisor.out 2>&1 &

Batch arguments go after ``--``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log", type=Path, required=True,
                        help="append every batch run's output here")
    parser.add_argument("--max-restarts", type=int, default=30,
                        help="give up after this many attempts (default 30)")
    parser.add_argument("--min-seconds", type=float, default=180.0,
                        help="a run shorter than this counts as a 'quick' failure")
    parser.add_argument("--max-quick-failures", type=int, default=3,
                        help="abort after this many consecutive quick failures "
                             "(guards against a fast crash loop)")
    parser.add_argument("batch_args", nargs=argparse.REMAINDER,
                        help="arguments for scripts/batch.py (pass after --)")
    args = parser.parse_args(argv)

    batch_args = list(args.batch_args)
    if batch_args and batch_args[0] == "--":
        batch_args = batch_args[1:]
    cmd = [sys.executable, "-u", str(REPO_ROOT / "scripts" / "batch.py"), *batch_args]

    args.log.parent.mkdir(parents=True, exist_ok=True)
    quick = 0
    with args.log.open("a", encoding="utf-8") as log:

        def emit(msg: str) -> None:
            log.write(msg)
            log.flush()
            sys.stdout.write(msg)
            sys.stdout.flush()

        for attempt in range(1, args.max_restarts + 1):
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            emit(f"\n===== [supervisor] attempt {attempt}/{args.max_restarts} "
                 f"at {stamp}\n")
            started = time.time()
            try:
                proc = subprocess.Popen(
                    cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
            except OSError as exc:
                emit(f"[supervisor] could not start batch: {exc}\n")
                return 2

            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            rc = proc.wait()
            elapsed = time.time() - started
            emit(f"===== [supervisor] batch exited rc={rc} after {elapsed:.0f}s\n")

            if rc == 0:
                emit("[supervisor] batch finished cleanly; stopping.\n")
                return 0

            if elapsed < args.min_seconds:
                quick += 1
                emit(f"[supervisor] quick failure {quick}/{args.max_quick_failures}\n")
                if quick >= args.max_quick_failures:
                    emit("[supervisor] too many quick failures; giving up "
                         "(inspect the log above).\n")
                    return rc
            else:
                quick = 0
            time.sleep(5)

        emit("[supervisor] reached --max-restarts; giving up.\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
