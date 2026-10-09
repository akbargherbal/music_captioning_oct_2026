#!/usr/bin/env python3
"""End-to-end demo: audio -> caption -> Suno-style prompt.

Usage:
    python scripts/demo.py /path/to/song.mp3            # transformers 4-bit (default)
    python scripts/demo.py /path/to/song.mp3 --backend llamacpp

The hf backend needs models/Ace-Step-Captioner-4bit; the llamacpp backend needs
the built llama-cli and the two GGUF files (see README).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODELS_DIR = REPO_ROOT / "models"
DEFAULT_CLI = REPO_ROOT / "llama.cpp" / "build" / "bin" / "llama-cli"
DEFAULT_MODEL = MODELS_DIR / "acestep-captioner-Q4_K_M.gguf"
DEFAULT_MMPROJ = MODELS_DIR / "acestep-captioner-mmproj-Q8_0.gguf"
DEFAULT_HF_MODEL = MODELS_DIR / "Ace-Step-Captioner-4bit"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="path to an audio file")
    parser.add_argument(
        "--backend", choices=("hf", "llamacpp"), default="hf",
        help="hf = transformers 4-bit (default); llamacpp = GGUF build",
    )
    parser.add_argument("--llama-cli", type=Path, default=DEFAULT_CLI)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mmproj", type=Path, default=DEFAULT_MMPROJ)
    parser.add_argument("--hf-model", type=Path, default=DEFAULT_HF_MODEL)
    parser.add_argument("--chunk-seconds", type=float, default=60.0)
    parser.add_argument(
        "--json", action="store_true", help="print only the JSON result"
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=None,
        help="write the JSON result to this path (parent dirs are created)",
    )
    args = parser.parse_args(argv)

    if args.backend == "hf":
        from src.captioner_hf import HFCaptioner, caption_and_tag
        captioner = HFCaptioner(
            model_dir=args.hf_model, chunk_seconds=args.chunk_seconds
        )
    else:
        from src.captioner import Captioner, caption_and_tag
        captioner = Captioner(
            llama_cli=args.llama_cli, model=args.model, mmproj=args.mmproj
        )

    result = caption_and_tag(captioner, args.audio)
    result.pop("chunk_captions", None)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved: {args.out}", file=sys.stderr)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Caption:\n" + str(result["caption"]) + "\n")
        print("Suno prompt:\n" + str(result["suno_prompt"]) + "\n")
        print("Tags:")
        print(json.dumps(
            {k: result[k] for k in ("instruments", "vocals", "production")},
            ensure_ascii=False,
            indent=2,
        ))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
