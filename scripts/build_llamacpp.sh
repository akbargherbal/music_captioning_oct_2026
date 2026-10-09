#!/usr/bin/env bash
# Idempotent CUDA build of the patched llama.cpp for Google Colab T4 (sm_75).
#
# Stock llama.cpp mishandles this model's audio input, so we build the
# tpsjr7 fork branch documented in AGENTS.md. Re-running is a no-op once the
# binaries exist.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMACPP_DIR="${LLAMACPP_DIR:-$REPO_ROOT/llama.cpp}"
BRANCH="${LLAMACPP_BRANCH:-ted/fix-qwen-audio-cleanup-merge}"
REMOTE="${LLAMACPP_REMOTE:-https://github.com/tpsjr7/llama.cpp}"
BIN="$LLAMACPP_DIR/build/bin/llama-cli"

if [ -x "$BIN" ]; then
  echo "[build] already built: $BIN"
  exit 0
fi

if [ ! -d "$LLAMACPP_DIR/.git" ]; then
  echo "[build] cloning $REMOTE ($BRANCH) -> $LLAMACPP_DIR"
  git clone -b "$BRANCH" "$REMOTE" "$LLAMACPP_DIR"
else
  echo "[build] reusing existing clone at $LLAMACPP_DIR"
fi

echo "[build] configuring"
cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON

echo "[build] compiling (this takes a while on Colab)"
cmake --build "$LLAMACPP_DIR/build" -j"$(nproc)" --target llama-cli

echo "[build] done: $BIN"
