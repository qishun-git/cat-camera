#!/usr/bin/env bash
set -euo pipefail

# Usage: compress_all_clips.sh [source_dir] [dest_dir] [crf]
SRC_DIR="${1:-data/clips}"
DST_DIR="${2:-data/compressed_clips}"
CRF="${3:-28}"

if [ ! -d "$SRC_DIR" ]; then
  echo "Source directory not found: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DST_DIR"

shopt -s nullglob
for clip in "$SRC_DIR"/*.mp4; do
  base="$(basename "$clip")"
  dst="$DST_DIR/$base"
  echo "Compressing $clip -> $dst (HEVC hvc1 CRF $CRF)"
  ffmpeg -y -i "$clip" -c:v libx265 -tag:v hvc1 -preset ultrafast -crf "$CRF" "$dst"
  rm -f "$clip"
done
echo "Compression complete."
