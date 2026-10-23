#!/usr/bin/env bash
# Convert TouchDesigner takes (recordings/*.mov, ProRes) to small shareable MP4s next to them.
# Skips takes that already have an .mp4.
set -euo pipefail
cd "$(dirname "$0")/recordings"
shopt -s nullglob
for f in *.mov; do
  out="${f%.mov}.mp4"
  [ -f "$out" ] && continue
  echo "converting $f"
  ffmpeg -loglevel error -i "$f" -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p -movflags +faststart -an "$out"
done
mp4s=(*.mp4)
if [ ${#mp4s[@]} -gt 0 ]; then ls -lh "${mp4s[@]}"; else echo "no takes in recordings/ yet"; fi
