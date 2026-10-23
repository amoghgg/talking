"""Pull training stills out of long webcam videos (e.g. recordings/raw_*.mov from TouchDesigner).

Looks at a few frames per second, keeps only ones with a big, sharp face, takes the sharpest
frame of each short window, then drops near-duplicates so every still is a different look
(clothes, expression, angle, light). Writes full frames.

    python training/frames_from_video.py recordings/raw_*.mov --out frames/video1

For the room version, caption these full 16:9 frames as they are and train with
RESOLUTION=768,448. For a face-only model, crop them first: python training/prepare.py frames/video1 --out dataset/myface
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from prepare import largest_face

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

SCAN_FPS = 3          # frames looked at per second of video
WINDOW_S = 2.0        # keep at most one (the sharpest) frame per this many seconds
MIN_SHARPNESS = 60    # Laplacian variance of the face; below this is motion blur


def face_stats(frame_bgr, face):
    x, y, w, h = (int(v) for v in face)
    roi = frame_bgr[max(y, 0):y + h, max(x, 0):x + w]
    if roi.size == 0:
        return 0.0, None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # measure at native size (or smaller): upscaling a small face would make it look blurry
    side = min(128, gray.shape[0], gray.shape[1])
    sharp = cv2.Laplacian(cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA), cv2.CV_64F).var()
    # look signature: the face (expression/angle) plus the body below it (clothes) + colours
    H, W = frame_bgr.shape[:2]
    body = frame_bgr[max(y, 0):min(y + 4 * h, H), max(x - w, 0):min(x + 2 * w, W)]
    sig = np.concatenate([
        cv2.resize(gray, (16, 16)).ravel() / 255.0,
        cv2.resize(body, (8, 8)).ravel() / 255.0,
    ])
    return sharp, sig


def scan(path, min_face):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(int(round(fps / SCAN_FPS)), 1)
    best = {}  # window index -> (sharpness, time, frame, sig, face height)
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            _, frame = cap.retrieve()
            face = largest_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if face is not None and face[3] >= min_face:
                sharp, sig = face_stats(frame, face)
                t = i / fps
                win = int(t // WINDOW_S)
                if sharp >= MIN_SHARPNESS and sharp > best.get(win, (0,))[0]:
                    best[win] = (sharp, t, frame, sig, face[3])
        i += 1
    cap.release()
    return [v for _, v in sorted(best.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=60, help="max stills to keep")
    ap.add_argument("--min-face", type=int, default=160, help="min face height in px (bigger = sharper training crops)")
    ap.add_argument("--min-diff", type=float, default=0.12, help="how different two stills must look to both be kept")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cands = []
    for v in args.videos:
        found = scan(v, args.min_face)
        print(f"{v.name}: {len(found)} sharp, big-face moments")
        cands += [(v.stem, *c) for c in found]

    # greedy: sharpest first, skip anything that looks like a still we already kept
    kept = []
    for c in sorted(cands, key=lambda c: -c[1]):
        if all(np.abs(c[4] - k[4]).mean() >= args.min_diff for k in kept):
            kept.append(c)
        if len(kept) == args.count:
            break

    for n, (stem, sharp, t, frame, _, fh) in enumerate(sorted(kept, key=lambda c: (c[0], c[2])), 1):
        name = f"{stem}_{int(t // 60):02d}m{t % 60:04.1f}s.png"
        cv2.imwrite(str(args.out / name), frame)
        print(f"{n:3d}  {name}  face {int(fh)}px  sharpness {sharp:.0f}")
    print(f"\n{len(kept)} stills written to {args.out} (from {len(cands)} candidates)")


if __name__ == "__main__":
    main()
