"""Build the LoRA training set from raw face photos.

Drops rejected shots, finds the main (largest) face in each photo, crops a square around it
(face + hair + a bit of shoulder) and writes PNGs to dataset/<name>/. Originals are never touched.

    python training/prepare.py path/to/your/photos --out dataset/myface
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

DETECTOR = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"

# Photos to leave out (face paint, sunglasses, other people, app screenshots...): one filename
# per line in training/drop.txt (git-ignored, it names your own photos). '#' starts a comment.
DROP_FILE = Path(__file__).resolve().parent / "drop.txt"
DROP = {line.split("#")[0].strip() for line in DROP_FILE.read_text().splitlines()} - {""} if DROP_FILE.exists() else set()

EXTS = {".jpg", ".jpeg", ".png", ".heic"}
CROP_SCALE = 2.6   # crop side = face height * this
HEAD_LIFT = 0.15   # shift crop up (fraction of face height) to keep the hair


def largest_face(rgb: np.ndarray):
    h, w = rgb.shape[:2]
    scale = 1024 / max(h, w)
    small = cv2.resize(rgb, (int(w * scale), int(h * scale)))
    det = cv2.FaceDetectorYN.create(str(DETECTOR), "", (small.shape[1], small.shape[0]), 0.7)
    _, faces = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if faces is None:
        return None
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4] / scale
    return x, y, fw, fh


def square_crop(img: Image.Image, face, scale=CROP_SCALE):
    x, y, fw, fh = face
    side = min(fh * scale, img.width, img.height)
    cx, cy = x + fw / 2, y + fh / 2 - fh * HEAD_LIFT
    # slide the box back inside the image instead of shrinking it
    left = min(max(cx - side / 2, 0), img.width - side)
    top = min(max(cy - side / 2, 0), img.height - side)
    return img.crop((int(left), int(top), int(left + side), int(top + side)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, default=Path("dataset/myface"))
    ap.add_argument("--size", type=int, default=768)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in args.src.iterdir() if p.suffix.lower() in EXTS)
    kept = 0
    for p in files:
        if p.name in DROP:
            print(f"drop  {p.name}")
            continue
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        face = largest_face(np.asarray(img))
        if face is None:
            print(f"SKIP  {p.name}: no face found")
            continue
        crop = square_crop(img, face).resize((args.size, args.size), Image.LANCZOS)
        kept += 1
        crop.save(args.out / f"{kept:03d}.png")
        print(f"keep  {p.name} -> {kept:03d}.png  (face {int(face[3])}px tall)")
    print(f"\n{kept} images written to {args.out}")


if __name__ == "__main__":
    main()
