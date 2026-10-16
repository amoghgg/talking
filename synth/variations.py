"""Generate a varied LoRA training set of ONE synthetic person from a reference face.

Uses IP-Adapter (plus-face) to lock the identity while prompts vary angle, expression,
lighting, background and outfit, per the shot list.

    python synth/variations.py --ref synth/out/candidates/p2_s0.png --who "a 23 year old woman with long red hair"
"""
import argparse
import random
from pathlib import Path

import torch
from diffusers.utils import load_image

from candidates import NEGATIVE, load_pipe

ANGLES = [
    "front view, looking at the camera",
    "three-quarter view facing left",
    "three-quarter view facing right",
    "side profile",
    "looking slightly up",
    "looking slightly down",
]
EXPRESSIONS = [
    "neutral expression", "gentle smile", "laughing", "talking mid-sentence",
    "surprised", "serious", "eyes closed", "frowning",
]
LIGHTING = [
    "soft window light", "harsh side light", "golden hour sunlight",
    "overcast daylight", "dim indoor lamp light", "fluorescent office light",
]
PLACES = [
    "in a kitchen", "on a city street", "in a park", "against a white wall",
    "in a dark room", "in a cafe", "in an office", "by the sea",
]
OUTFITS = ["black t-shirt", "grey hoodie", "white shirt", "denim jacket", "knitted sweater"]
FRAMING = ["close-up portrait", "close-up portrait", "head and shoulders photo", "half body photo"]

PROMPT = "RAW photo, {framing} of {who}, {angle}, {expr}, wearing a {outfit}, {place}, {light}, natural skin texture, dslr, film grain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="chosen candidate face image")
    ap.add_argument("--who", required=True, help="short description of the person")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--scale", type=float, default=0.7, help="identity strength (0.5-0.9)")
    ap.add_argument("--out", default="synth/out/dataset")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = load_pipe()
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.safetensors")
    pipe.set_ip_adapter_scale(args.scale)
    ref = load_image(args.ref)

    rng = random.Random(7)
    for i in range(args.n):
        # cycle angles/expressions so every one is covered, randomise the rest
        prompt = PROMPT.format(
            framing=rng.choice(FRAMING), who=args.who,
            angle=ANGLES[i % len(ANGLES)], expr=EXPRESSIONS[i % len(EXPRESSIONS)],
            outfit=rng.choice(OUTFITS), place=rng.choice(PLACES), light=rng.choice(LIGHTING),
        )
        image = pipe(
            prompt,
            negative_prompt=NEGATIVE,
            ip_adapter_image=ref,
            width=512,
            height=640,
            num_inference_steps=25,
            guidance_scale=5.5,
            generator=torch.Generator("cpu").manual_seed(i),
        ).images[0]
        path = out / f"{i:03d}.png"
        image.save(path)
        path.with_suffix(".txt").write_text(prompt)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
