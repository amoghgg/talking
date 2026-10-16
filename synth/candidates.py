"""Generate candidate faces for a synthetic (non-existent) person.

Run, look through synth/out/candidates/, pick the face you like, then feed it to variations.py.
"""
import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionPipeline

BASE_MODEL = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
VAE = "stabilityai/sd-vae-ft-mse"

PEOPLE = [
    "a 27 year old woman with short dark hair and freckles",
    "a 34 year old man with a light beard and messy brown hair",
    "a 23 year old woman with long red hair and pale skin",
    "a 30 year old man with a shaved head and dark skin",
    "a 41 year old woman with grey streaked hair and sharp cheekbones",
    "a 25 year old man with curly black hair and a thin moustache",
    "a 29 year old woman with a blonde buzzcut and strong eyebrows",
    "a 38 year old man with tousled blond hair and stubble",
]

PROMPT = (
    "RAW photo, close-up portrait of {who}, looking at the camera, neutral expression, "
    "plain grey background, soft window light, natural skin texture, pores, "
    "8k uhd, dslr, film grain, Fujifilm XT3"
)
NEGATIVE = (
    "cartoon, painting, illustration, 3d render, cgi, anime, deformed, disfigured, "
    "extra fingers, blurry, airbrushed, plastic skin, makeup, text, watermark"
)


def load_pipe():
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, vae=vae, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    return pipe.to("mps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2, help="seeds per person description")
    ap.add_argument("--out", default="synth/out/candidates")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pipe = load_pipe()

    for i, who in enumerate(PEOPLE):
        for seed in range(args.seeds):
            image = pipe(
                PROMPT.format(who=who),
                negative_prompt=NEGATIVE,
                width=512,
                height=640,
                num_inference_steps=25,
                guidance_scale=5.5,
                generator=torch.Generator("cpu").manual_seed(1000 * i + seed),
            ).images[0]
            path = out / f"p{i}_s{seed}.png"
            image.save(path)
            print(f"saved {path}  ({who})", flush=True)


if __name__ == "__main__":
    main()
