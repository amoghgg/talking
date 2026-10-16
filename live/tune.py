"""Grid-test live settings on one fixed webcam frame: which checkpoint / strength / LoRA weight
makes the face read as amoghface while keeping the pose. Writes one labelled contact sheet.

    python live/tune.py webcam_frame.png models/lora/face-000002.safetensors models/lora/face-000004.safetensors
"""
import sys
import time

import torch
from PIL import Image, ImageDraw, ImageFont

from bench import load_pipe

SIZE = 384
# (num_inference_steps, strength): LCM runs steps*strength denoising steps
SETTINGS = [(2, 0.5), (3, 0.67), (4, 0.75), (5, 0.8)]
LORA_WEIGHT = 1.0
PROMPT = "photo of amoghface, a man"


def main():
    src = Image.open(sys.argv[1]).convert("RGB").resize((SIZE, SIZE))
    CHECKPOINTS = sys.argv[2:]
    font = ImageFont.load_default(size=16)
    sheet = Image.new("RGB", (SIZE * (len(SETTINGS) + 1), SIZE * len(CHECKPOINTS)), "black")
    draw = ImageDraw.Draw(sheet)
    for r, ckpt in enumerate(CHECKPOINTS):
        pipe = load_pipe(ckpt)
        pipe.set_adapters(["lcm", "face"], adapter_weights=[1.0, LORA_WEIGHT])
        sheet.paste(src, (0, r * SIZE))
        draw.text((6, r * SIZE + 6), f"webcam | {ckpt.split('-')[-1][:6]}", fill="yellow", font=font)
        for c, (steps, strength) in enumerate(SETTINGS, 1):
            t = time.perf_counter()
            out = pipe(PROMPT, image=src, num_inference_steps=steps, strength=strength, guidance_scale=1.0,
                       generator=torch.Generator("cpu").manual_seed(7)).images[0]
            ms = (time.perf_counter() - t) * 1000
            sheet.paste(out, (c * SIZE, r * SIZE))
            draw.text((c * SIZE + 6, r * SIZE + 6), f"str {strength} ({round(steps * strength)} step) {ms:.0f}ms", fill="yellow", font=font)
        del pipe
    sheet.save("synth/out/tune_sheet.jpg", quality=88)
    print("wrote synth/out/tune_sheet.jpg")


if __name__ == "__main__":
    main()
