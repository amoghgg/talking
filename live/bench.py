"""Measure how fast this Mac can do real-time img2img (the live-overlay step), no webcam needed.

Same recipe StreamDiffusion uses: SD1.5 + LCM-LoRA (1-4 steps, no CFG) + TAESD tiny decoder.

    python live/bench.py --image some_face.png [--lora path/to/your_face.safetensors]
"""
import argparse
import time

import torch
from diffusers import AutoencoderTiny, LCMScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image

BASE_MODEL = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"
TAESD = "madebyollin/taesd"


def load_pipe(lora=None):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.vae = AutoencoderTiny.from_pretrained(TAESD, torch_dtype=torch.float16)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LCM_LORA, adapter_name="lcm")
    if lora:
        pipe.load_lora_weights(lora, adapter_name="face")
        pipe.set_adapters(["lcm", "face"], adapter_weights=[1.0, 0.9])
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("mps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="any photo with a face, used as a stand-in webcam frame")
    ap.add_argument("--lora")
    ap.add_argument("--prompt", default="photo of a man with curly hair and glasses")
    args = ap.parse_args()

    pipe = load_pipe(args.lora)
    src = Image.open(args.image).convert("RGB")

    # (size, steps, strength): steps*strength = denoising steps actually run
    for size, steps, strength in [(512, 2, 0.5), (512, 4, 0.5), (384, 2, 0.5), (384, 4, 0.5), (256, 2, 0.5)]:
        frame = src.resize((size, size))
        run = lambda: pipe(args.prompt, image=frame, num_inference_steps=steps, strength=strength,
                           guidance_scale=1.0, output_type="pil").images[0]
        run()  # warm-up (kernel compile)
        n, t0 = 8, time.perf_counter()
        for _ in range(n):
            out = run()
        dt = (time.perf_counter() - t0) / n
        out.save(f"synth/out/bench_{size}_{steps}.png")
        print(f"{size}px  steps={steps} strength={strength}:  {1/dt:5.2f} fps  ({dt*1000:.0f} ms/frame)", flush=True)


if __name__ == "__main__":
    main()
