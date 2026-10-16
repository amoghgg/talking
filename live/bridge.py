"""Live face bridge: TouchDesigner webcam -> Stable Diffusion (LCM) -> back into TouchDesigner.

TouchDesigner publishes the camera as Syphon server "td_cam" (Syphon Spout Out TOP);
this script repaints each frame with the face and publishes it as Syphon server "amoghface",
which TD picks up with a Syphon Spout In TOP.

    python live/bridge.py --preset detailed   # slower, more accurate face
    python live/bridge.py --preset smooth     # faster, softer
Presets come from live/presets.json (your own tuning, git-ignored), falling back to
live/presets.example.json. Any flag passed explicitly (e.g. --strength 0.45) overrides the preset.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import Metal
import torch
from PIL import Image
from syphon import SyphonMetalClient, SyphonMetalServer, SyphonServerDirectory
from syphon.utils.numpy import copy_image_to_mtl_texture, copy_mtl_texture_to_image
from syphon.utils.raw import create_mtl_texture

from bench import load_pipe

IN_NAME, OUT_NAME = "td_cam", "amoghface"
DEFAULT_LORA = "models/lora/amoghface-000004.safetensors"

_here = Path(__file__).resolve().parent
_presets_file = _here / "presets.json" if (_here / "presets.json").exists() else _here / "presets.example.json"
PRESETS = {k: v for k, v in json.loads(_presets_file.read_text()).items() if not k.startswith("_")}


def find_td_server(directory):
    print(f"waiting for Syphon server '{IN_NAME}' from TouchDesigner...", flush=True)
    while True:
        for s in directory.servers:
            if s.name == IN_NAME:
                print(f"found '{s.name}' ({s.app_name})", flush=True)
                return s
        time.sleep(0.5)


def to_rgb(rgba: np.ndarray, bgra: bool, width: int, height: int) -> Image.Image:
    """Center-crop the incoming frame to the output aspect ratio, then resize."""
    rgb = rgba[..., [2, 1, 0]] if bgra else rgba[..., :3]
    h, w = rgb.shape[:2]
    cw, ch = min(w, round(h * width / height)), min(h, round(w * height / width))
    top, left = (h - ch) // 2, (w - cw) // 2
    return Image.fromarray(np.ascontiguousarray(rgb[top:top + ch, left:left + cw])).resize((width, height))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="detailed")
    ap.add_argument("--lora", default=DEFAULT_LORA, help="face LoRA .safetensors ('' for the base model)")
    ap.add_argument("--prompt", default="photo of amoghface, a man")
    # SD needs both sides divisible by 64. Gentle repaint like the reference: keeps
    # clothes/hands/room, only the face identity drifts.
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--strength", type=float, help="how much of each frame gets repainted")
    ap.add_argument("--lora-weight", type=float)
    ap.add_argument("--seed", type=int, default=7, help="fixed noise each frame = less flicker")
    ap.add_argument("--dump", action="store_true", help="save the 5th frame's SD input/output for checking")
    args = ap.parse_args()
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    print(f"preset '{args.preset}': {args.width}x{args.height}, steps {args.steps}, "
          f"strength {args.strength}, lora {args.lora_weight}", flush=True)

    pipe = load_pipe(args.lora)
    if args.lora:
        pipe.set_adapters(["lcm", "face"], adapter_weights=[1.0, args.lora_weight])
    directory = SyphonServerDirectory()
    client = SyphonMetalClient(find_td_server(directory))
    server = SyphonMetalServer(OUT_NAME)
    device = Metal.MTLCreateSystemDefaultDevice()
    out_tex = create_mtl_texture(device, args.width, args.height)
    print(f"publishing Syphon server '{OUT_NAME}' at {args.width}x{args.height}", flush=True)

    frames, t0 = 0, time.perf_counter()
    stage = {"wait": 0.0, "grab": 0.0, "diffuse": 0.0, "publish": 0.0}
    while True:
        t = time.perf_counter()
        while not client.has_new_frame:
            directory.update_run_loop()
        t_wait = time.perf_counter()
        tex = client.new_frame_image
        bgra = tex.pixelFormat() == Metal.MTLPixelFormatBGRA8Unorm
        # TD's Syphon frames arrive bottom-up (OpenGL origin): flip so SD sees an upright face
        frame = to_rgb(copy_mtl_texture_to_image(tex)[::-1], bgra, args.width, args.height)
        t_grab = time.perf_counter()

        out = pipe(args.prompt, image=frame, num_inference_steps=args.steps, strength=args.strength,
                   guidance_scale=1.0, generator=torch.Generator("cpu").manual_seed(args.seed)).images[0]
        t_diffuse = time.perf_counter()

        # publish upright (checked with --dump: TD displays the array as-is)
        rgba = np.dstack([np.asarray(out), np.full((args.height, args.width), 255, np.uint8)])
        copy_image_to_mtl_texture(np.ascontiguousarray(rgba), out_tex)
        server.publish_frame_texture(out_tex, is_flipped=True)
        t_pub = time.perf_counter()
        if args.dump and frames == 5:
            frame.save("synth/out/dbg_sd_input.png")
            Image.fromarray(rgba[..., :3]).save("synth/out/dbg_published_array.png")

        for k, a, b in (("wait", t, t_wait), ("grab", t_wait, t_grab), ("diffuse", t_grab, t_diffuse), ("publish", t_diffuse, t_pub)):
            stage[k] += b - a
        frames += 1
        if frames % 10 == 0:
            ms = "  ".join(f"{k} {v / frames * 1000:.0f}ms" for k, v in stage.items())
            print(f"{frames / (time.perf_counter() - t0):.2f} fps  |  {ms}", flush=True)


if __name__ == "__main__":
    main()
