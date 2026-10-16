# talking

A live mirror that gives you my face.

You stand in front of a camera, and the screen shows the room, your clothes, your hands, your movements, but your face slowly turns into mine. It isn't a filter pasted on top. Every frame is repainted by an image model that has learned one face, so the face drifts, morphs and glitches as you move. It runs live on a MacBook.

It's about what it feels like to have your face borrowed, and to borrow someone else's body.

## Credit where it's due

This project is a learning re-build of **[Surrogate Bodies](https://youtu.be/HaBnUxYS8Xo) by Aisling Phelan**, an interactive installation where visitors' bodies are overlaid with the artist's AI-generated face, presented in her talk *Integrating AI and TouchDesigner into Interactive Installations* at the TouchDesigner Event Dublin, 28 June 2025. The concept, the pipeline (training images, then a face LoRA, then real-time Stable Diffusion in TouchDesigner fed by a webcam) and the "everyone wears my face" idea are hers. Go watch the talk.

I'm not affiliated with her or with Derivative; this is my own reconstruction from the talk, with my own face, adapted to run on a Mac instead of an NVIDIA PC.

Built on: [TouchDesigner](https://derivative.ca) · [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) · [Realistic Vision](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE) · [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) · [TAESD](https://github.com/madebyollin/taesd) · [diffusers](https://github.com/huggingface/diffusers) · [syphon-python](https://github.com/cansik/syphon-python) · [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) · [RunPod](https://runpod.io) for training.

## How it works

```
 camera ─► TouchDesigner ──(Syphon)──► Python: Stable Diffusion img2img ──(Syphon)──► TouchDesigner ─► screen
                                       + LCM-LoRA (1–2 steps) + my face LoRA
```

1. **Photos.** About 50 ordinary selfies of me: different places, light, clothes. Junk shots dropped, faces auto-cropped (`training/prepare.py`).
2. **Captions.** Each photo gets a trigger word (`amoghface`) plus whatever changes between photos (clothes, room, light). Things that make me *me*, like the glasses and curls, are deliberately left out of captions so the trigger word absorbs them.
3. **Face LoRA.** A small add-on file that teaches Stable Diffusion 1.5 one face, trained in about 15 minutes on a rented GPU (`training/train_runpod.sh`).
4. **Live loop.** TouchDesigner sends the mirrored webcam over Syphon. `live/bridge.py` repaints each frame *gently*: enough for the face to become mine, not so much that the room, hands and pose are lost. LCM-LoRA makes that take 1–2 diffusion steps instead of 25. The result goes back to TouchDesigner over Syphon.

Two versions ship:

| | | |
|---|---|---|
| **detailed** | `./run_detailed.sh` | slower, more accurate face |
| **smooth**   | `./run_smooth.sh`   | faster, softer face |

## Run it yourself

You need a Mac with Apple Silicon, TouchDesigner (the free licence works) and Python 3.13.

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# 1. dataset: your own photos -> square face crops (+ YuNet detector from opencv_zoo)
mkdir -p models && curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
./.venv/bin/python training/prepare.py path/to/your/photos --out dataset/myface
#    then write one .txt caption per image: "photo of myface, a person, <clothes, place, light>"

# 2. train on any CUDA GPU (see training/train_runpod.sh), put the .safetensors in models/lora/

# 3. live: open td/talking.toe in TouchDesigner, then
./run_smooth.sh --lora models/lora/myface.safetensors --prompt "photo of myface, a person"
```

The live presets in `live/presets.example.json` and the training settings in `training/train.example.env` are **generic starting points, not the values used in my piece**. Copy them to `presets.json` / `train.env` and tune them: `live/tune.py` and `live/bench.py` exist exactly for that.

Not included: my photos, my trained face model, or my exact settings.

## Please

- **Only train on your own face, or on someone who has said yes.** A face model of someone without their consent is exactly the thing the original work is critiquing.
- If you show it in public, tell people what the camera does. Nothing is recorded or stored by this code; frames only exist in memory.

## License

Code: MIT. The concept belongs to Aisling Phelan's *Surrogate Bodies*; credit her if you build on it.
