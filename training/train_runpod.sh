#!/usr/bin/env bash
# Train a face LoRA on a RunPod GPU pod (CUDA) with kohya-ss/sd-scripts.
# Upload your captioned dataset folder (pngs + .txt captions) to /workspace/<NAME>, plus this
# script, sample_prompts.txt and train.env (or train.example.env), then:
#
#   bash train_runpod.sh
#
# Output: /workspace/output/<NAME>*.safetensors (a checkpoint every SAVE_EVERY epochs, compare them)
set -euo pipefail
cd /workspace

# hyperparameters: your own tuning in train.env (git-ignored), else the generic example
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/train.env" ]; then source "$HERE/train.env"; else source "$HERE/train.example.env"; fi

# community hosts sometimes have a broken CUDA setup; fail fast instead of silently training on CPU
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available on this pod')"

if [ ! -f sd-scripts/.installed ]; then
  [ -d sd-scripts ] || git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git
  # requirements.txt ends with "-e .", so it has to be installed from inside the repo
  (cd sd-scripts && pip install -q -r requirements.txt && pip install -q bitsandbytes)
  touch sd-scripts/.installed
fi

# kohya reads repeats + class from the folder name: <repeats>_<name>
mkdir -p train
rm -rf "train/${REPEATS}_${NAME}"
cp -r "$NAME" "train/${REPEATS}_${NAME}"

cd sd-scripts
accelerate launch --num_cpu_threads_per_process 4 --mixed_precision fp16 train_network.py \
  --pretrained_model_name_or_path "$BASE_MODEL" \
  --train_data_dir /workspace/train \
  --output_dir /workspace/output \
  --output_name "$NAME" \
  --caption_extension .txt \
  --resolution "$RESOLUTION" \
  --enable_bucket \
  --network_module networks.lora \
  --network_dim "$NETWORK_DIM" \
  --network_alpha "$NETWORK_ALPHA" \
  --learning_rate "$UNET_LR" \
  --unet_lr "$UNET_LR" \
  --text_encoder_lr "$TEXT_ENCODER_LR" \
  --optimizer_type AdamW8bit \
  --lr_scheduler cosine \
  --lr_warmup_steps "$WARMUP_STEPS" \
  --train_batch_size "$BATCH_SIZE" \
  --max_train_epochs "$EPOCHS" \
  --save_every_n_epochs "$SAVE_EVERY" \
  --save_model_as safetensors \
  --mixed_precision fp16 \
  --save_precision fp16 \
  --sdpa \
  --cache_latents \
  --gradient_checkpointing \
  --seed 42 \
  --sample_prompts /workspace/sample_prompts.txt \
  --sample_every_n_epochs "$SAVE_EVERY" \
  --sample_sampler dpmsolver++
