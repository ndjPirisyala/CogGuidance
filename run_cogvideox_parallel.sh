#!/usr/bin/env bash
set -euo pipefail

# Change this path if your prompt file is somewhere else
PROMPT_FILE="/data/hunyuan/science_prompts.txt"

NUM_GPUS=8
SEED=42

MODEL_PATH="THUDM/CogVideoX-2B"
OUT_DIR="outputs_cogvideox_science/2B"
LOG_DIR="logs_cogvideox_science/2B"

STEPS=50
GUIDANCE=6.0
NUM_FRAMES=49
FPS=8
DTYPE="float16"

mkdir -p "$OUT_DIR" "$LOG_DIR"

mapfile -t PROMPTS < <(grep -v '^[[:space:]]*$' "$PROMPT_FILE")

echo "Found ${#PROMPTS[@]} prompts"
echo "Running up to ${NUM_GPUS} prompts in parallel"

for i in "${!PROMPTS[@]}"; do
    GPU=$((i % NUM_GPUS))
    IDX=$(printf "%04d" "$i")

    OUT_PATH="${OUT_DIR}/FPS8_49F_cogprompt_${IDX}_seed${SEED}.mp4"
    LOG_PATH="${LOG_DIR}/FPS8_49F_cogprompt_${IDX}_gpu${GPU}.log"

    PROMPT="${PROMPTS[$i]}"

    if [[ -s "$OUT_PATH" ]]; then
        echo "Skipping prompt ${IDX}: output already exists"
        continue
    fi

    echo "Starting prompt ${IDX} on GPU ${GPU}"

    (
        export CUDA_VISIBLE_DEVICES="$GPU"

        python inference/cli_demo.py \
            --prompt "$PROMPT" \
            --model_path "$MODEL_PATH" \
            --generate_type "t2v" \
            --output_path "$OUT_PATH" \
            --num_inference_steps "$STEPS" \
            --guidance_scale "$GUIDANCE" \
            --num_frames "$NUM_FRAMES" \
            --fps "$FPS" \
            --dtype "$DTYPE" \
            --seed "$SEED"
    ) > "$LOG_PATH" 2>&1 &

    # After launching 8 jobs, wait for them before starting the next batch
    if (( (i + 1) % NUM_GPUS == 0 )); then
        wait
        echo "Finished a batch of ${NUM_GPUS}"
    fi

    # Small stagger to avoid all 8 processes loading the model at exactly the same time
    sleep 5
done

wait
echo "All CogVideoX prompts finished."
