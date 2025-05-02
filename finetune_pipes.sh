#!/bin/bash

# --- Script for Fine-tuning a Pre-trained GAN to Fix Pipes ---

# --- Default values ---
USE_NOHUP=true

# --- Argument Parsing ---
# Need to parse the optional flag first
POS_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --foreground)
      USE_NOHUP=false
      shift # past argument
      ;;
    *)
      POS_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

# Restore positional arguments
set -- "${POS_ARGS[@]}"

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 [--foreground] <path_to_input_netG.pth> <path_to_input_netD.pth> <num_epochs>"
    echo "  --foreground : Run the training process in the foreground instead of using nohup."
    exit 1
fi

INPUT_NETG_PATH="$1"
INPUT_NETD_PATH="$2"
NUM_EPOCHS="$3"

# --- Resolve Paths --- 
# Convert input paths to absolute paths BEFORE changing directory
# This handles cases where relative paths are given
INPUT_NETG_PATH=$(realpath "$INPUT_NETG_PATH")
INPUT_NETD_PATH=$(realpath "$INPUT_NETD_PATH")

# Validate input files exist *after* resolving paths
if [ ! -f "$INPUT_NETG_PATH" ]; then
    echo "Error: Resolved Input Generator file not found: $INPUT_NETG_PATH" >&2
    exit 1
fi
if [ ! -f "$INPUT_NETD_PATH" ]; then
    echo "Error: Resolved Input Discriminator file not found: $INPUT_NETD_PATH" >&2
    exit 1
fi

# Validate epochs
if ! [[ "$NUM_EPOCHS" =~ ^[0-9]+$ ]] || [ "$NUM_EPOCHS" -le 0 ]; then
    echo "Error: Invalid number of epochs provided: '$NUM_EPOCHS'. Please provide a positive integer." >&2
    exit 1
fi

# --- Configuration ---
# Get the directory where this fine-tuning script resides
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Path to the main training script with loss terms
MAIN_SCRIPT_DIR="${SCRIPT_DIR}/pytorch_loss_term"
MAIN_SCRIPT_PATH="${MAIN_SCRIPT_DIR}/main.py"
# Path to the virtual environment python
VENV_PYTHON="${SCRIPT_DIR}/venv/bin/python3" # Adjust if your venv is elsewhere

# Extract a base name for descriptive output directories
INPUT_NETG_BASENAME=$(basename "$INPUT_NETG_PATH" .pth)
FINETUNE_ID="finetune_pipes_${INPUT_NETG_BASENAME}"

# Define output directories relative to this script's location
LOG_DIR="${SCRIPT_DIR}/logs_${FINETUNE_ID}"
SAMPLES_DIR="${SCRIPT_DIR}/samples_${FINETUNE_ID}"
LOG_FILE="${LOG_DIR}/finetuning.log"

# Set CUDA device (adjust if needed)
export CUDA_VISIBLE_DEVICES=1 # Or make this an argument

# --- Directory Setup ---
mkdir -p "${LOG_DIR}"
mkdir -p "${SAMPLES_DIR}"

# --- Build Command ---
# Ensure paths with spaces are handled correctly using quotes
# Match core parameters to the model being loaded (original settings)
# Apply pipe penalty from start with desired weight, disable others.
FULL_CMD="'${VENV_PYTHON}' -u '${MAIN_SCRIPT_PATH##*/}' \
  --netG '${INPUT_NETG_PATH}' \
  --netD '${INPUT_NETD_PATH}' \
  --cuda \
  --niter '${NUM_EPOCHS}' \
  --n_extra_layers 0 \
  --ngf 64 \
  --ndf 64 \
  --batchSize 32 \
  --diversity_weight 0 \
  --floating_enemy_weight 0 \
  --base_pipe_penalty_weight 0.1 \
  --penalty_start_epoch 0 \
  --pipe_presence_threshold 0.05 \
  --experiment '${SAMPLES_DIR}'" # main.py will handle relative/absolute path for experiment

# --- Execution ---
# Change to the main script's directory FIRST
cd "${MAIN_SCRIPT_DIR}" || { echo "Error: Could not change directory to ${MAIN_SCRIPT_DIR}" >&2; exit 1; }

echo "Starting Fine-tuning using ${MAIN_SCRIPT_PATH##*/}"
echo "  Input NetG: ${INPUT_NETG_PATH}"
echo "  Input NetD: ${INPUT_NETD_PATH}"
echo "  Epochs: ${NUM_EPOCHS}"
echo "  Output Samples/Models: ${SAMPLES_DIR}"
# Only print log path if actually logging to file (background mode)
if [ "$USE_NOHUP" = true ] ; then
  echo "  Output Log: ${LOG_FILE}"
fi
echo "  Running on GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Running in background flag: ${USE_NOHUP}"
# echo "Command: ${FULL_CMD}" # Keep this commented out or remove as direct execution is used now

# Debugging the flag
echo "Debug: USE_NOHUP is currently set to '${USE_NOHUP}'"

# Execute with nohup or directly
if [ "$USE_NOHUP" = true ] ; then
  echo "Running with nohup in the background..."
  # Execute with nohup, redirecting nohup's output
  nohup "${VENV_PYTHON}" -u "${MAIN_SCRIPT_PATH##*/}" \
    --netG "${INPUT_NETG_PATH}" \
    --netD "${INPUT_NETD_PATH}" \
    --cuda \
    --niter "${NUM_EPOCHS}" \
    --lrG 0.000005 \
    --n_extra_layers 0 \
    --ngf 64 \
    --ndf 64 \
    --batchSize 32 \
    --diversity_weight 0 \
    --floating_enemy_weight 0 \
    --base_pipe_penalty_weight 0.01 \
    --penalty_start_epoch 0 \
    --pipe_presence_threshold 0.05 \
    --experiment "${SAMPLES_DIR}" > "${LOG_FILE}" 2>&1 &

  NOHUP_PID=$!
  echo "Fine-tuning started in background with PID: ${NOHUP_PID}"
  echo "Check log: ${LOG_FILE}"
else
  echo "Running in the foreground..."
  # Execute directly, redirecting output TO TERMINAL
  "${VENV_PYTHON}" -u "${MAIN_SCRIPT_PATH##*/}" \
    --netG "${INPUT_NETG_PATH}" \
    --netD "${INPUT_NETD_PATH}" \
    --cuda \
    --niter "${NUM_EPOCHS}" \
    --lrG 0.000005 \
    --n_extra_layers 0 \
    --ngf 64 \
    --ndf 64 \
    --batchSize 32 \
    --diversity_weight 0 \
    --floating_enemy_weight 0 \
    --base_pipe_penalty_weight 0.01 \
    --penalty_start_epoch 0 \
    --pipe_presence_threshold 0.05 \
    --experiment "${SAMPLES_DIR}" # Removed > "${LOG_FILE}" 2>&1

  EXIT_CODE=$?
  echo "Fine-tuning finished with exit code: ${EXIT_CODE}"
  # echo "Check log: ${LOG_FILE}" # Comment out log check message for foreground
fi

# Go back to the original directory (optional)
cd "${SCRIPT_DIR}" || exit 1

exit 0 