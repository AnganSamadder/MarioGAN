#!/bin/bash

# Default number of epochs if not provided
DEFAULT_EPOCHS=5000
NUM_EPOCHS=${1:-$DEFAULT_EPOCHS} # Use argument $1 if present, otherwise use default

# Optional: Add validation for the input argument
if ! [[ "$NUM_EPOCHS" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid number of epochs provided: '$1'. Please provide an integer." >&2
    exit 1
fi

# Get the directory where the script resides
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set the CUDA device
export CUDA_VISIBLE_DEVICES=3

LOG_DIR="${SCRIPT_DIR}/logs"
SAMPLES_DIR="${SCRIPT_DIR}/samples"
LOG_FILE="${LOG_DIR}/training.log"
VENV_PYTHON="${SCRIPT_DIR}/../venv/bin/python3" # Path to venv python

# Create directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${SAMPLES_DIR}"

# Change directory FIRST
cd "${SCRIPT_DIR}" || exit 1 # Exit if cd fails

# Run the training script in the background, overwriting the log file
echo "Starting LOSS TERM training on GPU ${CUDA_VISIBLE_DEVICES} for ${NUM_EPOCHS} epochs..."
echo "Logging output to: ${LOG_FILE}"

# Define the full command with expanded variables and quoted paths
# Use original core params, add new loss args
FULL_CMD="'${VENV_PYTHON}' -u 'main.py' \\
  --cuda \\
  --niter '${NUM_EPOCHS}' \\
  --n_extra_layers 0 \\
  --ngf 64 \\
  --ndf 64 \\
  --batchSize 32 \\
  --diversity_weight 0 \\
  --floating_enemy_weight 0 \\
  --base_pipe_penalty_weight 0.1 \\
  --penalty_start_epoch 500 \\
  --pipe_presence_threshold 0 \\
  --experiment '${SAMPLES_DIR}'"

# Execute with nohup, redirecting nohup's output
nohup bash -c "${FULL_CMD}" > "${LOG_FILE}" 2>&1 &

NOHUP_PID=$!
echo "Training started in background with PID: ${NOHUP_PID}"
echo "Check log: ${LOG_FILE}"
exit 0 