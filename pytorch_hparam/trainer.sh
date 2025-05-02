#!/bin/bash

# Default number of epochs if not provided
DEFAULT_EPOCHS=5000 # Or whatever default you want
NUM_EPOCHS=${1:-$DEFAULT_EPOCHS} # Use argument $1 if present, otherwise use default

# Optional: Add validation for the input argument
if ! [[ "${NUM_EPOCHS}" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid number of epochs provided: '$1'. Please provide an integer." >&2
    exit 1
fi

# Set the CUDA device
export CUDA_VISIBLE_DEVICES=2

LOG_DIR="logs"
SAMPLES_DIR="samples"
LOG_FILE="${LOG_DIR}/training.log" # Fixed log file name (Overwrite)

# Create directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${SAMPLES_DIR}"

# Change directory FIRST
cd "$(dirname "$0")" || exit 1

# Run the training script in the background, overwriting the log file
echo "Starting HPARAM training on GPU ${CUDA_VISIBLE_DEVICES} for ${NUM_EPOCHS} epochs..."
echo "Logging output to: ${LOG_FILE}"

# Define the full command with expanded variables and quoted paths
FULL_CMD="'../venv/bin/python3' -u 'main.py' --cuda --niter '${NUM_EPOCHS}' --n_extra_layers 0 --ngf 64 --ndf 64 --batchSize 32 --experiment '${SAMPLES_DIR}'"

# Execute with nohup, redirecting nohup's output
# Use double quotes around ${FULL_CMD} to allow variable expansion here
nohup bash -c "${FULL_CMD}" > "${LOG_FILE}" 2>&1 &

NOHUP_PID=$!
echo "Training started in background with PID: ${NOHUP_PID}"
echo "Check log: ${LOG_FILE}"
exit 0 