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
export CUDA_VISIBLE_DEVICES=2

# Define log directory and file (relative to script location)
LOG_DIR="${SCRIPT_DIR}/logs"
SAMPLES_DIR="${SCRIPT_DIR}/samples" # Assuming samples dir is relative like logs
LOG_FILE="${LOG_DIR}/training.log"
VENV_PYTHON="${SCRIPT_DIR}/../venv/bin/python3" # Path to venv python

# Create directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${SAMPLES_DIR}" # Ensure samples dir exists

# Change directory FIRST
cd "${SCRIPT_DIR}" || exit 1 # Exit if cd fails

# Run training for N iterations on GPU in the background
echo "Starting pytorch training (${NUM_EPOCHS} iterations) on GPU ${CUDA_VISIBLE_DEVICES}... Log: $LOG_FILE"

# Define the full command with expanded variables and quoted paths
# Rely on main.py defaults for most parameters
FULL_CMD="'${VENV_PYTHON}' -u 'main.py' --cuda --niter '${NUM_EPOCHS}' --experiment '${SAMPLES_DIR}'"

# Execute with nohup, redirecting nohup's output
nohup bash -c "${FULL_CMD}" > "${LOG_FILE}" 2>&1 &

NOHUP_PID=$!
echo "Training started in background with PID: ${NOHUP_PID}"
echo "Check log: ${LOG_FILE}"

exit 0 # Exit successfully after launching background job

