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

LOG_DIR="${SCRIPT_DIR}/logs"
SAMPLES_DIR="${SCRIPT_DIR}/samples"
LOG_FILE="${LOG_DIR}/training.log"
VENV_ACTIVATE="${SCRIPT_DIR}/../venv/bin/activate"

# Create directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${SAMPLES_DIR}"

# Run the training script in the background, overwriting the log file
echo "Starting StyleGAN training on GPU ${CUDA_VISIBLE_DEVICES} for ${NUM_EPOCHS} epochs..."
echo "Logging output to: ${LOG_FILE}"

# Wrap python command in bash -c to ensure env activation works with nohup
# Overwrite log file using > instead of >>
nohup bash -c "
  if [ -f \"${VENV_ACTIVATE}\" ]; then
      source \"${VENV_ACTIVATE}\"
  else
      echo \"Warning: Virtual environment ${VENV_ACTIVATE} not found.\" >&2
  fi
  cd \"${SCRIPT_DIR}\" # Change to script's directory to run main.py
  # Assuming standard arguments; adjust if main.py differs
  python -u main.py --cuda --niter ${NUM_EPOCHS} --experiment \"${SAMPLES_DIR}\"
" > "${LOG_FILE}" 2>&1 &

NOHUP_PID=$!
echo "Training started in background with PID: ${NOHUP_PID}"
echo "Check log: ${LOG_FILE}"
exit 0 