#!/bin/bash

# Set the CUDA device
export CUDA_VISIBLE_DEVICES=1

LOG_DIR="logs"
SAMPLES_DIR="samples"
LOG_FILE="${LOG_DIR}/training.log" # Fixed log file name (Overwrite)

# Create directories if they don't exist
mkdir -p "${LOG_DIR}"
mkdir -p "${SAMPLES_DIR}"

# Run the training script in the background, overwriting the log file
echo "Starting LOSS TERM training on GPU ${CUDA_VISIBLE_DEVICES} for 500 epochs..."
echo "Logging output to: ${LOG_FILE}"

# Wrap python command in bash -c to ensure env activation works with nohup
# Overwrite log file using > instead of >>
nohup bash -c '
  if [ -f "../venv/bin/activate" ]; then
      source ../venv/bin/activate
  else
      echo "Warning: Virtual environment not found." >&2
  fi
  stdbuf -oL -eL python main.py --cuda --niter 500 --penalty_weight 1.0 --experiment "${SAMPLES_DIR}"
' > "${LOG_FILE}" 2>&1 &

NOHUP_PID=$!
echo "Training started in background with PID: ${NOHUP_PID}"
echo "Check log: ${LOG_FILE}"
exit 0
