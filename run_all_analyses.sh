#!/bin/bash

# Script to run analysis for multiple checkpoints in the background.

# --- Get Number of Levels ---
NUM_LEVELS=${1:-100} # Use first argument or default to 100
# Basic validation to ensure it's a positive integer
if ! [[ "$NUM_LEVELS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of levels must be a positive integer." >&2
    echo "Usage: $0 [num_levels]" >&2
    exit 1
fi
echo "Running analysis for $NUM_LEVELS levels per checkpoint."
# ---------------------------------

# Define directories
CHECKPOINT_DIR="analysis_checkpoints"
LOG_DIR="analysis/logs"
ANALYSIS_SCRIPT="analysis/run_analysis.py"
VENV_ACTIVATE="venv/bin/activate"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Check for virtual environment
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Error: Virtual environment not found at $VENV_ACTIVATE" >&2
    exit 1
fi
source "$VENV_ACTIVATE"
echo "Virtual environment activated."

# Define Checkpoints and their corresponding Generator Scripts
declare -A CHECKPOINTS
declare -A GENERATORS

CHECKPOINTS["pytorch"]="pytorch_iter5000.pth"
GENERATORS["pytorch"]="pytorch/generator_ws.py"

CHECKPOINTS["hparam"]="hparam_iter5000.pth"
GENERATORS["hparam"]="pytorch_hparam/generator_ws.py" # Use the newly created script

CHECKPOINTS["sagan"]="sagan_iter5000.pth"
GENERATORS["sagan"]="pytorch_sagan/generator_ws.py"

CHECKPOINTS["loss_term"]="loss_term_iter5000.pth"
GENERATORS["loss_term"]="pytorch_loss_term/generator_ws.py" # Use the newly created script

CHECKPOINTS["stylegan"]="stylegan_iter5000.pth"
GENERATORS["stylegan"]="pytorch_stylegan/generator_ws.py" # Use the newly created script

# --- Run Analysis for each Checkpoint ---
PIDS=()
echo "Starting analysis runs..."
for key in "${!CHECKPOINTS[@]}"; do
    checkpoint_file="${CHECKPOINTS[$key]}"
    checkpoint_path="$CHECKPOINT_DIR/$checkpoint_file"
    generator_script="${GENERATORS[$key]}" # Get the corresponding generator script
    log_file="$LOG_DIR/analysis_run_${key}_${NUM_LEVELS}lvls.log"

    if [ ! -f "$checkpoint_path" ]; then
        echo "Warning: Checkpoint file not found for $key: $checkpoint_path. Skipping."
        continue
    fi
    if [ ! -f "$generator_script" ]; then
        echo "Warning: Generator script not found for $key: $generator_script. Skipping."
        continue
    fi
    
    echo "Starting analysis for $key checkpoint ($checkpoint_path) using generator ($generator_script)... Log: $log_file"
    
    # Run analysis in background using nohup, passing the specific generator script
    # Clear previous log file before starting
    > "$log_file"
    nohup python3 "$ANALYSIS_SCRIPT" "$NUM_LEVELS" --cores 1 --checkpoint "$checkpoint_path" --generator-script "$generator_script" --no-save >> "$log_file" 2>&1 &
    PID=$!
    PIDS+=($PID)
    echo "- $key Analysis PID: $PID"
done

echo "All background analysis processes started."
echo "PIDs: ${PIDS[@]}"
echo "You can monitor progress with 'tail -f $LOG_DIR/analysis_run_*.log'"
echo "Or check running processes with 'ps -ef | grep run_analysis.py'"

# Deactivate environment (optional, script finishes anyway)
# deactivate

exit 0 