#!/bin/bash

# Script to run analysis for both old and new checkpoints in the background.

# --- Get Number of Iterations --- 
NUM_ITERATIONS=${1:-100} # Use first argument or default to 100
# Basic validation to ensure it's a positive integer
if ! [[ "$NUM_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Number of iterations must be a positive integer." >&2
    echo "Usage: $0 [num_iterations]" >&2
    exit 1
fi
echo "Running analysis for $NUM_ITERATIONS iterations per checkpoint."
# ---------------------------------

# Ensure log directory exists
LOG_DIR="analysis/logs"
mkdir -p "$LOG_DIR"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found at venv/bin/activate" >&2
    exit 1
fi

echo "Starting analysis for NEW checkpoint (final_new.pth)... Check $LOG_DIR/analysis_run_new.log"
nohup python3 analysis/run_analysis.py "$NUM_ITERATIONS" --cores 8 --checkpoint samples/final_new.pth --no-save > "$LOG_DIR/analysis_run_new.log" 2>&1 &
NEW_PID=$!

echo "Starting analysis for OLD (default) checkpoint (final.pth)... Check $LOG_DIR/analysis_run_old.log"
nohup python3 analysis/run_analysis.py "$NUM_ITERATIONS" --cores 8 --no-save > "$LOG_DIR/analysis_run_old.log" 2>&1 &
OLD_PID=$!

echo "Background processes started:"
echo "- New Checkpoint Analysis PID: $NEW_PID (Log: $LOG_DIR/analysis_run_new.log)"
echo "- Old Checkpoint Analysis PID: $OLD_PID (Log: $LOG_DIR/analysis_run_old.log)"

# Deactivate environment (optional, script finishes anyway)
# deactivate

exit 0 