import subprocess
import os
import sys
import statistics
import re
from collections import defaultdict
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_stats_output(output: str) -> dict:
    """Parses the Key: Value output of MarioLevelViewer.java into a dictionary."""
    stats = {}
    # Regex to find lines starting with a word, followed by ':', then the value
    # Updated to capture AStarResult which might have non-numeric values
    pattern = re.compile(r"^([A-Za-z]+)\s*:\s*(.*)$") 
    for line in output.strip().split('\n'):
        match = pattern.match(line.strip())
        if match:
            key = match.group(1).strip()
            value_str = match.group(2).strip()
            if key == 'AStarResult':
                stats[key] = value_str # Keep as string ("Win", "Loss", etc.)
            else:
                try:
                    # Attempt to convert other stats to float or int
                    if '.' in value_str:
                        stats[key] = float(value_str)
                    else:
                        stats[key] = int(value_str)
                except ValueError:
                    stats[key] = value_str # Keep as string if conversion fails
    return stats

def run_single_analysis(analyze_script_path, project_root, no_save, run_index, checkpoint_path=None):
    """Runs a single instance of analyze_level.py and returns parsed stats."""
    try:
        # Construct the path to the python executable within the virtual environment
        venv_python_executable = os.path.join(project_root, 'venv', 'bin', 'python3')
        
        # Check if the venv python executable exists
        if not os.path.exists(venv_python_executable):
             # Fallback to sys.executable if venv python isn't found, but print a warning
             # This might happen if the script is run without activating the venv, 
             # though ideally it should be run with the venv active.
             tqdm.write(f"Warning: Virtual environment python not found at {venv_python_executable}. Falling back to {sys.executable}.", file=sys.stderr)
             python_executable = sys.executable
        else:
             python_executable = venv_python_executable
             
        # Construct command for analyze_level.py using the determined python executable
        command = [python_executable, analyze_script_path]
        if no_save:
            command.append("--no-save")
        if checkpoint_path: # Check if checkpoint_path is not None
            command.extend(["--checkpoint", checkpoint_path]) # Pass checkpoint path

        # Run analyze_level.py. Note: It handles its own Java interaction.
        # We capture stdout to get the stats.
        result = subprocess.run(command, 
                                cwd=project_root, # Run from project root like before
                                check=True, 
                                capture_output=True, 
                                text=True,
                                env=os.environ.copy()) # Pass the current environment
        
        # Parse the output
        run_stats = parse_stats_output(result.stdout)
        
        # --- DEBUGGING --- 
        # if run_stats: # Removed debug print
        #     lw_val = run_stats.get('LevelWidth')
        #     bp_val = run_stats.get('BrokenPipes')
        #     tqdm.write(f"DEBUG Run {run_index + 1}: LevelWidth={lw_val} (Type: {type(lw_val)}), BrokenPipes={bp_val} (Type: {type(bp_val)})")
        # else:
        #     tqdm.write(f"DEBUG Run {run_index + 1}: run_stats is None")
        # --- END DEBUGGING ---

        # Check if parsing seemed successful (check for essential keys)
        if 'LevelWidth' in run_stats: 
            return run_stats # Success
        else:
             # Use tqdm.write for messages within the progress bar context
             tqdm.write(f"Warning: Could not parse expected stats from run {run_index + 1}. Output:\n{result.stdout}")
             # Print stderr from the analyze script if any
             if result.stderr:
                 tqdm.write(f"--- analyze_level.py stderr (Run {run_index + 1}) ---")
                 tqdm.write(result.stderr)
                 tqdm.write("-----------------------------------------")
             return None # Indicate failure

    except subprocess.CalledProcessError as e:
        tqdm.write(f"Error during analysis run {run_index + 1}: {e}", file=sys.stderr)
        tqdm.write(f"Stdout:\n{e.stdout}", file=sys.stderr)
        tqdm.write(f"Stderr:\\n{e.stderr}", file=sys.stderr)
        # Also log the command that failed
        tqdm.write(f"Failed command: {' '.join(e.cmd)}", file=sys.stderr)
        return None # Indicate failure
    except Exception as e:
        tqdm.write(f"An unexpected error occurred during run {run_index + 1}: {e}", file=sys.stderr)
        # Log the command if available in the exception context (might not always be)
        # if hasattr(e, 'cmd'):
        #     tqdm.write(f"Command context (if available): {' '.join(e.cmd)}", file=sys.stderr)
        return None # Indicate failure

def main(num_runs: int, no_save: bool, cores_arg: str, checkpoint_path: str = None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    analyze_script_path = os.path.join(script_dir, 'analyze_level.py')

    if not os.path.exists(analyze_script_path):
        print(f"Error: analyze_level.py not found at {analyze_script_path}", file=sys.stderr)
        sys.exit(1)

    all_stats = defaultdict(list)
    successful_runs = 0
    futures = []
    num_workers = 8 # Default number of workers
    # Determine the number of workers based on the --cores argument
    if cores_arg:
        if cores_arg.lower() == 'all':
            detected_cores = os.cpu_count()
            if detected_cores:
                num_workers = detected_cores
            else:
                print("Warning: Could not detect CPU count. Defaulting to 8 cores.", file=sys.stderr)
                num_workers = 8
        else:
            try:
                requested_cores = int(cores_arg)
                if requested_cores > 0:
                    num_workers = requested_cores
                else:
                    print(f"Warning: Number of cores must be positive. Defaulting to 8 cores.", file=sys.stderr)
                    num_workers = 8
            except ValueError:
                print(f"Warning: Invalid value '{cores_arg}' for --cores. Must be an integer or 'all'. Defaulting to 8 cores.", file=sys.stderr)
                num_workers = 8
    # else: num_workers remains the default (8)

    print(f"Starting analysis for {num_runs} level generations using up to {num_workers} cores...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all analysis tasks
        for i in range(num_runs):
            futures.append(executor.submit(run_single_analysis, analyze_script_path, project_root, no_save, i, checkpoint_path))

        # Process results as they complete, showing progress with tqdm
        for future in tqdm(as_completed(futures), total=num_runs, desc="Analyzing Levels"):
            run_stats = future.result() # Get result from completed future
            if run_stats:
                successful_runs += 1
                for key, value in run_stats.items():
                    all_stats[key].append(value)
            # Error reporting happens within run_single_analysis using tqdm.write

    print(f"\nFinished {successful_runs}/{num_runs} successful analysis runs.")

    if successful_runs == 0:
        print("No successful runs completed. Cannot calculate statistics.")
        sys.exit(1)

    # --- Calculate Aggregate Statistics ---
    print("\n--- Aggregate Statistics --- ")

    # A* Completable Stats (Moved to top)
    # --- Removed A* completable calculation and printing ---
    # total_astar_wins = sum(all_stats.get('AStarCompletable', [0]))
    # percentage_completable = (total_astar_wins / successful_runs) * 100.0 if successful_runs > 0 else 0.0
    # print(f"Percentage of Completable Levels: {percentage_completable:.2f}%")
    
    # Pipe Stats
    total_pipes_all_runs = sum(all_stats.get('TotalPipes', [0]))
    total_broken_pipes_all_runs = sum(all_stats.get('BrokenPipes', [0]))
    if total_pipes_all_runs > 0:
        overall_pipe_broken_percentage = (total_broken_pipes_all_runs / total_pipes_all_runs) * 100.0
    else:
        overall_pipe_broken_percentage = 0.0 
    print(f"Overall Pipe Broken Percentage (TotalBroken/TotalPipes): {overall_pipe_broken_percentage:.2f}%")

    levels_with_broken_pipes = sum(all_stats.get('LevelHasBrokenPipe', [0]))
    percentage_levels_with_broken_pipes = (levels_with_broken_pipes / successful_runs) * 100.0
    print(f"Percentage of Levels with Broken Pipes: {percentage_levels_with_broken_pipes:.2f}%")

    # Floating Pipe Stats
    # Using TotalPipes as the denominator since PipesWithBottom is not available
    total_floating_pipes_all_runs = sum(all_stats.get('FloatingPipes', [0]))
    # total_pipes_all_runs is already calculated above for the broken pipe percentage
    if total_pipes_all_runs > 0:
        overall_pipe_floating_percentage = (total_floating_pipes_all_runs / total_pipes_all_runs) * 100.0
    else:
        # If no pipes exist at all, floating percentage is 0%
        overall_pipe_floating_percentage = 0.0
    print(f"Overall Pipe Floating Percentage (TotalFloating/TotalPipes): {overall_pipe_floating_percentage:.2f}%") # Updated description
    
    levels_with_floating_pipes = sum(all_stats.get('LevelHasFloatingPipe', [0]))
    percentage_levels_with_floating_pipes = (levels_with_floating_pipes / successful_runs) * 100.0
    print(f"Percentage of Levels with Floating Pipes: {percentage_levels_with_floating_pipes:.2f}%")
    
    # Enemy Stats
    total_enemies_all_runs = sum(all_stats.get('TotalEnemies', [0]))
    total_floating_enemies_all_runs = sum(all_stats.get('FloatingEnemies', [0]))
    if total_enemies_all_runs > 0:
        overall_enemy_floating_percentage = (total_floating_enemies_all_runs / total_enemies_all_runs) * 100.0
    else:
        overall_enemy_floating_percentage = 0.0
    print(f"Overall Enemy Floating Percentage (TotalFloating/TotalEnemies): {overall_enemy_floating_percentage:.2f}%")

    levels_with_floating_enemies = sum(all_stats.get('LevelHasFloatingEnemy', [0]))
    percentage_levels_with_floating_enemies = (levels_with_floating_enemies / successful_runs) * 100.0
    print(f"Percentage of Levels with Floating Enemies: {percentage_levels_with_floating_enemies:.2f}%")
    
    print("--- Averages Per Level ---")
    # Calculate averages for numerical stats
    for key, values in all_stats.items():
        # Skip helper keys and derived aggregate percentages
        # Removed AStarResult and AStarCompletable from skip list
        if key in {'LevelHasBrokenPipe', 'LevelHasFloatingEnemy', 'LevelValidPipePercentage', 
                   'FloatingEnemies', 'LevelHasFloatingPipe', 'GroundedPipePercentage'}:
            continue 
            
        if values and isinstance(values[0], (int, float)):
            avg = statistics.mean(values)
            print(f"Average {key}: {avg:.2f}") 
        # else: # Handle non-numeric stats if any were added
           # print(f"{key}: Non-numeric data")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run Mario level analysis multiple times in parallel.")
    parser.add_argument("num_runs", type=int, nargs='?', default=3, 
                        help="The number of times to generate and analyze a level (default: 3)")
    parser.add_argument("--cores", type=str, default=None, 
                        help="Number of CPU cores to use (e.g., 4) or 'all' to use all available cores (default: 8)")
    parser.add_argument("--no-save", action="store_true", 
                        help="Prevent saving level images and text files.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the generator .pth checkpoint file to use.") 
    
    args = parser.parse_args()

    # Validate num_runs
    if args.num_runs <= 0:
        print(f"Error: Number of runs must be positive.", file=sys.stderr)
        sys.exit(1)

    # Pass the cores argument to main
    main(args.num_runs, args.no_save, args.cores, args.checkpoint)

    # # Default to 3 runs, but allow command line argument
    # runs = 3 
    # if len(sys.argv) > 1:
    #     try:
    #         runs = int(sys.argv[1])
    #         if runs <= 0:
    #             raise ValueError("Number of runs must be positive.")
    #     except ValueError as e:
    #         print(f"Invalid number of runs specified: '{sys.argv[1]}'. Error: {e}")
    #         print("Usage: python run_analysis.py [number_of_runs]")
    #         sys.exit(1)
    
    # main(runs) 