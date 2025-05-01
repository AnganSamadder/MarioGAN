import subprocess
import os
import sys
import argparse

def generate_and_print_stats(project_root: str, no_save: bool, load_level_path: str = None, latent_vector: str = None, checkpoint: str = None) -> str:
    """Generates level using Java OR loads from file and returns the stdout containing stats."""
    viewer_script_path = os.path.join(project_root, 'run_viewer.sh')
    stdout_content = "" # Default empty output

    # Construct base args for run_viewer.sh
    shell_args = [viewer_script_path]
    if no_save:
        # Note: run_viewer.sh needs modification to pass -Dmariogan.savefiles=false
        # For now, we control saving via the logic added in MarioLevelViewer.java
        # but we can still pass the flag if run_viewer.sh handles it.
        # This python arg might become redundant depending on Java logic.
        # Let's assume run_viewer.sh doesn't handle it directly yet.
        pass # The Java code now checks System.getProperty("mariogan.savefiles")

    # Construct args specifically for MarioLevelViewer.java
    java_args = []
    if load_level_path:
        java_args.extend(["--load-file", load_level_path])
    elif latent_vector:
        # Pass latent vector only if not loading from file
        java_args.append(latent_vector)
    
    # Combine args for the subprocess call
    # run_viewer.sh likely needs to be adjusted to properly pass arguments to the java command
    # Assuming run_viewer.sh passes all subsequent arguments to java:
    full_command_args = shell_args + java_args

    try:
        os.chmod(viewer_script_path, 0o755)
        # Use full_command_args here
        result = subprocess.run(full_command_args, cwd=project_root, check=True, capture_output=True, text=True)
        stdout_content = result.stdout # Capture the direct output from Java

        if result.stderr:
            print("--- Java Errors/Warnings (analyze_level.py) ---", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            print("-------------------------------------------------", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Viewer script not found at {viewer_script_path}", file=sys.stderr)
        stdout_content = "LOAD_FAILED: ScriptNotFound" # Specific error message
    except subprocess.CalledProcessError as e:
        print(f"Error running viewer script: {e}", file=sys.stderr)
        print(f"Java Stdout:\\n{e.stdout}", file=sys.stderr)
        print(f"Java Stderr:\\n{e.stderr}", file=sys.stderr)
        stdout_content = f"LOAD_FAILED: SubprocessError\\n{e.stderr}" # Include stderr
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        stdout_content = f"LOAD_FAILED: UnexpectedError\\n{e}"

    # Ensure dummy stats aren't generated on load failure, return specific message
    if "LOAD_FAILED" in stdout_content and "LevelWidth" not in stdout_content:
         # Add dummy stats if Java didn't even start/print them
         stdout_content += """\nLevelWidth: 0
LevelHeight: 0
TotalPipes: 0
BrokenPipes: 0
FloatingPipes: 0
TotalEnemies: 0
FloatingEnemies: 0
GroundTiles: 0
BreakableTiles: 0
QuestionBlocks: 0
PipeTiles: 0
FloorGaps: 0
LevelValidPipePercentage: 0.0
LevelHasBrokenPipe: 0
LevelHasFloatingEnemy: 0
LevelHasFloatingPipe: 0
AStarResult: LoadFail"""

    return stdout_content

# --- Main Execution ---
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run Mario level viewer/analyzer via shell script and print stats.")
    parser.add_argument("--no-save", action="store_true",
                        help="Prevent saving level images and text files (if generating).")
    parser.add_argument("--load-level", action="store_true", # New flag
                        help="Load level from 'level.txt' instead of generating.")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to the generator .pth checkpoint file to use (passed to run_viewer.sh).")
    parser.add_argument("latent_vector", nargs='?', default=None,
                        help="Optional latent vector string (e.g., '[0.1, -0.2, ...]') used only if --load-level is not specified.")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    level_file_to_load = None
    if args.load_level:
        level_file_to_load = "level.txt" # Hardcode filename for now
        if args.latent_vector:
             print("Warning: Latent vector argument ignored when --load-level is used.", file=sys.stderr)
             args.latent_vector = None # Ensure latent vector is not passed if loading

    # Run Java and get stats output
    stats_output = generate_and_print_stats(project_root, args.no_save, level_file_to_load, args.latent_vector, args.checkpoint)

    # Print the captured stdout (which contains the stats) for the runner script
    print(stats_output)

    # Exit based on stats or load failure
    exit_code = 0
    if "LOAD_FAILED" in stats_output or "AStarResult: LoadFail" in stats_output:
        exit_code = 1
    elif "LevelWidth: 0" in stats_output and "TotalPipes" in stats_output: # Crude check if dummy stats were returned during normal run
        exit_code = 1

    sys.exit(exit_code) 