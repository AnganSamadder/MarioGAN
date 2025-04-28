import subprocess
import os
import sys
import argparse

def generate_and_print_stats(project_root: str, no_save: bool, latent_vector: str = None) -> str:
    """Generates level using Java and returns the stdout containing stats."""
    viewer_script_path = os.path.join(project_root, 'run_viewer.sh')
    stdout_content = "" # Default empty output

    args = [viewer_script_path]
    if no_save:
        args.append("--no-save")
    if latent_vector:
        args.append(latent_vector)
        
    try:
        os.chmod(viewer_script_path, 0o755)
        result = subprocess.run(args, cwd=project_root, check=True, capture_output=True, text=True)
        stdout_content = result.stdout # Capture the direct output from Java

        if result.stderr:
            print("--- Java Errors/Warnings (analyze_level.py) ---", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            print("-------------------------------------------------", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Viewer script not found at {viewer_script_path}", file=sys.stderr)
        # Output dummy stats on critical failure
        stdout_content = """LevelWidth: 0
LevelHeight: 0
TotalPipes: 0
BrokenPipes: 0
TotalEnemies: 0
FloatingEnemies: 0
GroundTiles: 0
BreakableTiles: 0
QuestionBlocks: 0
PipeTiles: 0
FloorGaps: 0
LevelValidPipePercentage: 0.0
LevelHasBrokenPipe: 0
LevelHasFloatingEnemy: 0"""
        # sys.exit(1) # Don't exit here, let runner handle aggregation
    except subprocess.CalledProcessError as e:
        print(f"Error running viewer script: {e}", file=sys.stderr)
        print(f"Java Stdout:\n{e.stdout}", file=sys.stderr)
        print(f"Java Stderr:\n{e.stderr}", file=sys.stderr)
        # Output dummy stats
        stdout_content = """LevelWidth: 0
LevelHeight: 0
TotalPipes: 0
BrokenPipes: 0
TotalEnemies: 0
FloatingEnemies: 0
GroundTiles: 0
BreakableTiles: 0
QuestionBlocks: 0
PipeTiles: 0
FloorGaps: 0
LevelValidPipePercentage: 0.0
LevelHasBrokenPipe: 0
LevelHasFloatingEnemy: 0"""
    except Exception as e:
        print(f"An unexpected error occurred during generation: {e}", file=sys.stderr)
        # Output dummy stats
        stdout_content = """LevelWidth: 0
LevelHeight: 0
TotalPipes: 0
BrokenPipes: 0
TotalEnemies: 0
FloatingEnemies: 0
GroundTiles: 0
BreakableTiles: 0
QuestionBlocks: 0
PipeTiles: 0
FloorGaps: 0
LevelValidPipePercentage: 0.0
LevelHasBrokenPipe: 0
LevelHasFloatingEnemy: 0"""

    return stdout_content

# --- Main Execution ---
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run Mario level viewer via shell script and print stats.")
    parser.add_argument("--no-save", action="store_true", 
                        help="Prevent saving level images and text files.")
    parser.add_argument("latent_vector", nargs='?', default=None, 
                        help="Optional latent vector string (e.g., '[0.1, -0.2, ...]')")
    
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Run Java and get stats output
    stats_output = generate_and_print_stats(project_root, args.no_save, args.latent_vector)
    
    # Print the captured stdout (which contains the stats) for the runner script
    print(stats_output)
    
    # Exit cleanly if possible (errors handled within function)
    exit_code = 1 if "LevelWidth: 0" in stats_output else 0 # Crude check if dummy stats were returned
    sys.exit(exit_code) 