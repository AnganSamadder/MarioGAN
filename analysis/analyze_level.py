import subprocess
import os
import sys
import argparse
import json
import torch
import random
import tempfile
import numpy

# --- Add print statement at the very beginning ---
# print("--- analyze_level.py: Script starting ---", file=sys.stderr)
# sys.stderr.flush()

def generate_and_print_stats(project_root: str, no_save: bool, load_level_path: str = None, latent_vector: str = None, checkpoint: str = None, generator_script: str = None, latent_filepath: str = None) -> str:
    """Generates level using Java OR loads from file and captures stderr (containing stats), returning it."""
    # --- Add print statement at the start of this function ---
    # print("--- analyze_level.py: generate_and_print_stats() starting ---", file=sys.stderr)
    # sys.stderr.flush()

    viewer_script_path = os.path.join(project_root, 'run_viewer.sh')
    stats_content = "" # Default empty output

    # Construct base args for run_viewer.sh
    shell_args = [viewer_script_path]
    if no_save:
        shell_args.append("--no-save")

    # Pass checkpoint path if provided
    if checkpoint:
        shell_args.extend(["--checkpoint", checkpoint])
        # --- Add print statement when using checkpoint ---
        # print(f"--- analyze_level.py: Using checkpoint: {checkpoint} ---", file=sys.stderr)
        # sys.stderr.flush()

    # Pass generator script path if provided
    if generator_script:
         shell_args.extend(["--generator-script", generator_script])
         # --- Add print statement when using generator script ---
         # print(f"--- analyze_level.py: Using generator script: {generator_script} ---", file=sys.stderr)
         # sys.stderr.flush()

    # Add remaining arguments (load_level_path or latent_vector or latent_filepath)
    if load_level_path:
        shell_args.extend(["-ll", load_level_path])
    elif latent_filepath: # Check for filepath first
        shell_args.extend(["-lf", latent_filepath]) # Use -lf for latent file
    elif latent_vector:
        shell_args.extend(["-lvl", latent_vector])
    else:
        # Default: if neither load nor latent provided, just run viewer (might use default latent)
        pass

    try:
        # --- Add print statement before calling subprocess ---
        # print(f"--- analyze_level.py: Running command: {' '.join(shell_args)} ---", file=sys.stderr)
        # sys.stderr.flush()
        
        # Execute run_viewer.sh, capture stderr (where stats are now printed)
        # Also capture stdout to potentially see other messages or errors
        result = subprocess.run(shell_args, capture_output=True, text=True, check=False, env=os.environ.copy())
        
        # --- Add print statement after calling subprocess ---
        # print(f"--- analyze_level.py: Subprocess finished. RC: {result.returncode} ---", file=sys.stderr)
        # sys.stderr.flush()
        
        if result.returncode != 0:
            print(f"Error running run_viewer.sh. Return code: {result.returncode}", file=sys.stderr)
            print("--- Java stdout (if any) ---", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print("--- Java Errors/Warnings (stderr) ---", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            print("-------------------------------------------------", file=sys.stderr)
            sys.stderr.flush()
             # Return empty string or indicate error, but don't raise here
             # let run_analysis.py handle the failed run
            return "" 
             
        # Capture stderr for parsing
        stats_content = result.stderr 
        
        # Print stdout from Java process for debugging purposes, even on success
        if result.stdout:
            print("--- Java Stdout (analyze_level.py) ---", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print("------------------------------------", file=sys.stderr)
            sys.stderr.flush()

    except FileNotFoundError:
        print(f"Error: {viewer_script_path} not found.", file=sys.stderr)
        sys.stderr.flush()
        # Indicate error by returning empty string
        return ""
    except Exception as e:
        print(f"An unexpected error occurred in generate_and_print_stats: {e}", file=sys.stderr)
        sys.stderr.flush()
        return ""

    # --- Add print statement before returning ---
    # print(f"--- analyze_level.py: generate_and_print_stats() returning (length: {len(stats_content)}) ---", file=sys.stderr)
    # sys.stderr.flush()
    return stats_content

# --- Main Execution ---
if __name__ == "__main__":
    # --- Add print statement before parsing args ---
    # print("--- analyze_level.py: Parsing arguments ---", file=sys.stderr)
    # sys.stderr.flush()

    # StyleGAN specific dimension
    Z_DIM_STYLEGAN = 512

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run Mario level viewer/analyzer via shell script and print stats.")
    parser.add_argument("--no-save", action="store_true",
                        help="Prevent saving level images and text files (passed to run_viewer.sh).")
    parser.add_argument("--load-level", action="store_true", # New flag
                        help="Load level from 'level.txt' instead of generating.")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to the generator .pth checkpoint file to use (passed to run_viewer.sh).")
    parser.add_argument("--generator-script", type=str, default=None, # New argument
                        help="Path to the specific python generator script to use (passed to run_viewer.sh).")
    parser.add_argument("--latent-file", type=str, default=None, # Changed default to None
                        help="Base name for temporary latent file (if needed). Default: uses tempfile module.")
    parser.add_argument("latent_vector", nargs='?', default=None,
                        help="Optional latent vector string (e.g., '[0.1, -0.2, ...]') used only if --load-level or --generator-script=stylegan is not specified.")

    args = parser.parse_args()

    # --- Add print statement after parsing args, before calling function ---
    # print(f"--- analyze_level.py: Arguments parsed. Calling generate_and_print_stats... ---", file=sys.stderr)
    # sys.stderr.flush()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    level_file_to_load = None
    latent_vector_to_use = args.latent_vector
    latent_filepath_to_use = None # Initialize
    temp_latent_file = None # Variable to hold temp file object if created

    if args.load_level:
        level_file_to_load = "level.txt" # Hardcode filename for now
        if args.latent_vector:
             print("Warning: Latent vector argument ignored when --load-level is used.", file=sys.stderr)
             latent_vector_to_use = None # Ensure latent vector is not passed if loading
    elif args.generator_script and "stylegan" in args.generator_script.lower():
        # If using StyleGAN generator, we need to handle the latent vector via file
        latent_vector_to_use = None # Don't pass the string version
        
        # Check if a latent vector was *also* provided as string (error/warning?)
        if args.latent_vector:
            print("Warning: Latent vector string argument ignored when StyleGAN generator is used. Using file method instead.", file=sys.stderr)
            sys.stderr.flush()
        
        print(f"--- analyze_level.py: StyleGAN detected. Generating random {Z_DIM_STYLEGAN}-dim vector and saving to temporary file. ---", file=sys.stderr)
        sys.stderr.flush()
        # Generate a random latent vector of shape [Z_DIM_STYLEGAN] (flat)
        random_z = torch.randn(Z_DIM_STYLEGAN) # Changed shape to be flat
        
        # Create a uniquely named temporary file
        try:
            # Create named temporary file that persists until closed
            temp_latent_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, 
                                                         prefix=args.latent_file or 'latent_', 
                                                         dir='.') # Create in current dir for Java access
            latent_filepath_to_use = temp_latent_file.name
            # Save to the temporary file (space separated)
            numpy.savetxt(latent_filepath_to_use, random_z.numpy(), newline=" ") # Save as single line, space separated
            temp_latent_file.close() # Close the file handle but keep the file
            print(f"--- analyze_level.py: Saved latent vector to {latent_filepath_to_use}. ---", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"ERROR creating/saving temporary latent vector file: {e}", file=sys.stderr)
            sys.stderr.flush()
            if temp_latent_file: # Clean up if file object exists
                try:
                    os.remove(temp_latent_file.name)
                except OSError:
                    pass # Ignore error if file couldn't be removed
            sys.exit(1)

    # Run Java and get stats output (now from stderr)
    try:
        stats_output = generate_and_print_stats(
            project_root,
            args.no_save,
            level_file_to_load,
            latent_vector_to_use, # Use the string if not using StyleGAN file method
            args.checkpoint,
            args.generator_script,
            latent_filepath=latent_filepath_to_use # Pass the unique filepath 
        )
    finally:
        # --- Clean up temporary file --- 
        if latent_filepath_to_use and os.path.exists(latent_filepath_to_use):
            try:
                os.remove(latent_filepath_to_use)
                # print(f"--- analyze_level.py: Cleaned up temp file {latent_filepath_to_use} ---", file=sys.stderr)
                # sys.stderr.flush()
            except OSError as e:
                print(f"Warning: Could not remove temporary latent file {latent_filepath_to_use}: {e}", file=sys.stderr)
                sys.stderr.flush()

    # --- Add print statement before printing final output ---
    # print(f"--- analyze_level.py: Printing final stats output to stdout (length: {len(stats_output)}) ---", file=sys.stderr)
    # sys.stderr.flush()

    # Print the captured stats_output (originally from Java's stderr) to this script's stdout
    print(stats_output)

    # Exit based on stats or load failure
    exit_code = 0
    if "LOAD_FAILED" in stats_output or "AStarResult: LoadFail" in stats_output:
        exit_code = 1
    elif "LevelWidth: 0" in stats_output and "TotalPipes" in stats_output: # Crude check if dummy stats were returned during normal run
        exit_code = 1

    # --- Add print statement at the very end ---
    # print("--- analyze_level.py: Script finished ---", file=sys.stderr)
    # sys.stderr.flush()

    sys.exit(exit_code) 