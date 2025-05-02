# Generator script for StyleGAN models, adapted from DCGAN version.
# Input: JSON array of Z vectors (typically shape [batch_size, z_dim]).
# Output: JSON array of levels (tile IDs).

import torch
# import torchvision.utils as vutils # Unused
# from torch.autograd import Variable # StyleGAN forward pass usually doesn't need Variable wrapper

import sys
import json
import numpy as np
import models.stylegan as stylegan # StyleGAN model definition
import math
# import random # Unused
import os

def combine_images(generated_images):
    """Combines a batch of images into a grid (assumes square layout)."""
    # Note: This function seems unused in the main execution path below.
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img
    return image

if __name__ == '__main__':

    # --- Configuration ---
    # Default StyleGAN parameters (confirm these with training setup)
    z_dim_style = 512
    w_dim_style = 512
    image_size = 32 # Target resolution
    z_dims = 10 # Number of output tile types

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_local = "models/netG_final.pth" # Default model name in this dir
    default_model_abs = os.path.join(script_dir, default_model_local)

    # --- Determine Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stylegan_gen] Using device: {device}", file=sys.stderr)

    # --- Determine Model Path ---
    cmd_model = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('-') else default_model_abs
    env_model = os.environ.get('MARIOGAN_CHECKPOINT')
    model_to_load = default_model_abs # Start with default

    if env_model and os.path.exists(env_model):
        model_to_load = env_model
        print(f"[stylegan_gen] Using model from MARIOGAN_CHECKPOINT: {model_to_load}", file=sys.stderr)
    elif os.path.exists(cmd_model):
        model_to_load = cmd_model
        print(f"[stylegan_gen] Using model from command line/default: {model_to_load}", file=sys.stderr)
    elif not os.path.exists(default_model_abs):
         print(f"Error: Cannot find model. Tried env var, cmd arg '{cmd_model}', and default '{default_model_abs}'.", file=sys.stderr)
         sys.exit(1)
    else:
         print(f"[stylegan_gen] Using default model: {model_to_load}", file=sys.stderr)

    # --- Instantiate Generator ---
    print(f"[stylegan_gen] Instantiating StyleGAN Generator...", file=sys.stderr)
    try:
        generator = stylegan.Generator(z_dim=z_dim_style, w_dim=w_dim_style, out_channels=z_dims, target_resolution=image_size)
    except Exception as e:
        print(f"[stylegan_gen] ERROR during StyleGAN Generator instantiation: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Load Weights ---
    print(f"[stylegan_gen] Loading weights from: {model_to_load}", file=sys.stderr)
    try:
        # Prefer loading 'g_ema' key if present (common in StyleGAN training)
        # Load to CPU first to avoid potential GPU memory issues during load
        checkpoint = torch.load(model_to_load, map_location='cpu', weights_only=True)
        if 'g_ema' in checkpoint:
            generator.load_state_dict(checkpoint['g_ema'], strict=False)
            print(f"[stylegan_gen] Loaded 'g_ema' state_dict.", file=sys.stderr)
        else:
            # Fallback: load the entire checkpoint as state_dict
            generator.load_state_dict(checkpoint, strict=False)
            print(f"[stylegan_gen] Loaded entire checkpoint as state_dict (strict=False).", file=sys.stderr)

    except Exception as e:
        print(f"[stylegan_gen] Error loading weights (attempt 1): {e}. Trying direct load...", file=sys.stderr)
        # Fallback: Try loading the file directly as a state_dict
        try:
            generator.load_state_dict(torch.load(model_to_load, map_location='cpu', weights_only=True), strict=False)
            print(f"[stylegan_gen] Loaded weights directly (attempt 2, strict=False).", file=sys.stderr)
        except Exception as e2:
            print(f"[stylegan_gen] ERROR loading weights even on second attempt: {e2}", file=sys.stderr)
            sys.exit(1)

    generator.to(device) # Move model to target device
    generator.eval()     # Set to evaluation mode
    print(f"[stylegan_gen] Model loaded successfully onto {device}.", file=sys.stderr)

    # --- Simple Testing Block (Optional/Disabled) ---
    testing = False
    if testing:
        # ... (add testing code here if needed, e.g., generate from random noise)
        print("[stylegan_gen] Testing block executed. Exiting.")
        sys.exit(0)

    # --- Main Interaction Loop (Process one input) ---
    print("READY", flush=True) # Signal Java

    print(f"[stylegan_gen] Waiting for JSON input on stdin...", file=sys.stderr, flush=True)
    line = sys.stdin.readline()

    if not line or line.strip() == "0":
        print(f"[stylegan_gen] Received termination signal or EOF ('{line.strip()}'). Exiting.", file=sys.stderr, flush=True)
        sys.exit(0)

    # Decode JSON input
    lv_input = None
    try:
        lv_input = json.loads(line)
        if not isinstance(lv_input, list) or len(lv_input) == 0:
             raise ValueError("Input must be a non-empty list.")
        # Check if it's a list of lists (batch) or just a single vector list
        if not isinstance(lv_input[0], list):
             # Assume single vector sent, wrap in a list for batch processing
             lv_input = [lv_input]
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error decoding/validating JSON input: {e}\nInput: {line.strip()}", file=sys.stderr, flush=True)
        # Send empty list back on error?
        print("[]", flush=True)
        sys.exit(1)

    # Prepare latent vector tensor
    try:
        latent_vector_np = np.array(lv_input, dtype=np.float32)
        # Ensure correct shape [batch_size, z_dim_style]
        if latent_vector_np.shape[1] != z_dim_style:
             raise ValueError(f"Input latent vector dimension ({latent_vector_np.shape[1]}) != expected ({z_dim_style}).")
        latent_vector = torch.from_numpy(latent_vector_np).to(device)
        actual_batch_size = latent_vector.shape[0]
    except Exception as e:
        print(f"Error converting input to tensor: {e}", file=sys.stderr, flush=True)
        print("[]", flush=True)
        sys.exit(1)

    print(f"[stylegan_gen] Received batch size: {actual_batch_size}, Latent dim: {latent_vector.shape[1]}", file=sys.stderr, flush=True)

    # --- Generate Level ---
    with torch.no_grad():
        # print(f"[stylegan_gen] Generating level(s) on {device}...", file=sys.stderr, flush=True)
        levels_tensor = generator(latent_vector) # StyleGAN G takes z directly
        # print(f"[stylegan_gen] Generation complete.", file=sys.stderr, flush=True)

    # --- Post-process Output ---
    # Output shape: [batch_size, out_channels, height, width]
    # Convert to tile indices using argmax along the channel dimension
    level_indices = torch.argmax(levels_tensor, dim=1).cpu().numpy() # Shape: [batch_size, height, width]

    # Convert numpy arrays to lists for JSON serialization
    output_list = [arr.tolist() for arr in level_indices]

    # --- Send Output to Java/stdout ---
    print(json.dumps(output_list), flush=True)
    # print(f"[stylegan_gen] Sent {len(output_list)} level(s) to stdout.", file=sys.stderr, flush=True)

    # print(f"--- StyleGAN Generator Process Finished --- ", file=sys.stderr, flush=True)
    sys.exit(0) # Exit cleanly 