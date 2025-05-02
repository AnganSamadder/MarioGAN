# Generator script specifically for models trained within the pytorch_hparam directory.
# Expands a latent vector into a 2D array of tiles.
# Input: JSON array of Z vectors.
# Output: JSON array of levels (tile IDs).

import torch
# import torchvision.utils as vutils # Unused
from torch.autograd import Variable # Keep for legacy model compatibility if needed

import sys
import json
import numpy as np
# Assuming model definitions are in the root models/ directory
import models.dcgan as dcgan
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

    # --- Determine Model Path --- 
    # Default model relative to this script's dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_local = "models/netG_final.pth"
    default_model_abs = os.path.join(script_dir, default_model_local)

    cmd_model = sys.argv[1] if len(sys.argv) > 1 else default_model_abs
    env_model = os.environ.get('MARIOGAN_CHECKPOINT')

    model_to_load = default_model_abs # Start with default

    if env_model and os.path.exists(env_model):
        model_to_load = env_model
        print(f"[hparam_gen] Using model from MARIOGAN_CHECKPOINT: {model_to_load}", file=sys.stderr)
    elif os.path.exists(cmd_model):
        model_to_load = cmd_model
        print(f"[hparam_gen] Using model from command line/default: {model_to_load}", file=sys.stderr)
    # No further fallback needed if default exists, otherwise error out
    elif not os.path.exists(default_model_abs):
         print(f"Error: Cannot find model. Tried env var, cmd arg '{cmd_model}', and default '{default_model_abs}'.", file=sys.stderr)
         sys.exit(1)
    else:
         # Using default because env/cmd failed, but default exists
         print(f"[hparam_gen] Using default model: {model_to_load}", file=sys.stderr)

    # --- Determine Latent Vector Size (nz) ---
    nz = 32 # Default
    if len(sys.argv) >= 3:
        try:
            nz = int(sys.argv[2])
            print(f"[hparam_gen] Using nz={nz} from command line arg.", file=sys.stderr)
        except ValueError:
            print(f"[hparam_gen] Warning: Cannot parse cmd arg '{sys.argv[2]}' for nz. Using default nz={nz}.", file=sys.stderr)
    else:
        # Infer from common checkpoint names if possible
        model_name_lower = os.path.basename(model_to_load).lower()
        if "latent64" in model_name_lower:
            nz = 64
            print(f"[hparam_gen] Inferred nz={nz} from model name.", file=sys.stderr)

    # --- Model Parameters (Adjust based on hparam training setup) ---
    batch_size = 1 # Expect one vector array from Java caller
    image_size = 32
    # Parameters specific to hparam models (confirm these)
    ngf = 256
    ngpu = 1
    n_extra_layers = 0
    z_dims = 10 # Number of output tile types

    print(f"[hparam_gen] Model Params: nz={nz}, ngf={ngf}, layers={n_extra_layers}", file=sys.stderr)

    # --- Instantiate Generator ---
    try:
        generator = dcgan.DCGAN_G(image_size, nz, z_dims, ngf, ngpu, n_extra_layers)
    except Exception as e:
        print(f"Error instantiating DCGAN_G: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Load Weights ---
    print(f"[hparam_gen] Loading weights from: {model_to_load}", file=sys.stderr)
    try:
        # Use weights_only=True for security if available
        try:
            loaded_state_dict = torch.load(model_to_load, map_location=lambda storage, loc: storage, weights_only=True)
        except TypeError:
            print("[hparam_gen] Warning: weights_only=True not supported. Loading without it.", file=sys.stderr)
            loaded_state_dict = torch.load(model_to_load, map_location=lambda storage, loc: storage)

        # No key fixing applied here, assume hparam models have correct keys
        generator.load_state_dict(loaded_state_dict, strict=True)
        print(f"[hparam_gen] Successfully loaded state_dict.", file=sys.stderr)
    except RuntimeError as e:
         print(f"RuntimeError loading state_dict (strict=True failed): {e}", file=sys.stderr)
         print("Ensure model parameters match the checkpoint.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         print(f"Unexpected error loading state_dict: {e}", file=sys.stderr)
         sys.exit(1)

    generator.eval() # Set to evaluation mode

    # --- Simple Testing Block (Optional/Disabled) ---
    testing = False
    if testing:
        # ... (testing code can be added here if needed) ...
        print("[hparam_gen] Testing block executed. Exiting.")
        sys.exit(0)

    # --- Main Interaction Loop (Process one input) ---
    # Signal Java that Python script is ready
    print("READY", flush=True)

    print(f"[hparam_gen] Waiting for JSON input on stdin...", file=sys.stderr, flush=True)
    line = sys.stdin.readline()

    if not line or line.strip() == "0":
        print(f"[hparam_gen] Received termination signal or EOF ('{line.strip()}'). Exiting.", file=sys.stderr, flush=True)
        sys.exit(0) # Exit cleanly

    # Decode JSON input
    lv_input = None
    try:
        lv_input = json.loads(line)
        if not isinstance(lv_input, list) or len(lv_input) == 0 or not isinstance(lv_input[0], list):
             raise ValueError("Input must be a non-empty list of lists/vectors.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error decoding/validating JSON input: {e}\nInput: {line.strip()}", file=sys.stderr, flush=True)
        sys.exit(1)

    lv_np = np.array(lv_input)
    actual_batch_size = lv_np.shape[0]
    actual_input_len = lv_np.shape[1] if lv_np.ndim > 1 else 0

    print(f"[hparam_gen] Received batch size: {actual_batch_size}, Latent dim: {actual_input_len}", file=sys.stderr, flush=True)

    # --- Adjust input vector dimension if needed ---
    if actual_input_len != nz:
        print(f"[hparam_gen] Warning: Input latent dim ({actual_input_len}) != expected nz ({nz}). Adjusting.", file=sys.stderr, flush=True)
        if actual_input_len < nz:
            padding = np.zeros((actual_batch_size, nz - actual_input_len))
            lv_np = np.concatenate((lv_np, padding), axis=1)
        else: # actual_input_len > nz
            lv_np = lv_np[:, :nz]
        print(f"-> Adjusted input vector dimension to {lv_np.shape[1]}.")

    # --- Generate Level ---
    latent_vector = torch.FloatTensor(lv_np).view(actual_batch_size, nz, 1, 1)

    with torch.no_grad():
        # print(f"[hparam_gen] Generating level(s)...", file=sys.stderr, flush=True)
        # Use Variable only if necessary for older torch/model versions
        levels_tensor = generator(Variable(latent_vector))
        # print(f"[hparam_gen] Generation complete.", file=sys.stderr, flush=True)

    # --- Post-process Output ---
    output_np = levels_tensor.data.cpu().numpy()
    # Crop to standard Mario level slice size (14 height x 28 width)
    cropped_output = output_np[:, :, :14, :28]
    # Convert to tile indices by taking argmax along the feature dimension (axis 1)
    level_indices = np.argmax(cropped_output, axis=1)

    # Convert output to list for JSON serialization (send only first level if batch > 1?)
    # Assuming Java caller expects only the first level's data even if input batch > 1
    output_list = level_indices[0].tolist() # Send first level

    # --- Send Output to Java/stdout ---
    print(json.dumps(output_list), flush=True)
    # print(f"[hparam_gen] Sent level data ({len(output_list)} rows) to stdout.", file=sys.stderr, flush=True)

    # print(f"--- HParam Generator Process Finished ---", file=sys.stderr, flush=True)
    sys.exit(0) # Exit cleanly after processing one batch 