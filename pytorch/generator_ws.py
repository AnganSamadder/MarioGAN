# Generates a 2D array of tiles from a latent vector.
# Input: JSON array of Z vectors (arrays of floats -1 to 1).
# Output: JSON array of levels (arrays-of-arrays of integer tile IDs).

import torch
import torchvision.utils as vutils
from torch.autograd import Variable

import sys
import json
import numpy as np # Use standard alias
import models.dcgan as dcgan
import math
import random
import os

def combine_images(generated_images):
    """Combines a batch of images into a grid for saving.
    Assumes square grid layout.
    """
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

def fix_state_dict_keys(state_dict):
    """Attempts to fix common key mismatches from older checkpoints.
    Replaces patterns like '.initial.' -> '.initial_', '.convt.' -> '_convt.'
    """
    new_state_dict = {}
    keys_changed = False
    for k, v in state_dict.items():
        # General replacements (might need refinement)
        new_k = k.replace('.initial.', '.initial_') \
                 .replace('.pyramid.', '.pyramid_') \
                 .replace('.final.', '.final_') \
                 .replace('.convt.', '_convt.') \
                 .replace('.batchnorm.', '_batchnorm.')
        # Specific fixes (if general ones are too broad or miss cases)
        if ".convt.weight" in k and "_convt.weight" not in new_k:
            new_k = new_k.replace(".convt.weight", "_convt.weight")
        # ... (add other specific fixes if needed for bias, running_mean, etc.)

        if new_k != k:
            # Log only if a change actually happened
            # print(f"Renaming state_dict key: '{k}' -> '{new_k}'", file=sys.stderr)
            keys_changed = True
        new_state_dict[new_k] = v

    # if keys_changed:
    #     print("Attempted state_dict key renaming.", file=sys.stderr)
    # else:
    #     print("No state_dict key renaming needed.", file=sys.stderr)
    return new_state_dict

if __name__ == '__main__':

    # --- Determine Model Path ---
    default_model = "netG_epoch_5000.pth"
    cmd_model = sys.argv[1] if len(sys.argv) > 1 else default_model
    env_model = os.environ.get('MARIOGAN_CHECKPOINT')
    fallback_model = 'mario_gan.pth' # Last resort

    model_to_load = default_model # Start with default

    if env_model and os.path.exists(env_model):
        model_to_load = env_model
        print(f"Using model from MARIOGAN_CHECKPOINT: {model_to_load}", file=sys.stderr)
    elif os.path.exists(cmd_model):
        model_to_load = cmd_model
        print(f"Using model from command line/default: {model_to_load}", file=sys.stderr)
    elif os.path.exists(fallback_model):
        model_to_load = fallback_model
        print(f"Warning: Specified model '{cmd_model}' not found. Falling back to '{model_to_load}'.", file=sys.stderr)
    else:
        print(f"Error: Cannot find model. Tried env var, '{cmd_model}', and fallback '{fallback_model}'.", file=sys.stderr)
        sys.exit(1)

    # --- Determine Model Parameters (nz, ngf, n_extra_layers, modelType) ---
    nz = 32 # Default latent size
    ngf = 64 # Default generator features
    n_extra_layers = 0 # Default extra layers
    model_type = "dcgan" # Default model type
    z_dims = 10 # Default output features/tile types
    image_size = 32
    ngpu = 1

    model_name_lower = os.path.basename(model_to_load).lower()

    # Infer nz (simple check, adjust if needed)
    if "latent64" in model_name_lower:
        nz = 64
    elif len(sys.argv) >= 3:
        try:
            nz = int(sys.argv[2])
            print(f"Using nz={nz} from command line arg.", file=sys.stderr)
        except ValueError:
            print(f"Warning: Cannot parse cmd arg '{sys.argv[2]}' for nz. Using default nz={nz}.", file=sys.stderr)

    # Infer ngf/n_extra_layers/type (simple checks, adjust as needed)
    if "stylegan" in model_name_lower:
        model_type = "stylegan"
        # Default StyleGAN params (sync with training)
        nz = 512 # StyleGAN uses z_dim
        w_dim_style = 512
        ngf = 0 # Not applicable or set differently
        n_extra_layers = 0 # Not applicable or set differently
    elif "sagan" in model_name_lower:
        model_type = "sagan"
        # Assuming SAGAN uses larger architecture
        if ngf == 64: ngf = 128 # Upgrade default if needed
        if n_extra_layers == 0: n_extra_layers = 3 # Upgrade default if needed
    elif "hparam" in model_name_lower or "netg_latest" in model_name_lower or "loss_test" in model_name_lower:
        # Assume larger DCGAN variant
        if ngf == 64: ngf = 128 # Upgrade default if needed
        if n_extra_layers == 0: n_extra_layers = 3 # Upgrade default if needed
    elif "loss_no_float" in model_name_lower:
         if n_extra_layers == 0: n_extra_layers = 2 # Upgrade default if needed

    print(f"Model Params: Type={model_type}, nz={nz}, ngf={ngf}, layers={n_extra_layers}", file=sys.stderr)

    # --- Instantiate Generator ---
    generator = None
    if model_type == "stylegan":
        try:
            import models.stylegan as stylegan
            generator = stylegan.Generator(z_dim=nz, w_dim=w_dim_style, output_channels=z_dims, target_resolution=image_size)
        except ImportError:
            print("Error: Could not import models.stylegan for StyleGAN model.", file=sys.stderr)
            sys.exit(1)
    elif model_type == "sagan":
        # Assuming SAGAN uses the DCGAN structure from models.dcgan
        generator = dcgan.DCGAN_G(image_size, nz, z_dims, ngf, ngpu, n_extra_layers)
    else: # Default DCGAN
        generator = dcgan.DCGAN_G(image_size, nz, z_dims, ngf, ngpu, n_extra_layers)

    if generator is None:
        print(f"Error: Failed to instantiate generator for type {model_type}", file=sys.stderr)
        sys.exit(1)

    # --- Load Weights ---
    try:
        # Use weights_only=True for security if available
        try:
            loaded_state_dict = torch.load(model_to_load, map_location=lambda storage, loc: storage, weights_only=True)
        except TypeError:
            print(f"Warning: weights_only=True not supported. Loading without it.", file=sys.stderr)
            loaded_state_dict = torch.load(model_to_load, map_location=lambda storage, loc: storage)

        fixed_state_dict = fix_state_dict_keys(loaded_state_dict)
        generator.load_state_dict(fixed_state_dict, strict=True)
        print(f"Successfully loaded state_dict from {model_to_load}", file=sys.stderr)

    except RuntimeError as e:
         print(f"RuntimeError loading state_dict for {model_to_load} (strict=True failed after fix): {e}", file=sys.stderr)
         # Option: Try loading with strict=False as a last resort?
         # try:
         #     generator.load_state_dict(fixed_state_dict, strict=False)
         #     print("Warning: Loaded state_dict with strict=False due to mismatch.", file=sys.stderr)
         # except Exception as e2:
         #     print(f"Error: Failed to load state_dict even with strict=False: {e2}", file=sys.stderr)
         #     sys.exit(1)
         sys.exit(1)
    except Exception as e:
         print(f"Unexpected error loading state_dict for {model_to_load}: {e}", file=sys.stderr)
         sys.exit(1)

    generator.eval() # Set to evaluation mode

    # --- Simple Testing Block (Optional) ---
    testing = False
    if testing:
        print("--- Running Simple Test --- ", file=sys.stderr)
        test_batch_size = 4 # Example batch size
        test_line = [[random.uniform(-1.0, 1.0)] * nz for _ in range(test_batch_size)]
        # print(f"Test Input (JSON): {json.dumps(test_line)}")

        lv_np = np.array(test_line)
        latent_vector = torch.FloatTensor(lv_np).view(test_batch_size, nz, 1, 1)

        with torch.no_grad():
            levels = generator(Variable(latent_vector))

        im = levels.data.cpu().numpy()
        # Crop to standard Mario level slice size
        im = im[:, :, :14, :28]
        # Convert to tile indices
        im = np.argmax(im, axis=1)

        print(f"Output shape after argmax: {im.shape}", file=sys.stderr)
        # Saving requires matplotlib and combine_images function
        # try:
        #     import matplotlib.pyplot as plt
        #     im_colored = (plt.get_cmap('rainbow')(im / float(z_dims)))
        #     # Remove alpha channel if present
        #     if im_colored.shape[-1] == 4:
        #          im_colored = im_colored[..., :3]
        #     combined = combine_images(im_colored)
        #     plt.imsave('fake_sample_test.png', combined)
        #     print("Saved test output to fake_sample_test.png", file=sys.stderr)
        # except ImportError:
        #     print("Matplotlib not found, cannot save test image.", file=sys.stderr)
        # except Exception as e:
        #      print(f"Error saving test image: {e}", file=sys.stderr)
        print("--- Test Finished --- ", file=sys.stderr)
        sys.exit(0)
    # --- End Simple Testing Block ---

    # Signal Java that Python script is ready
    print("READY", flush=True)

    # Process one line of input from Java/stdin
    print(f"[generator_ws.py] Waiting for JSON input on stdin...", file=sys.stderr, flush=True)
    line = sys.stdin.readline()

    if not line or line.strip() == "0":
        print(f"[generator_ws.py] Received termination signal or EOF ('{line.strip()}'). Exiting.", file=sys.stderr, flush=True)
        sys.exit(0) # Exit cleanly on termination signal

    # print(f"[generator_ws.py] Read line: {line.strip()}", file=sys.stderr, flush=True)

    # Decode JSON input
    lv_input = None
    try:
        lv_input = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON input: {e}\nInput: {line.strip()}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Validate input structure (expects list of lists/vectors)
    if not isinstance(lv_input, list) or len(lv_input) == 0 or not isinstance(lv_input[0], list):
        print(f"[generator_ws.py] Received invalid input format. Expected list of vectors. Got: {type(lv_input)}", file=sys.stderr, flush=True)
        sys.exit(1)

    lv_np = np.array(lv_input)
    batch_size = lv_np.shape[0]
    actual_input_len = lv_np.shape[1] if lv_np.ndim > 1 else 0

    print(f"[generator_ws.py] Input batch size: {batch_size}, Latent dim: {actual_input_len}", file=sys.stderr, flush=True)

    # --- Workaround: Pad or truncate input vector to match expected nz ---
    if actual_input_len != nz:
        print(f"Warning: Input latent dimension ({actual_input_len}) does not match model's expected nz ({nz}). Attempting adjustment.", file=sys.stderr, flush=True)
        if actual_input_len < nz:
            # Pad with zeros
            padding = np.zeros((batch_size, nz - actual_input_len))
            lv_np = np.concatenate((lv_np, padding), axis=1)
            print(f"-> Padded input vectors to dimension {nz}.", file=sys.stderr, flush=True)
        else: # actual_input_len > nz
            # Truncate
            lv_np = lv_np[:, :nz]
            print(f"-> Truncated input vectors to dimension {nz}.", file=sys.stderr, flush=True)
    # --- End Workaround ---

    # Prepare tensor for the model
    latent_vector = torch.FloatTensor(lv_np).view(batch_size, nz, 1, 1)

    # Generate levels
    with torch.no_grad():
        levels = generator(Variable(latent_vector))

    # Post-process output
    output_np = levels.data.cpu().numpy()
    # Crop to standard Mario level slice size (14 height x 28 width)
    cropped_output = output_np[:, :, :14, :28]
    # Convert to tile indices by taking argmax along the feature dimension (axis 1)
    level_indices = np.argmax(cropped_output, axis=1)

    # Convert output to list for JSON serialization
    output_list = level_indices.tolist()

    # Send JSON output back to Java/stdout
    print(json.dumps(output_list), flush=True)
    print(f"[generator_ws.py] Sent {len(output_list)} levels to stdout.", file=sys.stderr, flush=True)

    sys.exit(0) # Exit cleanly after processing one batch


