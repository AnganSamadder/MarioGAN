# This generator program expands a low-dimentional latent vector into a 2D array of tiles.
# Each line of input should be an array of z vectors (which are themselves arrays of floats -1 to 1)
# Each line of output is an array of 32 levels (which are arrays-of-arrays of integer tile ids)

import torch
import torchvision.utils as vutils
from torch.autograd import Variable

import sys
import json
import numpy
import models.dcgan as dcgan
#import matplotlib.pyplot as plt
import math

import random
import os # Import os module

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = numpy.zeros((height*shape[0], width*shape[1],shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img
    return image

# Function to fix layer names with extra dots
def fix_state_dict_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # Replace patterns like '.initial.32-256.' with '.initial_32-256_'
        # This is a bit crude; might need refinement based on exact naming patterns
        new_k = k.replace('.initial.', '.initial_') \
                 .replace('.pyramid.', '.pyramid_') \
                 .replace('.final.', '.final_') \
                 .replace('.convt.', '_convt.') \
                 .replace('.batchnorm.', '_batchnorm.')
        # Specific fix for keys observed in error log
        if ".convt.weight" in k:
            new_k = new_k.replace(".convt.weight", "_convt.weight")
        if ".batchnorm.weight" in k:
            new_k = new_k.replace(".batchnorm.weight", "_batchnorm.weight")
        if ".batchnorm.bias" in k:
             new_k = new_k.replace(".batchnorm.bias", "_batchnorm.bias")
        if ".batchnorm.running_mean" in k:
             new_k = new_k.replace(".batchnorm.running_mean", "_batchnorm.running_mean")
        if ".batchnorm.running_var" in k:
             new_k = new_k.replace(".batchnorm.running_var", "_batchnorm.running_var")
             
        if new_k != k:
             print(f"Renaming state_dict key: '{k}' -> '{new_k}'", file=sys.stderr)
             
        new_state_dict[new_k] = v
    return new_state_dict

if __name__ == '__main__':
    # Since the Java program is not launched from the pytorch directory,
    # it cannot find this file when it is specified as being in the current
    # working directory. This is why the network has to be a command line
    # parameter. However, this model should load by default if no parameter
    # is provided.
    if len(sys.argv) ==1:
        modelToLoad = "netG_epoch_5000.pth"
    else:
        modelToLoad = sys.argv[1]
    if len(sys.argv) >=3:
        nz = int(sys.argv[2])
    else:
        nz = 32

    # --- Check for Environment Variable Checkpoint --- (Added)
    env_checkpoint_path = os.environ.get('MARIOGAN_CHECKPOINT')
    if env_checkpoint_path and os.path.exists(env_checkpoint_path):
        modelToLoad = env_checkpoint_path
        print(f"Loading model from environment variable MARIOGAN_CHECKPOINT: {modelToLoad}", file=sys.stderr)
    elif not os.path.exists(modelToLoad):
        # Fallback if default or command-line model doesn't exist
        # Maybe try a known default like mario_gan.pth if primary target fails
        fallback_model = 'mario_gan.pth'
        if os.path.exists(fallback_model):
            print(f"Warning: Model '{modelToLoad}' not found. Falling back to '{fallback_model}'.", file=sys.stderr)
            modelToLoad = fallback_model
        else:
            print(f"Error: Model '{modelToLoad}' not found and fallback '{fallback_model}' also not found.", file=sys.stderr)
            sys.exit(1) # Cannot proceed without a model
    else:
         print(f"Loading model: {modelToLoad}", file=sys.stderr)
    # --- End Checkpoint Check ---

    batchSize = 1
    imageSize = 32
    ngf = 64
    ngpu = 1
    n_extra_layers = 0
    z_dims = 10

    # --- Model Instantiation based on Type --- (Added)
    # Simple type inference based on path (can be improved)
    modelType = "dcgan" # Default
    if "stylegan" in modelToLoad.lower():
        modelType = "stylegan"
        # Need StyleGAN imports and parameters
        import models.stylegan as stylegan # Assuming stylegan model file exists
        # Default StyleGAN parameters (match training script if possible)
        z_dim_style = 512
        w_dim_style = 512
        generator = stylegan.Generator(z_dim=z_dim_style, w_dim=w_dim_style, output_channels=z_dims, target_resolution=imageSize)
        # Adjust nz for StyleGAN input
        nz = z_dim_style # StyleGAN uses z_dim
    elif "sagan" in modelToLoad.lower():
         modelType = "sagan"
         # Need SAGAN imports (assuming it uses dcgan.py with attention)
         import models.dcgan as dcgan_sagan # Use alias if needed
         # Parameters might be same as DCGAN, or specific if model file differs
         generator = dcgan_sagan.DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers) # Check if DCGAN_G is correct for SAGAN
    else: # Default DCGAN
        generator = dcgan.DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers)

    print(f"Instantiated generator type: {modelType}", file=sys.stderr)
    # --- End Model Instantiation ---

    # Load weights with fix for key mismatch
    try:
        # Load the raw state dict first
        loaded_state_dict = torch.load(modelToLoad, map_location=lambda storage, loc: storage, weights_only=True)
        # Fix the keys
        fixed_state_dict = fix_state_dict_keys(loaded_state_dict)
        # Load the fixed state dict
        generator.load_state_dict(fixed_state_dict, strict=True) # Keep strict=True after fixing
        print(f"Successfully loaded fixed state_dict from {modelToLoad}", file=sys.stderr)
    except TypeError:
        # Fallback if weights_only is not supported
        print(f"Warning: weights_only=True not supported by this PyTorch version for {modelToLoad}. Loading without it.", file=sys.stderr)
        loaded_state_dict = torch.load(modelToLoad, map_location=lambda storage, loc: storage)
        fixed_state_dict = fix_state_dict_keys(loaded_state_dict)
        generator.load_state_dict(fixed_state_dict, strict=True)
        print(f"Successfully loaded fixed state_dict from {modelToLoad} (no weights_only)", file=sys.stderr)
    except RuntimeError as e:
         print(f"RuntimeError loading state_dict for {modelToLoad} even after attempting key fixes: {e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
         print(f"Unexpected error loading state_dict for {modelToLoad}: {e}", file=sys.stderr)
         sys.exit(1)

    testing = False

    if testing:
        line = []
        for i in range (batchSize):
            line.append( [ random.uniform(-1.0, 1.0) ]*nz )

        #This is the format that we expect from sys.stdin
        print(line)
        line = json.dumps(line)
        lv = numpy.array(json.loads(line))
        latent_vector = torch.FloatTensor( lv ).view(batchSize, nz, 1, 1)
        # Use torch.no_grad() instead of volatile=True
        with torch.no_grad():
            levels = generator(Variable(latent_vector)) # Variable is also somewhat legacy, but keep for now
        im = levels.data.cpu().numpy()
        im = im[:,:,:14,:28] #Cut of rest to fit the 14x28 tile dimensions
        im = numpy.argmax( im, axis = 1)
        #print(json.dumps(levels.data.tolist()))
        print("Saving to file ")
        # Requires matplotlib, which might not be available
        # im = ( plt.get_cmap('rainbow')( im/float(z_dims) ) )
        # plt.imsave('fake_sample.png', combine_images(im) )

        exit()

    print("READY") # Java loops until it sees this special signal
    sys.stdout.flush() # Make sure Java can sense this output before Python blocks waiting for input

    # Removed the while 1: loop to process only one input batch
    # while 1:
    print(f"[Single Run PT] Waiting for JSON input on stdin...", file=sys.stderr)
    sys.stderr.flush()
    line = sys.stdin.readline()
    # Check for exit condition (0 sent by Java) or EOF
    if not line or line.strip() == "0":
        print(f"[Single Run PT] Received termination signal or EOF ('{line.strip()}'), exiting.", file=sys.stderr)
        sys.stderr.flush()
        # break # Exit script instead of breaking loop
        sys.exit(1) # Exit with error if no valid input

    print(f"[Single Run PT] Read line: {line.strip()}", file=sys.stderr)
    sys.stderr.flush()
        
    lv = None # Initialize lv
    try:
        lv = numpy.array(json.loads(line))
    except json.JSONDecodeError:
        print(f"Error decoding JSON input: {line.strip()}", file=sys.stderr)
        # continue # Exit script instead of continuing loop
        sys.exit(1) # Exit with error on bad JSON
            
    # Input validation (check if lv is a non-empty list/array)
    if not isinstance(lv, (list, numpy.ndarray)) or len(lv) == 0:
        print(f"[Single Run PT] Received invalid or empty input ({lv}), exiting.", file=sys.stderr)
        sys.stdout.flush()
        sys.exit(1) # Exit with error on empty/invalid list
        
    # Assuming batch size is determined by input length, but typically 1 for this script
    batchSize = len(lv) # Might need adjustment if input format differs
    print(f"[Single Run PT] Received batch size: {batchSize}", file=sys.stderr)
    sys.stderr.flush()
    
    # Adjust input vector size based on model type
    latent_vector = torch.FloatTensor( lv ).view(batchSize, nz, 1, 1)

    # Use torch.no_grad() instead of volatile=True
    with torch.no_grad():
        print(f"[Single Run PT] Generating level...", file=sys.stderr)
        sys.stderr.flush()
        # generator expects input Variable, even if deprecated
        levels = generator(Variable(latent_vector)) 
        print(f"[Single Run PT] Generation complete.", file=sys.stderr)
        sys.stderr.flush()

    level = levels.data.cpu().numpy()
    level = level[:,:,:14,:28] #Cut of rest to fit the 14x28 tile dimensions
    level = numpy.argmax( level, axis = 1)

    # Jacob: Only output first level, since we are only really evaluating one at a time
    print(f"[Single Run PT] Sending output.", file=sys.stderr)
    sys.stderr.flush()
    print(json.dumps(level[0].tolist()))
    sys.stdout.flush() # Make Java sense output before blocking on next input

    print(f"--- PyTorch Generator Process Finished ---", file=sys.stderr)
    sys.stderr.flush()
    sys.exit(0) # Exit normally after processing one input

# End of script (implicitly exits when loop breaks)


