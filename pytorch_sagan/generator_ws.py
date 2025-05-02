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
import os

import random

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

if __name__ == '__main__':
    # Since the Java program is not launched from the pytorch directory,
    # it cannot find this file when it is specified as being in the current
    # working directory. This is why the network has to be a command line
    # parameter. However, this model should load by default if no parameter
    # is provided.
    if len(sys.argv) ==1:
        # Use a default specific to sagan if known, otherwise keep generic
        modelToLoad = "models/netG_final.pth" # Placeholder - adjust if needed
    else:
        modelToLoad = sys.argv[1]
    if len(sys.argv) >=3:
        nz = int(sys.argv[2])
    else:
        nz = 32

    # --- Check for Environment Variable Checkpoint --- 
    env_checkpoint_path = os.environ.get('MARIOGAN_CHECKPOINT')
    if env_checkpoint_path and os.path.exists(env_checkpoint_path):
        modelToLoad = env_checkpoint_path
        print(f"Loading model from environment variable MARIOGAN_CHECKPOINT: {modelToLoad}", file=sys.stderr)
    elif not os.path.exists(modelToLoad):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rel_model_path = os.path.join(script_dir, modelToLoad) 
        if os.path.exists(rel_model_path):
             modelToLoad = rel_model_path
        else:
             print(f"Error: Model '{modelToLoad}' (or relative '{rel_model_path}') not found.", file=sys.stderr)
             sys.exit(1)
    # --- End Checkpoint Check ---

    batchSize = 1
    imageSize = 32
    # Use sagan specific parameters if they differ
    # Assuming defaults from original pytorch/generator_ws.py for now
    ngf = 64
    ngpu = 1
    n_extra_layers = 0
    z_dims = 10

    # Instantiate the SAGAN model (assuming it's defined in models.dcgan)
    generator = dcgan.DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers)
    print(f"Instantiated sagan generator type: dcgan (with potential attention)", file=sys.stderr)

    # Load weights with strict=False to ignore missing attention keys
    try:
        print(f"Attempting to load state_dict from {modelToLoad} with strict=False", file=sys.stderr)
        generator.load_state_dict(torch.load(modelToLoad, map_location=lambda storage, loc: storage, weights_only=True), strict=False)
        print(f"Successfully loaded state_dict from {modelToLoad} (strict=False)", file=sys.stderr)
    except TypeError:
        print(f"Warning: weights_only=True not supported by this PyTorch version for {modelToLoad}. Loading without it (strict=False).", file=sys.stderr)
        generator.load_state_dict(torch.load(modelToLoad, map_location=lambda storage, loc: storage), strict=False)
        print(f"Successfully loaded state_dict from {modelToLoad} (strict=False, no weights_only)", file=sys.stderr)
    except RuntimeError as e:
        print(f"RuntimeError loading state_dict for {modelToLoad} (strict=False): {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
         print(f"Unexpected error loading state_dict for {modelToLoad} (strict=False): {e}", file=sys.stderr)
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
    print(f"[Single Run SAGAN] Waiting for JSON input on stdin...", file=sys.stderr)
    sys.stderr.flush()
    line = sys.stdin.readline()
    # Check for exit condition (0 sent by Java) or EOF
    if not line or line.strip() == "0":
        print(f"[Single Run SAGAN] Received termination signal or EOF ('{line.strip()}'), exiting.", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1) # Exit with error if no valid input

    print(f"[Single Run SAGAN] Read line: {line.strip()}", file=sys.stderr)
    sys.stderr.flush()

    lv = None # Initialize lv
    try:
        lv = numpy.array(json.loads(line))
    except json.JSONDecodeError:
        print(f"Error decoding JSON input: {line.strip()}", file=sys.stderr)
        sys.exit(1) # Exit with error on bad JSON

    # Input validation (check if lv is a non-empty list/array)
    if not isinstance(lv, (list, numpy.ndarray)) or len(lv) == 0:
        print(f"[Single Run SAGAN] Received invalid or empty input ({lv}), exiting.", file=sys.stderr)
        sys.stdout.flush()
        sys.exit(1) # Exit with error on empty/invalid list

    batchSize = len(lv)
    print(f"[Single Run SAGAN] Received batch size: {batchSize}", file=sys.stderr)
    sys.stderr.flush()

    # Adjust input vector size based on model type
    latent_vector = torch.FloatTensor( lv ).view(batchSize, nz, 1, 1)

    with torch.no_grad():
        print(f"[Single Run SAGAN] Generating level...", file=sys.stderr)
        sys.stderr.flush()
        # Handle potential tuple output from SAGAN generator
        generator_output = generator(Variable(latent_vector))
        if isinstance(generator_output, tuple):
            levels = generator_output[0] # Assume levels tensor is the first element
        else:
            levels = generator_output # Assume it's just the levels tensor
        print(f"[Single Run SAGAN] Generation complete.", file=sys.stderr)
        sys.stderr.flush()

    # Check if levels tensor exists before processing
    if levels is None:
        print(f"[Single Run SAGAN] Error: Generator did not produce a valid levels tensor.", file=sys.stderr)
        sys.exit(1)

    level = levels.data.cpu().numpy()
    level = level[:,:,:14,:28]
    level = numpy.argmax( level, axis = 1)

    print(f"[Single Run SAGAN] Sending output.", file=sys.stderr)
    sys.stderr.flush()
    print(json.dumps(level[0].tolist()))
    sys.stdout.flush()

    print(f"--- SAGAN Generator Process Finished ---", file=sys.stderr)
    sys.stderr.flush()
    sys.exit(0) # Exit normally after processing one input

# End of script


