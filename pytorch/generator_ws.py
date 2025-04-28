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
        modelToLoad = "netG_epoch_5000.pth"
    else:
        modelToLoad = sys.argv[1]
    if len(sys.argv) >=3:
        nz = int(sys.argv[2])
    else:
        nz = 32

    batchSize = 1
    imageSize = 32
    ngf = 64
    ngpu = 1
    n_extra_layers = 0
    z_dims = 10

    generator = dcgan.DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers)
    # Load weights with weights_only=True for security and to suppress warning
    try:
        # Attempt with weights_only=True first
        generator.load_state_dict(torch.load(modelToLoad, map_location=lambda storage, loc: storage, weights_only=True))
    except TypeError:
        # Fallback if weights_only is not supported in this torch version
        # Print a warning that we are falling back
        print(f"Warning: weights_only=True not supported by this PyTorch version for {modelToLoad}. Loading without it.", file=sys.stderr)
        generator.load_state_dict(torch.load(modelToLoad, map_location=lambda storage, loc: storage))

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

    while 1:
        line = sys.stdin.readline()
        # Check for exit condition (0 sent by Java)
        try:
            if line.strip() == "0":
                break
        except ValueError:
            # Handle cases where the line isn't just "0"
            pass
            
        if not line: # Handle EOF
            break
            
        try:
            lv = numpy.array(json.loads(line))
        except json.JSONDecodeError:
            print(f"Error decoding JSON input: {line.strip()}", file=sys.stderr)
            continue # Skip this input line
            
        latent_vector = torch.FloatTensor( lv ).view(batchSize, nz, 1, 1)

        # Use torch.no_grad() instead of volatile=True
        with torch.no_grad():
            # generator expects input Variable, even if deprecated
            levels = generator(Variable(latent_vector)) 

        level = levels.data.cpu().numpy()
        level = level[:,:,:14,:28] #Cut of rest to fit the 14x28 tile dimensions
        level = numpy.argmax( level, axis = 1)

        # Jacob: Only output first level, since we are only really evaluating one at a time
        print(json.dumps(level[0].tolist()))
        sys.stdout.flush() # Make Java sense output before blocking on next input

# End of script (implicitly exits when loop breaks)


