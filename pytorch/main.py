from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import sys

import models.dcgan as dcgan
# import models.mlp as mlp # Seems unused, commented out

parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='generator features')
parser.add_argument('--ndf', type=int, default=64, help='discriminator features')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01, help='WGAN weight clamp lower bound')
parser.add_argument('--clamp_upper', type=float, default=0.01, help='WGAN weight clamp upper bound')
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models (relative to script dir)')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--problem', type=int, default=0, help='Level examples index (0 for default example.json)')
opt = parser.parse_args()
print(opt)

# Determine experiment directory path (relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
if opt.experiment is None:
    opt.experiment = os.path.join(script_dir, 'samples') # Default: ./samples/
elif not os.path.isabs(opt.experiment):
    opt.experiment = os.path.join(script_dir, opt.experiment)

print(f"Experiment results will be saved to: {opt.experiment}")
os.makedirs(opt.experiment, exist_ok=True)

# Define model save directory (relative to this script) and create it
model_save_dir = os.path.join(script_dir, 'models')
print(f"Model checkpoints will be saved to: {model_save_dir}")
os.makedirs(model_save_dir, exist_ok=True)

opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed) # Seed all GPUs if using cuda

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, consider running with --cuda")

# --- Data Preparation ---
map_size = 32 # Target size for model input (levels padded to this)
z_dims = 10 # Number of different tile types

if opt.problem == 0:
    examples_json_path = os.path.join(script_dir, "example.json")
else:
    # Assumes a 'sepEx' subdirectory relative to the script
    examples_json_path = os.path.join(script_dir, "sepEx", f"examplemario{opt.problem}.json")

try:
    with open(examples_json_path, 'r') as f:
        X = np.array(json.load(f))
except FileNotFoundError:
    print(f"Error: Cannot find level examples file: {examples_json_path}", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not parse JSON from: {examples_json_path}", file=sys.stderr)
    sys.exit(1)

print(f"Loaded {X.shape[0]} levels from {os.path.basename(examples_json_path)}. Original shape: {X.shape}")

# Convert to one-hot encoding
# Input X shape: (num_samples, height, width)
# Output X_onehot shape: (num_samples, height, width, z_dims)
X_onehot = np.eye(z_dims, dtype='uint8')[X]

# Roll axis to get shape: (num_samples, z_dims, height, width)
X_onehot = np.rollaxis(X_onehot, 3, 1)
print(f"One-hot shape: {X_onehot.shape}")

# Pad levels to the target map_size x map_size
# Initialize with empty space (assuming index 2 represents empty)
X_train = np.zeros((X.shape[0], z_dims, map_size, map_size), dtype=X_onehot.dtype)
X_train[:, 2, :, :] = 1.0 # Fill background with one-hot empty space

# Place original level data into the padded array
h_orig, w_orig = X.shape[1], X.shape[2]
X_train[:, :, :h_orig, :w_orig] = X_onehot

print(f"Padded training data shape: {X_train.shape}")
num_batches = int(np.ceil(X_train.shape[0] / opt.batchSize))
print(f"Number of batches per epoch: {num_batches}")

# --- Model Initialization ---
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
n_extra_layers = int(opt.n_extra_layers)

def weights_init(m):
    """Custom weights initialization called on netG and netD."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = dcgan.DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)
netG.apply(weights_init)
if opt.netG != '' and os.path.exists(opt.netG):
    try:
        netG.load_state_dict(torch.load(opt.netG))
        print(f"Loaded pre-trained netG from: {opt.netG}")
    except Exception as e:
        print(f"Warning: Could not load netG state_dict from {opt.netG}: {e}", file=sys.stderr)
elif opt.netG != '':
     print(f"Warning: Pre-trained netG path specified but not found: {opt.netG}", file=sys.stderr)
# print(netG)

netD = dcgan.DCGAN_D(map_size, nz, z_dims, ndf, ngpu, n_extra_layers)
netD.apply(weights_init)
if opt.netD != '' and os.path.exists(opt.netD):
    try:
        netD.load_state_dict(torch.load(opt.netD))
        print(f"Loaded pre-trained netD from: {opt.netD}")
    except Exception as e:
        print(f"Warning: Could not load netD state_dict from {opt.netD}: {e}", file=sys.stderr)
elif opt.netD != '':
     print(f"Warning: Pre-trained netD path specified but not found: {opt.netD}", file=sys.stderr)
# print(netD)

# --- Tensor Setup ---
input_tensor = torch.FloatTensor(opt.batchSize, z_dims, map_size, map_size)
noise_tensor = torch.FloatTensor(opt.batchSize, nz, 1, 1)
# Fixed noise vector for consistent visualization during training
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

def tiles2image(tiles):
    """Converts tile indices to an RGB image using a colormap."""
    # Normalize tile indices to [0, 1] for the colormap
    normalized_tiles = tiles / float(z_dims)
    colored_image = plt.get_cmap('rainbow')(normalized_tiles)
    # Return RGB part (drop alpha if present)
    return colored_image[..., :3]

def combine_images_grid(generated_images):
    """Combines a batch of images into a square grid."""
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:] # H, W, C
    # Ensure channel dimension is last if needed by vutils
    # image_batch_for_vutils = generated_images.permute(0, 3, 1, 2) # B, C, H, W
    # grid = vutils.make_grid(image_batch_for_vutils, nrow=width, padding=2, normalize=False)
    # return grid.permute(1, 2, 0).numpy() # H, W, C

    # Manual grid combination (like original code)
    image = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img
    return image

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input_tensor = input_tensor.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise_tensor, fixed_noise = noise_tensor.cuda(), fixed_noise.cuda()

# --- Optimizers ---
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    print("Using Adam optimizer")
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
    print("Using RMSprop optimizer")

# --- Training Loop ---
gen_iterations = 0
print("Starting Training Loop...")
for epoch in range(opt.niter):
    # Shuffle training data each epoch
    X_train = X_train[torch.randperm(len(X_train))]

    batch_idx = 0
    while batch_idx < num_batches:

        # Determine number of Discriminator iterations
        # Train D more iterations early on or periodically
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters

        # --- (1) Update D network --- #
        d_iter = 0
        while d_iter < Diters and batch_idx < num_batches:
            d_iter += 1

            # Clamp parameters (WGAN requirement)
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            # Get real data batch
            start_idx = batch_idx * opt.batchSize
            end_idx = min((batch_idx + 1) * opt.batchSize, len(X_train))
            current_batch_size = end_idx - start_idx
            data = X_train[start_idx:end_idx]
            batch_idx += 1 # Increment batch index after using data

            netD.zero_grad()

            # Prepare real data tensor
            real_cpu = torch.FloatTensor(data)
            if opt.cuda:
                real_cpu = real_cpu.cuda()

            # Use resize_ like this only if necessary and sure about dimensions
            # input_tensor.resize_as_(real_cpu).copy_(real_cpu)
            # Safer: Create Variable directly from real_cpu if dimensions match or slice input_tensor
            input_tensor_resized = input_tensor[:current_batch_size].resize_as_(real_cpu)
            input_tensor_resized.copy_(real_cpu)
            inputv = Variable(input_tensor_resized)

            # Train D with real batch
            errD_real = netD(inputv)
            errD_real.backward(one)

            # Train D with fake batch
            noise_tensor.resize_(current_batch_size, nz, 1, 1).normal_(0, 1)
            # Use torch.no_grad() for G forward pass when only training D
            with torch.no_grad():
                noisev = Variable(noise_tensor) # noisev is Variable(Tensor), not detached
                # fake = netG(noisev).data # .data is legacy, use detach()
                fake = netG(noisev).detach()

            inputv = Variable(fake)
            errD_fake = netD(inputv)
            errD_fake.backward(mone)

            errD = errD_real - errD_fake # WGAN loss for D
            optimizerD.step()

        # --- (2) Update G network --- #
        # Prevent D parameters from requiring gradients
        for p in netD.parameters():
            p.requires_grad = False

        netG.zero_grad()
        # Generate fresh noise for G update
        # Ensure noise matches the expected batch size for G (opt.batchSize or last D batch size?)
        # Let's assume we need a full opt.batchSize for G
        noise_tensor.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise_tensor)

        fake = netG(noisev)
        errG = netD(fake) # Pass fake samples through D to get G loss
        errG.backward(one) # Maximize D's output for fake samples
        optimizerG.step()

        gen_iterations += 1

        # --- Logging & Visualization --- #
        # Re-enable gradients for D for the next iteration
        for p in netD.parameters():
            p.requires_grad = True

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
              % (epoch, opt.niter, batch_idx, num_batches, gen_iterations,
                 errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

        if gen_iterations % 100 == 0:
            # Generate images from fixed noise for visualization
            with torch.no_grad():
                fixed_noise_v = Variable(fixed_noise)
                fake_fixed = netG(fixed_noise_v).data.cpu().numpy()

            # Process generated images (crop, argmax, convert to color)
            fake_fixed = fake_fixed[:, :, :h_orig, :w_orig] # Crop to original size
            fake_tiles = np.argmax(fake_fixed, axis=1)
            img_grid = combine_images_grid(tiles2image(fake_tiles))

            # Save the image grid
            save_path = os.path.join(opt.experiment, 'fake_samples_iter_{0:06d}.png'.format(gen_iterations))
            try:
                plt.imsave(save_path, img_grid)
            except Exception as e:
                print(f"Warning: Failed to save sample image to {save_path}: {e}", file=sys.stderr)

    # --- Save Checkpoints --- #
    if (epoch + 1) % 500 == 0 or epoch == opt.niter - 1: # Save every 500 epochs and at the end
        g_save_path = os.path.join(model_save_dir, f'netG_epoch_{epoch}.pth')
        d_save_path = os.path.join(model_save_dir, f'netD_epoch_{epoch}.pth')
        try:
            torch.save(netG.state_dict(), g_save_path)
            torch.save(netD.state_dict(), d_save_path)
            print(f"Saved models to {model_save_dir} for epoch {epoch}")
        except Exception as e:
            print(f"Error saving models for epoch {epoch}: {e}", file=sys.stderr)

print("Training Finished.")
