from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.datasets as dset # Unused
# import torchvision.transforms as transforms # Unused
import torchvision.utils as vutils
from torch.autograd import Variable # Keep if model requires Variable input
import os
import sys # Added for stderr
import numpy as np
import matplotlib.pyplot as plt
import math
import json

import models.dcgan as dcgan
# import models.mlp as mlp # Seems unused

parser = argparse.ArgumentParser(description="PyTorch DCGAN Training for Mario Levels (HParam Variant)")
parser.add_argument('--nz', type=int, default=32, help='Size of the latent z vector')
parser.add_argument('--ngf', type=int, default=256, help='Generator features (HParam default)')
parser.add_argument('--ndf', type=int, default=256, help='Discriminator features (HParam default)')
parser.add_argument('--batchSize', type=int, default=256, help='Input batch size (HParam default)')
parser.add_argument('--niter', type=int, default=5000, help='Number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='Learning rate for Critic (Discriminator)')
parser.add_argument('--lrG', type=float, default=0.00005, help='Learning rate for Generator')
parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
parser.add_argument('--cuda', action='store_true', help='Enable CUDA training')
parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--netG', default='', help="Path to pre-trained netG (to continue training)")
parser.add_argument('--netD', default='', help="Path to pre-trained netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01, help='WGAN weight clamp lower bound')
parser.add_argument('--clamp_upper', type=float, default=0.01, help='WGAN weight clamp upper bound')
parser.add_argument('--Diters', type=int, default=5, help='Number of D iterations per G iteration')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on generator and discriminator')
parser.add_argument('--experiment', default='samples', help='Directory to store samples and models (relative to script dir)')
parser.add_argument('--adam', action='store_true', help='Use Adam optimizer (default is RMSprop)')
# parser.add_argument('--problem', type=int, default=0, help='Level examples index (REMOVED - uses example.json)')

opt = parser.parse_args()
print(opt)

# --- Setup Directories --- 
script_dir = os.path.dirname(os.path.abspath(__file__))

# Experiment directory (for samples)
if opt.experiment is None:
    # Should not happen due to default='samples'
    opt.experiment = os.path.join(script_dir, 'samples')
elif not os.path.isabs(opt.experiment):
    opt.experiment = os.path.join(script_dir, opt.experiment)
print(f"Experiment results will be saved to: {opt.experiment}")
os.makedirs(opt.experiment, exist_ok=True)

# Model save directory
model_save_dir = os.path.join(script_dir, 'models')
print(f"Model checkpoints will be saved to: {model_save_dir}")
os.makedirs(model_save_dir, exist_ok=True)

# --- Setup Reproducibility --- 
opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True # Enable cuDNN benchmark mode for potential speedup

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, consider running with --cuda")

# --- Data Preparation --- 
map_size = 32 # Target size for model input (levels padded to this)
z_dims = 10 # Number of different tile types

# Load data from example.json relative to the script directory
examples_json_path = os.path.join(script_dir, "example.json")
try:
    with open(examples_json_path, 'r') as f:
        X = np.array(json.load(f))
except FileNotFoundError:
    print(f"Error: Cannot find level examples file: {examples_json_path}", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not parse JSON from: {examples_json_path}", file=sys.stderr)
    sys.exit(1)

print(f"Loaded {X.shape[0]} levels. Original shape: {X.shape}")

# Convert to one-hot encoding
X_onehot = np.eye(z_dims, dtype='uint8')[X]
# Roll axis: (num_samples, z_dims, height, width)
X_onehot = np.rollaxis(X_onehot, 3, 1)
print(f"One-hot shape: {X_onehot.shape}")

# Pad levels to the target map_size x map_size
X_train = np.zeros((X.shape[0], z_dims, map_size, map_size), dtype='float32') # Use float32 for PyTorch
# Fill background with one-hot empty space (assuming index 2)
X_train[:, 2, :, :] = 1.0

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

# Initialize Generator
netG = dcgan.DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)
netG.apply(weights_init)
if opt.netG != '' and os.path.exists(opt.netG):
    try:
        netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))
        print(f"Loaded pre-trained netG from: {opt.netG}")
    except Exception as e:
        print(f"Warning: Could not load netG state_dict from {opt.netG}: {e}", file=sys.stderr)
elif opt.netG != '':
     print(f"Warning: Pre-trained netG path specified but not found: {opt.netG}", file=sys.stderr)
# print(netG)

# Initialize Discriminator (Critic)
netD = dcgan.DCGAN_D(map_size, nz, z_dims, ndf, ngpu, n_extra_layers)
netD.apply(weights_init)
if opt.netD != '' and os.path.exists(opt.netD):
    try:
        netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, loc: storage))
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
one = torch.FloatTensor([1]) # For WGAN loss
mone = one * -1 # For WGAN loss

# --- Visualization Utilities ---
def tiles2image(tiles):
    """Converts tile indices to an RGB image using a colormap."""
    normalized_tiles = tiles / float(z_dims)
    colored_image = plt.get_cmap('rainbow')(normalized_tiles)
    return colored_image[..., :3] # Return RGB

def combine_images_grid(image_batch):
    """Combines a batch of images (H, W, C) into a square grid."""
    num = image_batch.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = image_batch.shape[1:] # H, W, C
    grid = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=image_batch.dtype)
    for index, img in enumerate(image_batch):
        i = int(index / width)
        j = index % width
        grid[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img
    return grid

# --- Move models and tensors to GPU if enabled --- 
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
print(f"Starting training for {opt.niter} epochs...")
for epoch in range(opt.niter):
    # Shuffle training data each epoch
    X_train = X_train[np.random.permutation(len(X_train))]

    batch_idx = 0
    while batch_idx < num_batches:

        # --- (1) Update Discriminator (Critic) --- #
        # Train D more iterations early on or periodically
        Diters = opt.Diters
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100 # More D updates initially/periodically

        d_iter = 0
        while d_iter < Diters and batch_idx < num_batches:
            d_iter += 1

            # Clamp D weights (WGAN requirement)
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            # Get real data batch
            start_idx = batch_idx * opt.batchSize
            end_idx = min((batch_idx + 1) * opt.batchSize, len(X_train))
            current_batch_size = end_idx - start_idx
            if current_batch_size <= 0: continue # Skip if somehow batch is empty
            
            data = X_train[start_idx:end_idx]
            batch_idx += 1 # Increment main batch index

            netD.zero_grad()

            # Prepare real data tensor
            real_cpu = torch.FloatTensor(data)
            if opt.cuda:
                real_cpu = real_cpu.cuda()

            # Ensure input tensor matches current batch size
            input_tensor_batch = input_tensor[:current_batch_size]
            input_tensor_batch.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input_tensor_batch)

            # Train D with real batch
            errD_real = netD(inputv)
            errD_real.backward(one)

            # Train D with fake batch
            noise_tensor_batch = noise_tensor[:current_batch_size]
            noise_tensor_batch.resize_(current_batch_size, nz, 1, 1).normal_(0, 1)
            with torch.no_grad(): # No need gradients for G when training D
                noisev = Variable(noise_tensor_batch)
                fake = netG(noisev).detach() # Detach G output

            inputv = Variable(fake) # Wrap detached fake data
            errD_fake = netD(inputv)
            errD_fake.backward(mone)

            errD = errD_real - errD_fake # WGAN loss for D
            optimizerD.step()

        # --- (2) Update Generator --- #
        # Prevent D gradients calculations
        for p in netD.parameters():
            p.requires_grad = False

        netG.zero_grad()
        # Generate fresh noise (use full batch size for G?)
        noise_tensor.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise_tensor)

        fake = netG(noisev)
        errG = netD(fake) # Pass fake samples through D
        errG.backward(one) # Maximize D's output for fake samples (G wants to fool D)
        optimizerG.step()

        gen_iterations += 1

        # Re-enable D gradients for the next D iteration cycle
        for p in netD.parameters():
            p.requires_grad = True

        # --- Logging & Visualization --- #
        if batch_idx % 50 == 0: # Log every 50 batches
            print('[%d/%d][%d/%d][%d] Loss_D: %.4f Loss_G: %.4f Loss_D_real: %.4f Loss_D_fake %.4f'
                  % (epoch, opt.niter, batch_idx, num_batches, gen_iterations,
                     errD.data.item(), errG.data.item(), errD_real.data.item(), errD_fake.data.item()))

        if gen_iterations % 100 == 0: # Generate sample images every 100 G iterations
            with torch.no_grad():
                fixed_noise_v = Variable(fixed_noise)
                fake_fixed = netG(fixed_noise_v).data.cpu().numpy()

            # Process generated images
            fake_fixed = fake_fixed[:, :, :h_orig, :w_orig] # Crop to original size
            fake_tiles = np.argmax(fake_fixed, axis=1)
            img_grid = combine_images_grid(tiles2image(fake_tiles))

            # Save the image grid
            save_path = os.path.join(opt.experiment, f'fake_samples_iter_{gen_iterations:06d}.png')
            try:
                plt.imsave(save_path, img_grid)
            except Exception as e:
                print(f"Warning: Failed to save sample image to {save_path}: {e}", file=sys.stderr)

    # --- End of Epoch --- #
    
    # Save checkpoint every 500 epochs and at the very end
    if (epoch + 1) % 500 == 0 or (epoch + 1) == opt.niter:
        g_save_path = os.path.join(model_save_dir, f'netG_epoch_{epoch+1}.pth')
        d_save_path = os.path.join(model_save_dir, f'netD_epoch_{epoch+1}.pth')
        print(f"---> Checkpointing models at epoch {epoch + 1} <---")
        try:
            torch.save(netG.state_dict(), g_save_path)
            torch.save(netD.state_dict(), d_save_path)
            print(f"Saved checkpoints to {model_save_dir}")
            # Save a final marker too
            if (epoch + 1) == opt.niter:
                 torch.save(netG.state_dict(), os.path.join(model_save_dir, 'netG_final.pth'))
                 torch.save(netD.state_dict(), os.path.join(model_save_dir, 'netD_final.pth'))
                 print("Saved final models as netG_final.pth and netD_final.pth")
        except Exception as e:
            print(f"Error saving checkpoint/final models for epoch {epoch + 1}: {e}", file=sys.stderr)

print("--- Training Finished --- ")
