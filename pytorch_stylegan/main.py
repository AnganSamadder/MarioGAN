# Training script for StyleGAN on Mario levels.
from __future__ import print_function
import matplotlib
matplotlib.use('Agg') # Use non-GUI backend BEFORE importing pyplot

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
# from torch.autograd import Variable # Generally not needed with modern PyTorch
import os
import sys # Added
import numpy as np
import matplotlib.pyplot as plt
import math
import json

# Import StyleGAN models
import models.stylegan as stylegan

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="PyTorch StyleGAN Training for Mario Levels")
# StyleGAN specific parameters
parser.add_argument('--z_dim', type=int, default=512, help='Size of the latent z vector')
parser.add_argument('--w_dim', type=int, default=512, help='Size of the intermediate style vector w')
# parser.add_argument('--map_hidden_dim', type=int, default=512, help='Mapping network hidden dim') # Currently unused by Generator
# parser.add_argument('--map_layers', type=int, default=8, help='Number of layers in mapping network') # Currently unused by Generator
# General GAN/Training parameters
parser.add_argument('--batchSize', type=int, default=64, help='Input batch size (adjust based on GPU memory)')
parser.add_argument('--niter', type=int, default=5000, help='Number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='Learning rate for Discriminator')
parser.add_argument('--lrG', type=float, default=0.0002, help='Learning rate for Generator')
parser.add_argument('--beta1', type=float, default=0.0, help='Beta1 for Adam (StyleGAN default: 0.0)')
parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam (StyleGAN default: 0.99)')
parser.add_argument('--cuda', action='store_true', help='Enable CUDA training')
parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs to use (Currently only supports 1)')
parser.add_argument('--netG', default='', help="Path to pre-trained netG (to continue training)")
parser.add_argument('--netD', default='', help="Path to pre-trained netD (to continue training)")
parser.add_argument('--experiment', default='samples', help='Directory for samples/models (relative to script)')
# parser.add_argument('--problem', type=int, default=0, help='Level examples index (IGNORED - uses example.json)')

opt = parser.parse_args()
print(opt)

# --- Setup Directories & Reproducibility ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(opt.experiment):
    opt.experiment = os.path.join(script_dir, opt.experiment)
print(f"Experiment results will be saved to: {opt.experiment}")
os.makedirs(opt.experiment, exist_ok=True)

model_save_dir = os.path.join(script_dir, 'models')
print(f"Model checkpoints will be saved to: {model_save_dir}")
os.makedirs(model_save_dir, exist_ok=True)

opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.cuda) else "cpu")
print("Using device:", device)
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: CUDA available but not enabled via --cuda flag.")

# --- Data Loading & Preprocessing ---
map_size = 32 # Target resolution for StyleGAN output
z_dims_data = 10 # Number of tile types from the data

examples_json_path = os.path.join(script_dir, "example.json")
try:
    with open(examples_json_path, 'r') as f:
        X = np.array(json.load(f))
except FileNotFoundError:
    print(f"Error: Cannot find level examples file: {examples_json_path}", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Could not parse JSON from {examples_json_path}: {e}", file=sys.stderr)
    sys.exit(1)

print(f"Loaded {X.shape[0]} levels. Original shape: {X.shape}")

# One-hot encode
X_onehot = np.eye(z_dims_data, dtype='uint8')[X]
X_onehot = np.rollaxis(X_onehot, 3, 1).astype('float32') # Shape: (N, C, H, W), float32 for torch

# Pad levels to map_size x map_size
h_orig, w_orig = X.shape[1], X.shape[2]
X_padded = np.zeros((X.shape[0], z_dims_data, map_size, map_size), dtype='float32')
ID_EMPTY = 2 # Assuming index 2 is empty based on original code
X_padded[:, ID_EMPTY, :, :] = 1.0 # Fill background
h_pad = min(h_orig, map_size)
w_pad = min(w_orig, map_size)
X_padded[:, :, :h_pad, :w_pad] = X_onehot[:, :, :h_pad, :w_pad]

print(f"Padded training data shape: {X_padded.shape}")

# Create DataLoader
dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_padded))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=4, pin_memory=True)

# --- Model Initialization ---
print("Initializing StyleGAN models...")
netG = stylegan.Generator(opt.z_dim, opt.w_dim, z_dims_data, target_resolution=map_size).to(device)
netD = stylegan.Discriminator(z_dims_data, target_resolution=map_size).to(device)

# Optional: Apply custom weight initialization
# def weights_init(m): ...
# netG.apply(weights_init)
# netD.apply(weights_init)

# Load checkpoints if specified
if opt.netG != '' and os.path.exists(opt.netG):
    try:
        # StyleGAN checkpoints might contain more than just state_dict (e.g., g_ema)
        checkpoint = torch.load(opt.netG, map_location=device)
        state_dict_key = 'g_ema' if 'g_ema' in checkpoint else None
        if state_dict_key:
            netG.load_state_dict(checkpoint[state_dict_key], strict=False)
            print(f"Loaded netG state_dict ('{state_dict_key}') from: {opt.netG}")
        else: # Assume checkpoint is the state_dict itself
            netG.load_state_dict(checkpoint, strict=False)
            print(f"Loaded netG state_dict directly from: {opt.netG}")
    except Exception as e:
        print(f"Warning: Could not load netG state_dict from {opt.netG}: {e}", file=sys.stderr)
elif opt.netG != '':
     print(f"Warning: Pre-trained netG path specified but not found: {opt.netG}", file=sys.stderr)
# print(netG)

if opt.netD != '' and os.path.exists(opt.netD):
    try:
        # Discriminator checkpoint likely just the state_dict
        netD.load_state_dict(torch.load(opt.netD, map_location=device), strict=False)
        print(f"Loaded pre-trained netD from: {opt.netD}")
    except Exception as e:
        print(f"Warning: Could not load netD state_dict from {opt.netD}: {e}", file=sys.stderr)
elif opt.netD != '':
     print(f"Warning: Pre-trained netD path specified but not found: {opt.netD}", file=sys.stderr)
# print(netD)

# --- Loss and Optimizers ---
# Use standard GAN loss (BCEWithLogits) for StyleGAN training initially
criterion = nn.BCEWithLogitsLoss()

# Labels for BCE loss
real_label = 1.
fake_label = 0.

# Adam optimizers with StyleGAN betas
optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))

# --- Fixed noise for visualization ---
fixed_noise = torch.randn(opt.batchSize, opt.z_dim, device=device)

# --- Visualization Utilities ---
def tiles2image(tiles):
    """Converts tile indices to an RGB image using a colormap."""
    norm_tiles = tiles / float(z_dims_data) if z_dims_data > 0 else tiles
    colored_image = plt.get_cmap('rainbow')(norm_tiles)
    return colored_image[..., :3]

def combine_images_grid(image_batch):
    """Combines a batch of images (H, W, C) into a square grid."""
    num = image_batch.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = image_batch.shape[1:]
    grid = np.zeros((height * shape[0], width * shape[1], shape[2]), dtype=image_batch.dtype)
    for index, img in enumerate(image_batch):
        i = int(index / width)
        j = index % width
        grid[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img
    return grid

# --- Training Loop ---
print("Starting StyleGAN Training Loop...")
gen_iterations = 0
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        # --- (1) Update Discriminator --- #
        netD.zero_grad()
        # Real batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        output_real = netD(real_cpu).view(-1) # Flatten D output
        errD_real = criterion(output_real, label)
        errD_real.backward()
        D_x = output_real.mean().item()

        # Fake batch
        noise = torch.randn(b_size, opt.z_dim, device=device)
        with torch.no_grad(): # No need to track G gradients here
            fake = netG(noise)
        label.fill_(fake_label)

        output_fake = netD(fake).view(-1)
        errD_fake = criterion(output_fake, label)
        errD_fake.backward()
        D_G_z1 = output_fake.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        # --- (2) Update Generator --- #
        netG.zero_grad()
        label.fill_(real_label) # G wants D to think fake images are real
        # Generate fake data again (gradients will flow back)
        noise = torch.randn(b_size, opt.z_dim, device=device) # Generate fresh noise
        fake = netG(noise)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        gen_iterations += 1

        # --- Logging & Visualization --- #
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if gen_iterations % 100 == 0:
            with torch.no_grad():
                fake_fixed = netG(fixed_noise).detach().cpu()

            im_raw = fake_fixed.numpy()
            # Crop output to original data dimensions for visualization
            im_cropped = im_raw[:, :, :h_orig, :w_orig]
            im_tiles = np.argmax(im_cropped, axis=1)
            im_display = combine_images_grid(tiles2image(im_tiles))

            if im_display is not None:
                 save_path = os.path.join(opt.experiment, f'fake_samples_stylegan_iter_{gen_iterations:06d}.png')
                 try:
                     plt.imsave(save_path, im_display)
                 except Exception as e:
                     print(f"Warning: Failed to save sample image to {save_path}: {e}", file=sys.stderr)

            # Save latest models (overwriting)
            try:
                torch.save(netG.state_dict(), os.path.join(model_save_dir, 'netG_latest.pth'))
                torch.save(netD.state_dict(), os.path.join(model_save_dir, 'netD_latest.pth'))
            except Exception as e:
                print(f"Warning: Error saving latest models: {e}", file=sys.stderr)

    # --- End of Epoch --- #
    if (epoch + 1) % 500 == 0 or (epoch + 1) == opt.niter:
        g_save_path = os.path.join(model_save_dir, f'netG_stylegan_epoch_{epoch+1}.pth')
        d_save_path = os.path.join(model_save_dir, f'netD_stylegan_epoch_{epoch+1}.pth')
        print(f"---> Checkpointing models at epoch {epoch + 1} <---")
        try:
            torch.save(netG.state_dict(), g_save_path)
            torch.save(netD.state_dict(), d_save_path)
            print(f"Saved checkpoints to {model_save_dir}")
            if (epoch + 1) == opt.niter:
                 torch.save(netG.state_dict(), os.path.join(model_save_dir, 'netG_final.pth'))
                 torch.save(netD.state_dict(), os.path.join(model_save_dir, 'netD_final.pth'))
                 print("Saved final models.")
        except Exception as e:
            print(f"Error saving checkpoint/final models for epoch {epoch + 1}: {e}", file=sys.stderr)

print("--- StyleGAN Training Finished --- ") 