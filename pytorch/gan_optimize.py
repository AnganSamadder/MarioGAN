# Uses CMA-ES to optimize latent vectors for a trained GAN
# to produce levels with specific tile distributions (e.g., maximize pipes).

import torch
import torchvision.utils as vutils
from torch.autograd import Variable
import sys
import json
import numpy as np
import models.dcgan as dcgan
import cma
import random
import math
import matplotlib.pyplot as plt
import os
import re

# --- Configuration ---
batch_size = 1 # CMA-ES typically evaluates one solution at a time
nz = 32  # Dimensionality of latent vector (SHOULD match the loaded model)
image_size = 32 # Internal image size of the GAN
ngf = 64 # Generator features (SHOULD match the loaded model)
ngpu = 1 # Assumed, check if model uses DataParallel
n_extra_layers = 0 # Extra layers (SHOULD match the loaded model)
features = 10 # Number of tile types output by the GAN

# Target tile index for optimization (example: PIPE)
PIPE = 6 # Index depends on the GAN's output mapping
GROUND = 0 # Example for another potential fitness function
ENEMY = 5 # Example

# --- Find and Load Generator Model ---
checkpoint_dir = '.' # Directory containing model checkpoints (relative to script)
model_pattern = re.compile(r"netG_epoch_(\d+)\.pth$")
best_epoch = -1
best_model_path = None

print(f"Searching for latest netG checkpoint in: {os.path.abspath(checkpoint_dir)}")
try:
    for filename in os.listdir(checkpoint_dir):
        match = model_pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_model_path = os.path.join(checkpoint_dir, filename)
except FileNotFoundError:
    print(f"Error: Checkpoint directory not found: {checkpoint_dir}", file=sys.stderr)
    sys.exit(1)

if best_model_path:
    print(f"Loading latest model: {best_model_path} (Epoch {best_epoch})")
else:
    # Fallback or error if no suitable model found
    fallback_model = 'netG_epoch_5000.pth' # Default fallback
    if os.path.exists(os.path.join(checkpoint_dir, fallback_model)):
         print(f"Warning: No netG_epoch_*.pth found. Falling back to {fallback_model}. Ensure parameters (nz, ngf, etc.) match!", file=sys.stderr)
         best_model_path = os.path.join(checkpoint_dir, fallback_model)
    else:
         print(f"Error: No suitable generator model found (netG_epoch_*.pth or {fallback_model}) in {checkpoint_dir}.", file=sys.stderr)
         sys.exit(1)

generator_path = best_model_path

try:
    generator = dcgan.DCGAN_G(image_size, nz, features, ngf, ngpu, n_extra_layers)
    # Use weights_only=True for added security if PyTorch version supports it
    try:
        generator.load_state_dict(torch.load(generator_path, map_location=lambda storage, loc: storage, weights_only=True))
    except TypeError:
        print("Warning: weights_only=True not supported by this PyTorch version. Loading without it.", file=sys.stderr)
        generator.load_state_dict(torch.load(generator_path, map_location=lambda storage, loc: storage))
    generator.eval() # Set to evaluation mode
    print("Generator loaded successfully.")
except Exception as e:
    print(f"Error loading generator state_dict from {generator_path}: {e}", file=sys.stderr)
    print("Ensure model parameters (nz, ngf, features, n_extra_layers) match the checkpoint.")
    sys.exit(1)

# --- Utility Functions ---
def combine_images(generated_images):
    """Combines a batch of images into a grid (assumes square layout)."""
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

def generate_level(latent_vector_list):
    """Generates a level segment from a latent vector list."""
    x = np.array(latent_vector_list)
    if x.shape[0] != batch_size:
        print(f"Warning: Input vector list length ({x.shape[0]}) doesn't match expected batch size ({batch_size}). Reshaping.", file=sys.stderr)
        # Handle potential mismatch - this assumes batch_size=1 is intended
        if batch_size == 1 and x.ndim == 1:
             x = x.reshape(1, -1) # Reshape single vector for batch
        else:
             # More complex handling might be needed for other cases
             pass

    latent_tensor = torch.FloatTensor(x).view(batch_size, nz, 1, 1)
    with torch.no_grad():
        levels = generator(Variable(latent_tensor))

    # Post-process: crop and convert to tile indices
    levels_np = levels.data.cpu().numpy()
    # Crop to standard Mario level slice size (14 height x 28 width)
    levels_cropped = levels_np[:, :, :14, :28]
    # Convert to tile indices by taking argmax along the feature dimension (axis 1)
    level_indices = np.argmax(levels_cropped, axis=1)
    return level_indices # Return shape (batch_size, H, W)

# --- Fitness Functions (Examples) ---
def fitness_maximize_pipes(x):
    """Fitness function: MINIMIZE negative count of PIPE tiles (to maximize pipes)."""
    level = generate_level(x)
    # level shape is (batch_size, H, W)
    pipe_count = np.sum(level == PIPE)
    return -pipe_count # CMA-ES minimizes, so return negative count

def fitness_target_solidity_and_ground(x):
    """Fitness: Minimize distance from target solid block fraction and ground fraction."""
    level_indices = generate_level(x) # Shape (1, H, W)
    level_height, level_width = level_indices.shape[1], level_indices.shape[2]
    total_tiles = level_height * level_width

    # Calculate solidity (fraction of non-empty tiles - assuming tile 0 is empty? Check this)
    # This depends heavily on the tile mapping. Assuming 0=sky, 2=empty ground?
    # Let's assume any tile > 0 is somewhat solid for this example.
    solid_block_count = np.sum(level_indices > 0) # Adjust condition based on actual mapping
    solidity_fraction = solid_block_count / total_tiles

    # Calculate ground block fraction (e.g., count of GROUND tile index on bottom row)
    ground_count = np.sum(level_indices[0, level_height - 1, :] == GROUND)
    ground_fraction = ground_count / level_width

    # Define targets
    target_solidity = 0.4
    target_ground = 0.8

    # Calculate squared error (fitness to be minimized)
    solidity_error = (solidity_fraction - target_solidity) ** 2
    ground_error = (ground_fraction - target_ground) ** 2

    # Combine errors (can weight them if needed)
    fitness = solidity_error + ground_error
    return fitness

# --- CMA-ES Optimization ---
print("Starting CMA-ES optimization...")
# Initial solution (mean): vector of zeros
# Initial standard deviation: 0.5 (controls exploration)
initial_mean = nz * [0]
initial_sigma = 0.5

# Choose the fitness function to optimize
es = cma.CMAEvolutionStrategy(initial_mean, initial_sigma)

# Optimize! CMA-ES will call the fitness function repeatedly.
es.optimize(fitness_maximize_pipes) # Or use fitness_target_solidity_and_ground

print("Optimization finished.")

# --- Results ---
es.result_pretty()
best_solution_vector = es.best.get()[0]
best_fitness = es.best.get()[1]

print(f"\nBest solution vector (latent vector): {np.array(best_solution_vector)}")
print(f"Best fitness found: {best_fitness}")

# Generate and save the best level found
print("Generating level from best solution...")
best_level_tiles = generate_level(best_solution_vector)

# Save the tile array as JSON (optional)
# try:
#     with open('optimized_level.json', 'w') as f:
#         json.dump(best_level_tiles[0].tolist(), f)
#     print("Saved best level tiles to optimized_level.json")
# except Exception as e:
#     print(f"Error saving level JSON: {e}", file=sys.stderr)

# Convert tiles to image and save
print("Saving image of best level...")
try:
    # best_level_tiles has shape (1, H, W), need (H, W) for tiles2image
    img_data = tiles2image(best_level_tiles[0])
    # combine_images expects batch dim (B, H, W, C), add it back
    img_to_combine = img_data[np.newaxis, ...]
    plt.imsave('optimized_level.png', combine_images(img_to_combine))
    print("Saved best level image to optimized_level.png")
except Exception as e:
    print(f"Error saving level image: {e}", file=sys.stderr)

# Plot CMA-ES convergence (optional)
try:
    cma.plot()
    print("Displayed CMA-ES plot.")
except Exception as e:
    print(f"Could not display CMA-ES plot: {e}", file=sys.stderr)

print("Script finished.")
