# Script intended for Latin Hypercube Sampling (LHC) of the latent space.
# However, the current code seems incomplete and uses undefined variables (e.g., 'es', 'x').
# It also hardcodes model loading ('netG_epoch_24.pth') and appears to mix generation with undefined optimization results.

import torch
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import models.dcgan as dcgan # Assuming this model structure
# import random # Not used directly in remaining code
# from pyDOE import lhs # Need this library for actual LHC sampling

# --- Configuration (Should ideally match the loaded model) ---
batchSize = 1 # LHC typically generates one sample at a time, but could do batches
nz = 32  # Dimensionality of latent vector
imageSize = 32
ngf = 64
ngpu = 1 # Assumed
n_extra_layers = 0
features = 10 # Number of output features/tile types - NEEDS CONFIRMATION for netG_epoch_24.pth
model_path = 'netG_epoch_24.pth' # Hardcoded model path
output_height = 14
output_width = 28
num_samples = 10 # Example: Number of LHC samples to generate

# --- Load Generator Model ---
print(f"Loading generator: {model_path}")
try:
    # Assuming DCGAN_G structure; adjust if needed for the specific epoch 24 model
    generator = dcgan.DCGAN_G(imageSize, nz, features, ngf, ngpu, n_extra_layers)
    # Use weights_only=True for security if PyTorch version supports it
    try:
        generator.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=True))
    except TypeError:
        print("Warning: weights_only=True not supported. Loading without it.", file=sys.stderr)
        generator.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    generator.eval()
    print("Generator loaded successfully.")
except FileNotFoundError:
     print(f"Error: Model file not found: {model_path}", file=sys.stderr)
     sys.exit(1)
except Exception as e:
    print(f"Error loading generator state_dict from {model_path}: {e}", file=sys.stderr)
    print("Ensure model parameters (nz, ngf, features, etc.) match the checkpoint.")
    sys.exit(1)


# --- LHC Sampling (Requires pyDOE library: pip install pyDOE) ---
# This section is commented out as it requires pyDOE and replaces the original incomplete code.
# print(f"Generating {num_samples} samples using Latin Hypercube Sampling...")
# try:
#     from pyDOE import lhs
# except ImportError:
#     print("Error: pyDOE library not found. Cannot perform LHC sampling.")
#     print("Install it using: pip install pyDOE")
#     sys.exit(1)
#
# # Generate samples in the range [0, 1]
# lhc_samples_norm = lhs(nz, samples=num_samples)
# # Scale samples to the typical latent range [-1, 1]
# lhc_samples = (lhc_samples_norm * 2) - 1
#
# print(f"Generated {lhc_samples.shape[0]} LHC samples with dimension {lhc_samples.shape[1]}")
#
# # --- Generate Levels from Samples ---
# all_levels_list = []
# for i in range(num_samples):
#     latent_vector_np = lhc_samples[i]
#     latent_vector = torch.FloatTensor(latent_vector_np).view(1, nz, 1, 1) # Batch size 1
#
#     with torch.no_grad():
#         levels_tensor = generator(Variable(latent_vector))
#
#     levels_np = levels_tensor.data.cpu().numpy()
#     levels_cropped = levels_np[:, :, :output_height, :output_width]
#     level_indices = np.argmax(levels_cropped, axis=1)
#     all_levels_list.append(level_indices[0]) # Add the single level (H, W) to the list
#
# # Optional: Save levels (e.g., as JSON or individual images)
# print(f"Generated {len(all_levels_list)} level segments.")
# # Example: save first level as image
# if all_levels_list:
#     try:
#         import matplotlib.pyplot as plt
#         from pytorch.gan_optimize import tiles2image, combine_images_grid # Assuming these are available
#         first_level_img = tiles2image(all_levels_list[0])
#         plt.imsave('lhc_sample_0.png', first_level_img)
#         print("Saved sample image lhc_sample_0.png")
#     except ImportError:
#         print("Could not import helper functions or matplotlib to save image.")
#     except Exception as e:
#         print(f"Error saving sample image: {e}")

print("--- Original Code Snippet (Incomplete/Non-functional) ---")
# best = numpy.array(es.best.get()[0]) # 'es' is undefined
# latent_vector = torch.FloatTensor(x).view(batchSize, nz, 1, 1) # 'x' is undefined
# levels = generator(Variable(latent_vector, volatile=True))
# levels.data = levels.data[:, :, :14, :28]
# vutils.save_image(levels.data, 'generated_samples.png')
print("Original code was incomplete. LHC sampling part requires pyDOE library.")
print("Script finished.")
