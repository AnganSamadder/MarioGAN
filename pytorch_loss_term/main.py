import torch.nn.functional as F
import os
import sys # Added
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable # Keep if model needs Variable
import argparse
import random
import json
import math
import matplotlib
matplotlib.use('Agg') # Use non-GUI backend BEFORE importing pyplot
import matplotlib.pyplot as plt

# Imports from original file that might be needed
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
# import torchvision.datasets as dset # Unused
# import torchvision.transforms as transforms # Unused
import torchvision.utils as vutils

# Model imports (assuming they exist in ./models relative to this script)
import models.dcgan as dcgan
# import models.mlp as mlp # Keep commented

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="PyTorch DCGAN Training with Custom Loss Terms")

# Core GAN parameters
parser.add_argument('--nz', type=int, default=32, help='Size of latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='Generator features')
parser.add_argument('--ndf', type=int, default=64, help='Discriminator features')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Extra layers in G and D')

# Training parameters
parser.add_argument('--batchSize', type=int, default=32, help='Input batch size')
parser.add_argument('--niter', type=int, default=5000, help='Number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='Learning rate for Critic (Discriminator)')
parser.add_argument('--lrG', type=float, default=0.00005, help='Learning rate for Generator')
parser.add_argument('--adam', action='store_true', help='Use Adam optimizer (default is RMSprop)')
parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
parser.add_argument('--Diters', type=int, default=5, help='Number of D iterations per G iteration')
parser.add_argument('--clamp_lower', type=float, default=-0.01, help='WGAN weight clamp lower bound')
parser.add_argument('--clamp_upper', type=float, default=0.01, help='WGAN weight clamp upper bound')

# Infrastructure
parser.add_argument('--cuda', action='store_true', help='Enable CUDA training')
parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('--netG', default='', help="Path to pre-trained netG (to continue training)")
parser.add_argument('--netD', default='', help="Path to pre-trained netD (to continue training)")
parser.add_argument('--experiment', default='samples', help='Directory for samples/models (relative to script)')
# parser.add_argument('--problem', type=int, default=0, help='Level examples index (IGNORED - uses example.json)')

# Custom Loss Term Weights & Parameters
parser.add_argument('--diversity_weight', type=float, default=0.5, help='Weight for target distribution diversity loss')
parser.add_argument('--floating_enemy_weight', type=float, default=0.2, help='Weight for floating enemy penalty')
parser.add_argument('--base_pipe_penalty_weight', type=float, default=0.1, help='Base weight for pipe correctness/presence penalties')
parser.add_argument('--penalty_start_epoch', type=int, default=500, help='Epoch to start applying pipe penalties')
parser.add_argument('--pipe_presence_threshold', type=float, default=0.1, help='Min expected pipe tops per level for presence penalty')

opt = parser.parse_args()
print(opt)

# --- Setup Directories --- 
script_dir = os.path.dirname(os.path.abspath(__file__))

if not os.path.isabs(opt.experiment):
    opt.experiment = os.path.join(script_dir, opt.experiment)
print(f"Experiment results will be saved to: {opt.experiment}")
os.makedirs(opt.experiment, exist_ok=True)

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

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, consider running with --cuda")

# --- Constants and Data Loading --- 
map_size = 32 # Target size for model input
z_dims = 10 # Number of different tile types

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
X_onehot = np.eye(z_dims, dtype='uint8')[X]
X_onehot = np.rollaxis(X_onehot, 3, 1) # Shape: (N, C, H, W)
print(f"One-hot shape: {X_onehot.shape}")

# Pad levels to map_size x map_size
X_train = np.zeros((X.shape[0], z_dims, map_size, map_size), dtype='float32')
# Define tile IDs based on common conventions and inspection of data/code
ID_EMPTY = 2          # Assuming index 2 is empty based on padding
ID_GROUND = 0
ID_SOLID_BLOCK = 1
ID_QUESTION_BLOCK = 4
ID_GOOMBA = 3         # Example enemy ID
ID_PIPE_TOP_LEFT = 6
ID_PIPE_TOP_RIGHT = 7
ID_PIPE_LEFT = 8
ID_PIPE_RIGHT = 9
ENEMY_TILE_IDS = [ID_GOOMBA] # Add others if needed
BLOCKING_TILE_IDS = [ID_GROUND, ID_SOLID_BLOCK, ID_QUESTION_BLOCK]

# Fill background with one-hot empty space
X_train[:, ID_EMPTY, :, :] = 1.0

h_orig, w_orig = X.shape[1], X.shape[2]
h_padded = min(h_orig, map_size)
w_padded = min(w_orig, map_size)
X_train[:, :, :h_padded, :w_padded] = X_onehot[:, :, :h_padded, :w_padded]

print(f"Padded training data shape: {X_train.shape}")
num_batches = int(np.floor(X_train.shape[0] / opt.batchSize))
if X_train.shape[0] % opt.batchSize != 0:
    print(f"Warning: {X_train.shape[0] % opt.batchSize} examples will be dropped each epoch due to batch size.")
print(f"Number of batches per epoch: {num_batches}")

# --- Target Tile Distribution for Diversity Loss ---
# Must sum to 1.0. Use fractions based on analysis or desired output.
TARGET_TILE_DISTRIBUTION = {
    ID_EMPTY: 0.70,
    ID_GROUND: 0.10,
    ID_SOLID_BLOCK: 0.05,
    ID_QUESTION_BLOCK: 0.03,
    ID_PIPE_TOP_LEFT: 0.01,
    ID_PIPE_TOP_RIGHT: 0.01,
    ID_PIPE_LEFT: 0.02,
    ID_PIPE_RIGHT: 0.02,
    ID_GOOMBA: 0.02,
    # Add ID for tile 5 if needed, or adjust others
    # Tile 5 (assuming it exists): 0.04 # Example placeholder
}
# Verify sum is close to 1.0 (adjust as needed)
target_sum = sum(TARGET_TILE_DISTRIBUTION.values())
if not math.isclose(target_sum, 1.0):
    print(f"Warning: TARGET_TILE_DISTRIBUTION sums to {target_sum}, not 1.0. Normalizing.")
    # Normalize
    normalized_dist = {k: v / target_sum for k, v in TARGET_TILE_DISTRIBUTION.items()}
    TARGET_TILE_DISTRIBUTION = normalized_dist

# Convert target distribution dict to a tensor for faster lookup
# Ensure all z_dims (0-9) are covered, assign 0 prob if missing
target_prob_list = [TARGET_TILE_DISTRIBUTION.get(i, 0.0) for i in range(z_dims)]
target_probs_tensor = torch.tensor(target_prob_list, dtype=torch.float32)
if opt.cuda:
    target_probs_tensor = target_probs_tensor.cuda()
# ---------------------------------------------------

# --- Model Initialization --- 
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
n_extra_layers = int(opt.n_extra_layers)

def weights_init(m):
    """Custom weights initialization."""
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
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

# --- Visualization Utilities ---
def tiles2image(tiles):
    """Converts tile indices to an RGB image using a colormap."""
    norm_tiles = tiles / float(z_dims) if z_dims > 0 else tiles
    colored_image = plt.get_cmap('rainbow')(norm_tiles)
    return colored_image[..., :3] # RGB

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

# --- CUDA Setup --- 
if opt.cuda:
    netD.cuda()
    netG.cuda()
    input_tensor = input_tensor.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise_tensor, fixed_noise = noise_tensor.cuda(), fixed_noise.cuda()
    target_probs_tensor = target_probs_tensor.cuda() # Move target dist to GPU too

# --- Optimizers --- 
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    print("Using Adam optimizer")
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
    print("Using RMSprop optimizer")

# --- Loss Helper Functions ---
def prob_at(prob_map, y, x, h, w):
    """ Safely get probability at (y, x), return 0 if out of bounds."""
    if 0 <= y < h and 0 <= x < w:
        return prob_map[:, y, x]
    else:
        return torch.zeros_like(prob_map[:, 0, 0]) # Return tensor of zeros with same batch/device

def prob_below(prob_map, y, x, h, w): return prob_at(prob_map, y + 1, x, h, w)
def prob_above(prob_map, y, x, h, w): return prob_at(prob_map, y - 1, x, h, w)
def prob_left(prob_map, y, x, h, w): return prob_at(prob_map, y, x - 1, h, w)
def prob_right(prob_map, y, x, h, w): return prob_at(prob_map, y, x + 1, h, w)

def calculate_pipe_penalty(level_logits, penalty_weight):
    """Calculates penalty for incorrectly formed pipes.

    Args:
        level_logits: Tensor (Batch, Channel, Height, Width) - Raw output from Generator.
        penalty_weight: Scalar weight for this loss term.

    Returns:
        Scalar tensor representing the pipe penalty loss.
    """
    if penalty_weight <= 0:
        return torch.tensor(0.0, device=level_logits.device)

    # Use softmax to get probabilities for each tile type at each location
    probs = F.softmax(level_logits, dim=1)
    b, c, h, w = probs.shape

    total_penalty = torch.tensor(0.0, device=level_logits.device)

    # Iterate through each cell in the grid (consider vectorizing later if slow)
    for y in range(h):
        for x in range(w):
            # --- Penalty 1: Pipe top must have empty above --- 
            prob_pipe_top_l = prob_at(probs[:, ID_PIPE_TOP_LEFT, :, :], y, x, h, w)
            prob_pipe_top_r = prob_at(probs[:, ID_PIPE_TOP_RIGHT, :, :], y, x, h, w)
            prob_pipe_top = prob_pipe_top_l + prob_pipe_top_r # Probability it's any pipe top

            # Prob it's NOT empty above
            prob_not_empty_above = 1.0 - prob_above(probs[:, ID_EMPTY, :, :], y, x, h, w)
            # Penalty = P(PipeTop) * P(NotEmptyAbove)
            penalty1 = prob_pipe_top * prob_not_empty_above
            total_penalty += penalty1.sum() # Sum over batch

            # --- Penalty 2: Pipe body must have pipe body or top below --- 
            prob_pipe_body_l = prob_at(probs[:, ID_PIPE_LEFT, :, :], y, x, h, w)
            prob_pipe_body_r = prob_at(probs[:, ID_PIPE_RIGHT, :, :], y, x, h, w)
            prob_pipe_body = prob_pipe_body_l + prob_pipe_body_r

            # Prob it IS a valid pipe part below
            prob_valid_pipe_below = prob_below(probs[:, ID_PIPE_LEFT, :, :], y, x, h, w) + \
                                    prob_below(probs[:, ID_PIPE_RIGHT, :, :], y, x, h, w) + \
                                    prob_below(probs[:, ID_PIPE_TOP_LEFT, :, :], y, x, h, w) + \
                                    prob_below(probs[:, ID_PIPE_TOP_RIGHT, :, :], y, x, h, w)
            # Prob it is NOT a valid pipe part below
            prob_invalid_below = 1.0 - prob_valid_pipe_below
            # Penalty = P(PipeBody) * P(InvalidBelow)
            penalty2 = prob_pipe_body * prob_invalid_below
            total_penalty += penalty2.sum()

            # --- Penalty 3: Pipe body must have pipe body or ground directly below edge --- 
            # Check only if it's a pipe body, needs ground directly at bottom
            if y == h - 1: # If on the bottom row
                 # Penalty = P(PipeBody) * P(NotGroundDirectlyBelow - which is 1 here)
                 penalty3 = prob_pipe_body # Penalize pipe bodies ending mid-air at bottom
                 total_penalty += penalty3.sum()

            # --- Penalty 4: Pipe top should not have blocking tiles directly above --- 
            prob_blocking_above = torch.zeros_like(prob_pipe_top)
            for block_id in BLOCKING_TILE_IDS:
                 prob_blocking_above += prob_above(probs[:, block_id, :, :], y, x, h, w)
            # Penalty = P(PipeTop) * P(BlockingTileAbove)
            penalty4 = prob_pipe_top * prob_blocking_above
            total_penalty += penalty4.sum()

    # Average penalty over batch and grid size
    average_penalty = total_penalty / (b * h * w)
    return average_penalty * penalty_weight

def calculate_pipe_presence_penalty(level_logits, presence_threshold, penalty_weight):
    """Calculates penalty if the expected number of pipe tops is below a threshold."""
    if penalty_weight <= 0:
        return torch.tensor(0.0, device=level_logits.device)
        
    probs = F.softmax(level_logits, dim=1)
    b, c, h, w = probs.shape
    
    # Probability of pipe top (left or right) at each location
    prob_pipe_top = probs[:, ID_PIPE_TOP_LEFT, :, :] + probs[:, ID_PIPE_TOP_RIGHT, :, :]
    
    # Expected number of pipe tops per level in the batch
    expected_pipe_tops = torch.sum(prob_pipe_top, dim=[1, 2]) # Sum over H and W
    
    # Calculate penalty for levels below the threshold
    # Penalty = max(0, threshold - expected_count)
    # We use the target number of pipes based on the threshold and grid size
    target_num_pipes = presence_threshold * h * w 
    penalty = F.relu(target_num_pipes - expected_pipe_tops) 
    
    # Average penalty over the batch
    average_penalty = penalty.mean()
    
    return average_penalty * penalty_weight

def calculate_diversity_loss(level_logits, target_distribution_tensor, diversity_weight):
    """Calculates MSE loss between generated tile distribution and target distribution."""
    if diversity_weight <= 0:
        return torch.tensor(0.0, device=level_logits.device)

    probs = F.softmax(level_logits, dim=1)
    b, c, h, w = probs.shape

    # Calculate average probability for each tile type across the spatial dimensions (H, W)
    # Result shape: (Batch, Channel)
    avg_spatial_probs = torch.mean(probs, dim=[2, 3])

    # Calculate average distribution across the batch
    # Result shape: (Channel)
    avg_batch_probs = torch.mean(avg_spatial_probs, dim=0)

    # Calculate MSE loss between the average generated distribution and the target
    # Target tensor needs to be on the same device
    loss_mse = F.mse_loss(avg_batch_probs, target_distribution_tensor)

    return loss_mse * diversity_weight

def calculate_floating_enemy_penalty(level_logits, penalty_weight):
    """Calculates penalty for enemies potentially placed over empty space."""
    if penalty_weight <= 0:
        return torch.tensor(0.0, device=level_logits.device)

    probs = F.softmax(level_logits, dim=1)
    b, c, h, w = probs.shape

    total_penalty = torch.tensor(0.0, device=level_logits.device)

    for enemy_id in ENEMY_TILE_IDS:
        prob_enemy = probs[:, enemy_id, :, :] # Shape (B, H, W)
        # Probability of NOT having solid ground below (approximated by P(EmptyBelow))
        # Need to be careful with edges
        prob_empty_below_shifted = F.pad(probs[:, ID_EMPTY, :-1, :], (0, 0, 1, 0)) # Shift empty probs up
        # Penalty = P(Enemy) * P(EmptyBelow)
        penalty_enemy = prob_enemy * prob_empty_below_shifted
        total_penalty += penalty_enemy.sum() # Sum over batch, H, W

    # Average penalty over batch and grid size
    average_penalty = total_penalty / (b * h * w)
    return average_penalty * penalty_weight

# --- Training Loop --- 
gen_iterations = 0
print(f"Starting training for {opt.niter} epochs...")
for epoch in range(opt.niter):
    # Shuffle training data each epoch
    X_train = X_train[np.random.permutation(len(X_train))]

    batch_idx = 0
    while batch_idx < num_batches:

        # --- (1) Update Discriminator (Critic) --- #
        Diters = opt.Diters
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100

        d_iter = 0
        errD = torch.tensor(0.0) # Initialize
        errD_real = torch.tensor(0.0)
        errD_fake = torch.tensor(0.0)
        while d_iter < Diters and batch_idx < num_batches:
            d_iter += 1

            # Clamp D weights
            for p in netD.parameters(): p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            # Get real data batch
            start_idx = batch_idx * opt.batchSize
            end_idx = start_idx + opt.batchSize
            data = X_train[start_idx:end_idx]
            batch_idx += 1

            netD.zero_grad()

            real_cpu = torch.FloatTensor(data)
            if opt.cuda: real_cpu = real_cpu.cuda()

            current_batch_size = real_cpu.size(0) # Actual batch size
            input_tensor_batch = input_tensor[:current_batch_size]
            input_tensor_batch.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input_tensor_batch)

            # Train D with real
            errD_real_batch = netD(inputv)
            errD_real_batch.backward(one)

            # Train D with fake
            noise_tensor_batch = noise_tensor[:current_batch_size]
            noise_tensor_batch.resize_(current_batch_size, nz, 1, 1).normal_(0, 1)
            with torch.no_grad():
                noisev = Variable(noise_tensor_batch)
                fake = netG(noisev).detach()

            inputv = Variable(fake)
            errD_fake_batch = netD(inputv)
            errD_fake_batch.backward(mone)

            errD_batch = errD_real_batch - errD_fake_batch
            optimizerD.step()
            
            # Accumulate last batch losses for reporting
            if d_iter == Diters:
                 errD = errD_batch
                 errD_real = errD_real_batch
                 errD_fake = errD_fake_batch

        # --- (2) Update Generator --- #
        for p in netD.parameters(): p.requires_grad = False # Freeze D
        netG.zero_grad()

        # Generate noise for G update (use full batch size)
        noise_tensor.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise_tensor)
        fake = netG(noisev) # fake now holds the raw logits (B, C, H, W)

        # --- Calculate Generator Losses --- 
        # Base WGAN loss for G (wants to fool D)
        errG_base = netD(fake)

        # Calculate dynamic penalty weight for pipes
        current_pipe_penalty_weight = 0.0
        if epoch >= opt.penalty_start_epoch:
            current_pipe_penalty_weight = opt.base_pipe_penalty_weight
            # Optional: Schedule the weight (e.g., linear increase)
            # progress = (epoch - opt.penalty_start_epoch) / (opt.niter - opt.penalty_start_epoch)
            # current_pipe_penalty_weight = opt.base_pipe_penalty_weight * progress

        # Pipe Correctness Penalty
        pipe_correctness_penalty = calculate_pipe_penalty(fake, current_pipe_penalty_weight)
        
        # Pipe Presence Penalty (only applies if weight > 0)
        pipe_presence_penalty = calculate_pipe_presence_penalty(fake, opt.pipe_presence_threshold, current_pipe_penalty_weight)

        # Diversity Loss
        diversity_loss = calculate_diversity_loss(fake, target_probs_tensor, opt.diversity_weight)

        # Floating Enemy Penalty
        floating_enemy_penalty = calculate_floating_enemy_penalty(fake, opt.floating_enemy_weight)

        # Total Generator Loss
        # G wants to MAXIMIZE errG_base (minimize -errG_base), and minimize penalties/losses
        total_errG = -errG_base + pipe_correctness_penalty + pipe_presence_penalty + diversity_loss + floating_enemy_penalty

        # Backpropagate total loss
        total_errG.backward()
        optimizerG.step()
        gen_iterations += 1

        # Re-enable D gradients for the next D iteration cycle
        for p in netD.parameters(): p.requires_grad = True

        # --- Logging & Visualization --- #
        if batch_idx % 50 == 0: # Log every 50 D batches
            print('[%d/%d][%d/%d][%d] Loss_D: %.4f Loss_G_Base: %.4f ' 
                  'P_PipeC: %.4f P_PipeP: %.4f L_Div: %.4f P_EnemyF: %.4f'
                  % (epoch, opt.niter, batch_idx, num_batches, gen_iterations,
                     errD.item(), errG_base.item(), 
                     pipe_correctness_penalty.item(), 
                     pipe_presence_penalty.item(), 
                     diversity_loss.item(),
                     floating_enemy_penalty.item()))

        if gen_iterations % 100 == 0: # Generate sample images every 100 G iterations
            with torch.no_grad():
                fixed_noise_v = Variable(fixed_noise)
                fake_fixed = netG(fixed_noise_v).data.cpu().numpy()

            # Process generated images
            fake_fixed_cropped = fake_fixed[:, :, :h_orig, :w_orig] # Crop
            fake_tiles = np.argmax(fake_fixed_cropped, axis=1)
            img_grid = combine_images_grid(tiles2image(fake_tiles))

            save_path = os.path.join(opt.experiment, f'fake_samples_iter_{gen_iterations:06d}.png')
            try:
                plt.imsave(save_path, img_grid)
            except Exception as e:
                print(f"Warning: Failed to save sample image to {save_path}: {e}", file=sys.stderr)

    # --- End of Epoch --- #
    
    # Save checkpoint periodically and at the end
    if (epoch + 1) % 50 == 0 or (epoch + 1) == opt.niter:
        epoch_num_str = f"{epoch+1}_{int(opt.diversity_weight*10)}_{int(opt.base_pipe_penalty_weight*100)}_{nz}"
        g_save_path = os.path.join(model_save_dir, f'netG_epoch_{epoch_num_str}.pth')
        d_save_path = os.path.join(model_save_dir, f'netD_epoch_{epoch_num_str}.pth')
        print(f"---> Checkpointing models at epoch {epoch + 1} <---")
        try:
            torch.save(netG.state_dict(), g_save_path)
            torch.save(netD.state_dict(), d_save_path)
            print(f"Saved checkpoints: {g_save_path}, {d_save_path}")
            # Save a final marker too
            if (epoch + 1) == opt.niter:
                 torch.save(netG.state_dict(), os.path.join(model_save_dir, f'netG_final_{epoch_num_str}.pth'))
                 torch.save(netD.state_dict(), os.path.join(model_save_dir, f'netD_final_{epoch_num_str}.pth'))
                 print("Saved final models.")
        except Exception as e:
            print(f"Error saving checkpoint/final models for epoch {epoch + 1}: {e}", file=sys.stderr)

print("--- Training Finished --- ")