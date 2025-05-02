import argparse
import torch
import time
import numpy as np
import os
import sys
import random
import math
import matplotlib.pyplot as plt

# Optional: Import thop for FLOPs calculation
try:
    from thop import profile
    from thop import clever_format
    thop_available = True
except ImportError:
    thop_available = False

# Assuming models.dcgan is importable
import models.dcgan as dcgan

parser = argparse.ArgumentParser(description='Time the generation process of a DCGAN model.')
parser.add_argument('--netG', required=True, help='Path to the pre-trained generator model (netG.pth)')
parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate for timing.')
parser.add_argument('--nz', type=int, default=32, help='Size of the latent z vector (must match the trained model).')
parser.add_argument('--ngf', type=int, default=64, help='Generator feature maps (must match the trained model).')
parser.add_argument('--features', type=int, default=10, help='Number of output features/tile types (must match the trained model).')
parser.add_argument('--imageSize', type=int, default=32, help='Internal image size (must match the trained model).')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Extra layers used in model (must match).')
parser.add_argument('--output_dir', default='timed_output', help='Directory to save generated samples (optional).')
parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA (run on CPU).')
parser.add_argument('--save_samples', action='store_true', help='Save the generated samples as images.')
parser.add_argument('--level_height', type=int, default=14, help='Target height of the level slice.')
parser.add_argument('--level_width', type=int, default=28, help='Target width of the level slice.')

args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device("cuda:0" if use_cuda else "cpu")
print(f"Using device: {device}")
if use_cuda:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

print(f"Loading generator from: {args.netG}")
# Assuming 1 GPU internally for DCGAN_G instantiation?
netG = dcgan.DCGAN_G(args.imageSize, args.nz, args.features, args.ngf, 1, args.n_extra_layers)
try:
    # Load the state dict, ignoring mismatches and preferring weights_only for security
    state_dict = torch.load(args.netG, map_location=device, weights_only=True)
    netG.load_state_dict(state_dict, strict=False)
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    print("Please ensure the model parameters (--nz, --ngf, etc.) match the loaded checkpoint.", file=sys.stderr)
    sys.exit(1)

netG.to(device)
netG.eval() # Set model to evaluation mode

print("Model loaded successfully.")

# Calculate Parameters and optionally FLOPs
model_params = 0
model_flops = 0
flops_str = "N/A (thop not installed)"
params_str = "N/A"

try:
    # Calculate trainable parameters
    model_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    params_str = f"{model_params:,}"

    if thop_available:
        dummy_input = torch.randn(1, args.nz, 1, 1, device=device)
        # Calculate FLOPs using thop
        flops, _ = profile(netG, inputs=(dummy_input,), verbose=False)
        model_flops = flops
        flops_str, _ = clever_format([model_flops, model_params], "%.3f")
    else:
        print("Warning: 'thop' library not found. Skipping FLOPs calculation.", file=sys.stderr)
        print("Install it via: pip install thop", file=sys.stderr)

except Exception as e:
    print(f"Warning: Could not calculate model stats. Error: {e}", file=sys.stderr)
    if not thop_available:
        flops_str = "N/A (thop not installed)"
    else:
        flops_str = "N/A (Calculation Error)"
    params_str = f"{model_params:,}" if model_params > 0 else "N/A (Calculation Error)"


if args.save_samples:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving generated samples to: {args.output_dir}")

preprocess_times = []
inference_times = []
postprocess_times = []

print(f"Starting generation of {args.num_samples} samples for timing...")

for i in range(args.num_samples):
    # 1. Pre-processing: Generate noise
    start_time = time.perf_counter()
    noise = torch.randn(1, args.nz, 1, 1, device=device) # Generate 1 sample at a time
    if use_cuda:
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    preprocess_times.append(end_time - start_time)

    # 2. Inference: Run model
    start_time = time.perf_counter()
    with torch.no_grad():
        output_tensor = netG(noise)
    if use_cuda:
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    inference_times.append(end_time - start_time)

    # 3. Post-processing: Crop and convert
    start_time = time.perf_counter()
    # Crop to desired level size
    output_tensor = output_tensor[:, :, :args.level_height, :args.level_width]
    # Convert to level representation (argmax)
    level_representation = torch.argmax(output_tensor, dim=1) # Get tile index
    level_representation_cpu = level_representation.cpu().numpy() # shape (1, H, W)
    if use_cuda:
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    postprocess_times.append(end_time - start_time)

    # Optional: Save Sample as image
    if args.save_samples:
        try:
            img_data = level_representation_cpu.squeeze(0)
            # Normalize for colormap
            img_normalized = img_data / float(args.features)
            # Apply colormap
            img_colored = plt.get_cmap('rainbow')(img_normalized)
            # Save the image (RGB only)
            plt.imsave(os.path.join(args.output_dir, f'timed_sample_{i:04d}.png'), img_colored[:, :, :3])
        except Exception as e:
            print(f"Warning: Could not save sample {i}. Error: {e}", file=sys.stderr)


# Calculate and Print Results
avg_preprocess = np.mean(preprocess_times)
avg_inference = np.mean(inference_times)
avg_postprocess = np.mean(postprocess_times)
avg_total = avg_preprocess + avg_inference + avg_postprocess

std_preprocess = np.std(preprocess_times)
std_inference = np.std(inference_times)
std_postprocess = np.std(postprocess_times)
std_total = np.std([sum(x) for x in zip(preprocess_times, inference_times, postprocess_times)])

samples_per_sec = 1.0 / avg_total if avg_total > 0 else float('inf')

print("--- Timing Results ---")
print(f"Generated {args.num_samples} samples.")
print(f"Device: {device}")
print(f"Generator: {args.netG}")
print("Average Time per Sample (seconds):")
print(f"  Pre-processing:  {avg_preprocess:.6f} +/- {std_preprocess:.6f}")
print(f"  Inference:       {avg_inference:.6f} +/- {std_inference:.6f}")
print(f"  Post-processing: {avg_postprocess:.6f} +/- {std_postprocess:.6f}")
print(f"  -----------------------------+")
print(f"  Total:           {avg_total:.6f} +/- {std_total:.6f}")
print(f"Throughput: {samples_per_sec:.2f} samples/second")

print("--- Model Statistics ---")
print(f"Trainable Parameters: {params_str}")
print(f"Estimated FLOPs (forward pass): {flops_str}")

# Identify Bottleneck
times = {"Pre-processing": avg_preprocess, "Inference": avg_inference, "Post-processing": avg_postprocess}
bottleneck_stage = max(times, key=times.get)
bottleneck_time = times[bottleneck_stage]
total_time_check = sum(times.values())

print(f"Bottleneck Analysis:")
if abs(total_time_check - avg_total) > 1e-5:
     print(f"  Warning: Sum of averages ({total_time_check:.6f}) differs slightly from calculated total average ({avg_total:.6f}).", file=sys.stderr)

if avg_total > 0:
    preprocess_perc = (avg_preprocess / avg_total) * 100
    inference_perc = (avg_inference / avg_total) * 100
    postprocess_perc = (avg_postprocess / avg_total) * 100
    print(f"  Major Bottleneck: {bottleneck_stage} ({bottleneck_time:.6f}s, {times[bottleneck_stage]/avg_total*100:.1f}% of total time)")
    print(f"  Time Distribution: Pre: {preprocess_perc:.1f}%, Infer: {inference_perc:.1f}%, Post: {postprocess_perc:.1f}%" )
else:
    print("  Total time was zero, cannot determine bottleneck percentage.")

print("Script finished.") 