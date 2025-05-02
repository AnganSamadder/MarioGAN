import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import sys

# --- Core StyleGAN Components ---

class PixelNorm(nn.Module):
    """Normalize features maps per pixel."""
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization."""
    def __init__(self, channels, style_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale_transform = nn.Linear(style_dim, channels)
        self.style_shift_transform = nn.Linear(style_dim, channels)

    def forward(self, image, style):
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(style)[:, :, None, None] # Add spatial dims
        style_shift = self.style_shift_transform(style)[:, :, None, None] # Add spatial dims
        transformed_image = style_scale * normalized_image + style_shift
        return transformed_image

class MappingNetwork(nn.Module):
    """Maps latent z to style vector w."""
    def __init__(self, z_dim, hidden_dim, w_dim, num_layers=8):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)
        self.to_w = nn.Linear(hidden_dim, w_dim) # Final layer to get w

    def forward(self, z):
        z = F.normalize(z, dim=1) # Normalize latent vector
        hidden = self.mapping(z)
        w = self.to_w(hidden)
        return w

class NoiseInjection(nn.Module):
    """Injects learnable noise."""
    def __init__(self, channels):
        super().__init__()
        # Learnable per-channel scaling factor
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, image):
        batch, _, height, width = image.shape
        noise = torch.randn(batch, 1, height, width, device=image.device)
        return image + self.weight * noise

class SynthesisBlock(nn.Module):
    """StyleGAN Generator Block."""
    def __init__(self, in_channels, out_channels, style_dim, use_upsample=True):
        super().__init__()
        self.use_upsample = use_upsample
        if use_upsample:
            # Corrected mode name
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.noise1 = NoiseInjection(out_channels)
        self.adain1 = AdaIN(out_channels, style_dim)
        self.leaky_relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.noise2 = NoiseInjection(out_channels)
        self.adain2 = AdaIN(out_channels, style_dim)
        self.leaky_relu2 = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.leaky_relu1(x)
        x = self.adain1(x, w)

        x = self.conv2(x)
        x = self.noise2(x)
        x = self.leaky_relu2(x)
        x = self.adain2(x, w)
        return x

class Generator(nn.Module):
    """StyleGAN Generator."""
    def __init__(self, z_dim, w_dim, out_channels, start_resolution=4, target_resolution=32):
        super().__init__()
        self.w_dim = w_dim
        self.mapping_network = MappingNetwork(z_dim, 512, w_dim) # Example hidden_dim=512

        # Calculate number of blocks needed
        num_blocks = int(math.log2(target_resolution) - math.log2(start_resolution))
        
        # Starting constant input
        self.start_const = nn.Parameter(torch.randn(1, 512, start_resolution, start_resolution))
        self.initial_adain1 = AdaIN(512, w_dim)
        self.initial_adain2 = AdaIN(512, w_dim)
        self.initial_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.initial_noise1 = NoiseInjection(512)
        self.initial_noise2 = NoiseInjection(512)
        
        # Simplified feature map progression (can be adjusted)
        features = [512, 512, 256, 128, 64, 32] 
        
        self.torgbs = nn.ModuleList()
        
        # Initialize in_channels (or in_feat) correctly before the loop
        in_feat = 512 # Start with the channel size after the initial block
        
        # --- Add debug print RIGHT BEFORE the loop ---
        # print(f"DEBUG SynthesisNetwork: Before loop, resolutions = {features}, type = {type(features)}", file=sys.stderr)
        # sys.stderr.flush()
        # --- End debug print ---

        blocks = []
        for i in range(num_blocks):
             out_feat = features[i+1] if (i+1) < len(features) else features[-1]
             blocks.append(SynthesisBlock(in_feat, out_feat, w_dim))
             in_feat = out_feat
        self.blocks = nn.ModuleList(blocks)

        # Final convolution to get desired output channels (e.g., tile types)
        self.to_out = nn.Conv2d(in_feat, out_channels, kernel_size=1) # 1x1 conv

    def forward(self, z, alpha=1.0, steps=0): # alpha/steps for progressive growing if used
        w = self.mapping_network(z)
        
        x = self.start_const.repeat(z.shape[0], 1, 1, 1) # Repeat const for batch size
        x = self.initial_noise1(x)
        x = self.initial_adain1(x, w)
        x = self.initial_conv(x)
        x = self.initial_noise2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.initial_adain2(x, w)

        for block in self.blocks:
            x = block(x, w)
            
        out = self.to_out(x)
        # No explicit activation like Tanh/Softmax needed if using appropriate loss (e.g., CrossEntropy on logits)
        return out

# --- Basic Discriminator --- (Can be replaced with more complex StyleGAN D)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(2) # Downsample

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.downsample(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels, target_resolution=32):
        super().__init__()
        
        # Simplified feature map progression
        features = [64, 128, 256, 512, 512] 
        num_blocks = int(math.log2(target_resolution) - math.log2(4)) # Down to 4x4

        self.from_in = nn.Conv2d(in_channels, features[0], kernel_size=1) # 1x1 conv
        
        blocks = []
        in_feat = features[0]
        for i in range(num_blocks):
             out_feat = features[i+1] if (i+1) < len(features) else features[-1]
             blocks.append(DiscriminatorBlock(in_feat, out_feat))
             in_feat = out_feat
        self.blocks = nn.Sequential(*blocks)

        # Final layers for 4x4 resolution
        self.final_conv = nn.Conv2d(in_feat, features[-1], kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        # Flatten and final linear layer for classification
        self.flatten = nn.Flatten()
        # Calculate flattened size: features[-1] * 4 * 4
        self.final_linear = nn.Linear(features[-1] * 4 * 4, 1) 

    def forward(self, x):
        x = self.leaky_relu(self.from_in(x))
        x = self.blocks(x)
        x = self.leaky_relu(self.final_conv(x))
        x = self.flatten(x)
        out = self.final_linear(x) # Output single logit
        return out 