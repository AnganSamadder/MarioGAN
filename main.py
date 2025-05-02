from __future__ import print_function
import matplotlib
matplotlib.use('Agg') # Use non-GUI backend BEFORE importing pyplot
import matplotlib.pyplot as plt

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
import math

import models.dcgan as dcgan
import json

parser = argparse.ArgumentParser()
# ... (rest of parser arguments)
opt = parser.parse_args()
print(opt)

# ... (potential setup code like mkdir, random seed, cudnn.benchmark)

# --- Data Loading ---
# ... (transforms.Compose, dset.ImageFolder or custom dataset, DataLoader)

# --- Model Initialization ---
# ... (dcgan.weights_init, netG, netD)
# ... (loading checkpoints if opt.netG or opt.netD are provided)

# --- Loss & Optimizers ---
# ... (criterion, optimizerD, optimizerG)


# --- Training Loop ---
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        # ... (prepare real data, noise)

        # (1) Update D network
        # ... (D update code: zero_grad, forward pass real, backward, forward pass fake, backward, step)
        optimizerD.step()

        # (2) Update G network
        for p in netD.parameters():
            p.requires_grad = False # Avoid computation for D when updating G
        netG.zero_grad()

        # Ensure full batch of noise
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)

        # Generate fake data
        fake_output, _ = netG(noise) # Assuming G returns (output, maybe_attention)

        # Calculate G loss based on D's evaluation of fake data
        errG = netD(fake_output)
        errG.backward(one) # Assuming 'one' is defined for loss backward pass
        optimizerG.step()

        # Output training stats
        # ... (print losses, maybe save images/checkpoints)

    # Save checkpoint per epoch (optional)
    # torch.save(...) if epoch % opt.saveInterval == 0: 