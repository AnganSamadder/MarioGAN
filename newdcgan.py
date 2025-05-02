import torch
import torch.nn as nn
import torch.nn.parallel
import math # Ensure math is imported

# +++ ADD THIS FUNCTION +++
def generate_normalized_coords(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Generates a 2-channel tensor representing normalized (0 to 1) coordinates.
    Channel 0: y-coordinates (normalized height)
    Channel 1: x-coordinates (normalized width)

    Args:
        h: Height of the grid.
        w: Width of the grid.
        device: The torch device (e.g., 'cuda' or 'cpu').

    Returns:
        A tensor of shape (2, h, w) with normalized coordinates.
    """
    # Using 0 to 1 range as in the previous example
    y_coords = torch.linspace(0., 1., steps=h, device=device)
    x_coords = torch.linspace(0., 1., steps=w, device=device)
    y_grid = y_coords.unsqueeze(1).repeat(1, w)
    x_grid = x_coords.unsqueeze(0).repeat(h, 1)
    coord_grid = torch.stack((y_grid, x_grid), dim=0)
    return coord_grid.float()
# +++ END OF ADDED FUNCTION +++


class DCGAN_D(nn.Module):
    # --- MODIFIED __init__ for DCGAN_D ---
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        self.nc = nc # Store original channel count if needed
        self.isize = isize # Store input size
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # --- Define layers individually or in smaller blocks ---
        # +++ Input channel changed from nc to nc + 2 +++
        self.initial_conv = nn.Conv2d(nc + 2, ndf, 4, 2, 1, bias=False)
        self.initial_relu = nn.LeakyReLU(0.2, inplace=True)

        # --- Build the rest of the network in a separate Sequential ---
        self.main_rest = nn.Sequential()
        csize, cndf = isize / 2, ndf

        # Extra layers (Copied from original)
        for t in range(n_extra_layers):
            self.main_rest.add_module(f'extra_layers_{t}_{cndf}_conv',
                                      nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            self.main_rest.add_module(f'extra_layers_{t}_{cndf}_batchnorm',
                                      nn.BatchNorm2d(cndf))
            self.main_rest.add_module(f'extra_layers_{t}_{cndf}_relu',
                                      nn.LeakyReLU(0.2, inplace=True))

        # Pyramid layers (Copied from original)
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            self.main_rest.add_module(f'pyramid_{in_feat}-{out_feat}_conv',
                                      nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            self.main_rest.add_module(f'pyramid_{out_feat}_batchnorm',
                                      nn.BatchNorm2d(out_feat))
            self.main_rest.add_module(f'pyramid_{out_feat}_relu',
                                      nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize // 2

        # Final layer (Copied from original)
        # state size. K x 4 x 4
        self.main_rest.add_module(f'final_{cndf}-1_conv',
                                  nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))

    # --- MODIFIED forward for DCGAN_D ---
    def forward(self, input):
        # input shape: (batch_size, nc, isize, isize)
        batch_size, _, h, w = input.shape
        device = input.device

        # 1. Generate and Prepare PE grid
        # Ensure h, w match self.isize if fixed size expected
        if h != self.isize or w != self.isize:
             print(f"Warning: DCGAN_D input size ({h}x{w}) doesn't match expected ({self.isize}x{self.isize})", file=sys.stderr)
             # Decide how to handle this: error, resize, or proceed? Assuming proceed for now.
             pe_grid = generate_normalized_coords(h, w, device)
        else:
             pe_grid = generate_normalized_coords(self.isize, self.isize, device)

        # Expand PE grid -> (batch_size, 2, h, w)
        pe_grid_batch = pe_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 2. Concatenate input and PE grid
        # Output shape: (batch_size, nc + 2, h, w)
        x_with_pe = torch.cat((input, pe_grid_batch), dim=1)

        # 3. Pass through initial layers
        x = self.initial_conv(x_with_pe)
        x = self.initial_relu(x)

        # 4. Pass through the rest of the main network
        # --- Handle multi-gpu - Simplified: Assuming single GPU or external wrapper ---
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            # Direct data_parallel on self.main_rest might work if input 'x' is already on multiple GPUs
            # but it's safer to handle parallelism outside or use DDP.
            # For simplicity, running non-parallel here. Add parallel logic if needed.
            output = self.main_rest(x)
        else:
            output = self.main_rest(x)

        # 5. Final processing (as before)
        output = output.mean(0)
        return output.view(1)


class DCGAN_G(nn.Module):
    # --- MODIFIED __init__ for DCGAN_G ---
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        self.isize = isize
        self.nz = nz
        self.nc = nc # Output channels (tile types)
        self.ngf = ngf

        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # --- Calculate cngf and initial size ---
        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2
        self.cngf = cngf # Channels after initial block

        # --- Store initial H/W for PE generation ---
        self.h_small = 4
        self.w_small = 4

        # --- Define layer blocks separately ---
        # Block 1: Initial layers from nz to cngf x H_small x W_small
        self.initial_block = nn.Sequential(
            nn.ConvTranspose2d(nz, self.cngf, self.h_small, 1, 0, bias=False),
            nn.BatchNorm2d(self.cngf),
            nn.ReLU(True)
        )

        # Block 2: Pyramid layers
        self.pyramid_blocks = nn.ModuleList()
        csize_current = self.h_small
        cngf_current = self.cngf
        # +++ Input channels for the first pyramid block modified +++
        in_channels_pyramid = cngf_current + 2 # Add 2 PE channels
        while csize_current < isize // 2:
            out_channels_pyramid = cngf_current // 2
            self.pyramid_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels_pyramid, out_channels_pyramid, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels_pyramid),
                    nn.ReLU(True)
                )
            )
            cngf_current = out_channels_pyramid
            csize_current = csize_current * 2
            # Input for subsequent pyramid blocks is the output of the previous one
            in_channels_pyramid = cngf_current

        self.last_cngf_after_pyramid = cngf_current # Store channels after pyramid

        # Block 3: Extra layers block (if any)
        self.extra_layers_block = nn.Sequential()
        cngf_extra_in = self.last_cngf_after_pyramid
        # +++ Check if pyramid block was skipped; if so, extra layers need PE +++
        needs_pe_input = not self.pyramid_blocks # True if list is empty
        for t in range(n_extra_layers):
            in_channels_extra = cngf_extra_in + 2 if (t == 0 and needs_pe_input) else cngf_extra_in
            # Assuming extra layers use Conv2d and don't change channels
            self.extra_layers_block.add_module(f'extra_layers_{t}_{cngf_extra_in}_conv',
                                               nn.Conv2d(in_channels_extra, cngf_extra_in, 3, 1, 1, bias=False))
            self.extra_layers_block.add_module(f'extra_layers_{t}_{cngf_extra_in}_batchnorm',
                                               nn.BatchNorm2d(cngf_extra_in))
            self.extra_layers_block.add_module(f'extra_layers_{t}_{cngf_extra_in}_relu',
                                               nn.ReLU(True))
            # Input for next extra layer is output of current
            in_channels_extra = cngf_extra_in

        self.last_cngf_after_extra = cngf_extra_in # Store channels after extra layers

        # Block 4: Final block
        cngf_final_in = self.last_cngf_after_extra
        # +++ Check if previous blocks were skipped; if so, final layer needs PE +++
        needs_pe_input_final = not self.pyramid_blocks and n_extra_layers == 0
        in_channels_final = cngf_final_in + 2 if needs_pe_input_final else cngf_final_in
        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels_final, nc, 4, 2, 1, bias=False),
            # Using ReLU based on original code/paper. Consider nn.Tanh() or nn.Sigmoid()
            # or nn.Softmax(dim=1) depending on desired output interpretation/range.
            nn.ReLU(True)
            # nn.Softmax(dim=1) # Example alternative
        )

    # --- MODIFIED forward for DCGAN_G ---
    def forward(self, input):
        # input shape: (batch_size, nz, 1, 1)
        batch_size = input.shape[0]
        device = input.device

        # 1. Pass through initial block
        # Output shape: (B, cngf, H_small, W_small) -> (B, cngf, 4, 4)
        initial_features = self.initial_block(input)

        # 2. Generate and Prepare PE grid for the small dimensions (H_small x W_small)
        pe_grid_small = generate_normalized_coords(self.h_small, self.w_small, device)
        # Expand PE grid -> (batch_size, 2, H_small, W_small)
        pe_grid_batch = pe_grid_small.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # 3. Concatenate initial features and PE grid
        # Output shape: (batch_size, cngf + 2, H_small, W_small)
        features_with_pe = torch.cat((initial_features, pe_grid_batch), dim=1)

        # 4. Pass through subsequent blocks, handling potential PE need
        x = features_with_pe # Start with PE-enhanced features

        # Pass through pyramid blocks (first layer expects PE input)
        if self.pyramid_blocks:
            for i, block in enumerate(self.pyramid_blocks):
                x = block(x)
        else:
            # If no pyramid blocks, x still contains PE for the *next* block
            pass

        # Pass through extra layers block
        if self.extra_layers_block:
            x = self.extra_layers_block(x)
        else:
            # If no extra blocks, x might still contain PE for the *final* block
            pass

        # Pass through final block
        output = self.final_block(x)
        # output shape: (batch_size, nc, isize, isize)

        # --- Handle multi-gpu - Simplified ---
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            # Parallelism needs careful handling with this structure.
            # Simplest is single GPU or DDP. Return non-parallel output.
            pass # Output is already calculated

        return output


###############################################################################
# --- DCGAN_D_nobn and DCGAN_G_nobn are NOT modified. ---
# --- Apply similar refactoring if you need the no-batchnorm versions. ---
class DCGAN_D_nobn(nn.Module):
    # ... (Original code - needs refactoring similar to DCGAN_D if used) ...
    def __init__(self, isize, nz, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        # input is nc x isize x isize
        main.add_module('initial_conv_{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial_relu_{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra_layers_{0}_{1}_conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra_layers_{0}_{1}_relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final_{0}-{1}_conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.mean(0)
        return output.view(1)

class DCGAN_G_nobn(nn.Module):
     # ... (Original code - needs refactoring similar to DCGAN_G if used) ...
    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G_nobn, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial_{0}-{1}_convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra_layers_{0}_{1}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra_layers_{0}_{1}_relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final_{0}-{1}_convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        # Note: Original used Softmax(), Tanh() was commented. Paper used ReLU.
        # Stick with ReLU unless you have reason to change.
        main.add_module('final_{0}_relu'.format(nc), # Changed from Softmax/Tanh to match DCGAN_G final layer
                        nn.ReLU(True))
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,  range(self.ngpu))
        else:
            output = self.main(input)
        return output
###############################################################################