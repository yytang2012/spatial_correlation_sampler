"""
Core implementation of Spatial Correlation Sampler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .utils import to_tuple, calculate_output_size


class SpatialCorrelationSampler(nn.Module):
    """
    Pure PyTorch implementation of Spatial Correlation Sampler

    This module computes correlation between two feature maps, commonly used
    in optical flow estimation (FlowNet) and object tracking (MM-Tracker).

    Args:
        kernel_size (int or tuple): Size of the correlation kernel (search window)
        patch_size (int or tuple): Size of the patch for comparison
        stride (int or tuple): Stride for the correlation output
        padding (int or tuple): Padding applied to input
        dilation (int or tuple): Dilation for the correlation kernel
        dilation_patch (int or tuple): Dilation for the patch

    Shape:
        - Input1, Input2: (B, C, H, W)
        - Output: (B, PatchH, PatchW, oH, oW) when kernel_size=1
        - Output: (B, 1, KernelH * KernelW, oH, oW) when patch_size=1
    """

    def __init__(
            self,
            kernel_size: int = 1,
            patch_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            dilation_patch: int = 1
    ):
        super(SpatialCorrelationSampler, self).__init__()

        self.kernel_size = to_tuple(kernel_size)
        self.patch_size = to_tuple(patch_size)
        self.stride = to_tuple(stride)
        self.padding = to_tuple(padding)
        self.dilation = to_tuple(dilation)
        self.dilation_patch = to_tuple(dilation_patch)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        B, C, H, W = input1.shape
        
        # Apply padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            input1 = F.pad(input1, [self.padding[1], self.padding[1],
                                   self.padding[0], self.padding[0]])
            input2 = F.pad(input2, [self.padding[1], self.padding[1],
                                   self.padding[0], self.padding[0]])
        
        if self.kernel_size[0] == 1 and self.kernel_size[1] == 1:
            # Special case: kernel_size = 1, variable patch_size
            return self._correlation_patch_based(input1, input2)
        elif self.patch_size[0] == 1 and self.patch_size[1] == 1:
            # Special case: patch_size = 1, variable kernel_size
            return self._correlation_kernel_based(input1, input2)
        else:
            # General case: both kernel_size and patch_size > 1
            return self._correlation_general(input1, input2)
    
    def _correlation_kernel_based(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Optimized implementation for patch_size=1, variable kernel_size
        Used in MM-Tracker configuration
        """
        B, C, H, W = input1.shape
        
        # Calculate output dimensions
        out_h = (H - 1) // self.stride[0] + 1
        out_w = (W - 1) // self.stride[1] + 1
        
        # Initialize output - format (B, 1, K_h*K_w, out_h, out_w)
        K_h, K_w = self.kernel_size
        output = torch.zeros(B, 1, K_h * K_w, out_h, out_w, 
                            device=input1.device, dtype=input1.dtype)
        
        # Extract query positions from input1
        for y in range(out_h):
            for x in range(out_w):
                y_coord = y * self.stride[0]
                x_coord = x * self.stride[1]
                
                # Get single pixel from input1
                query = input1[:, :, y_coord:y_coord+1, x_coord:x_coord+1]  # (B, C, 1, 1)
                
                # Compute correlation with kernel window in input2
                idx = 0
                for ky in range(K_h):
                    for kx in range(K_w):
                        # Calculate offset with dilation
                        dy = ky * self.dilation[0]
                        dx = kx * self.dilation[1]
                        
                        y_start = y_coord + dy - (K_h // 2) * self.dilation[0]
                        x_start = x_coord + dx - (K_w // 2) * self.dilation[1]
                        
                        # Check boundaries
                        if 0 <= y_start < H and 0 <= x_start < W:
                            target = input2[:, :, y_start:y_start+1, x_start:x_start+1]
                            # Compute correlation
                            corr = (query * target).sum(dim=1).squeeze()  # (B,)
                            output[:, 0, idx, y, x] = corr
                        
                        idx += 1
        
        return output
    
    def _correlation_patch_based(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        Optimized implementation for kernel_size=1, variable patch_size
        Used in FlowNet configuration
        """
        B, C, H, W = input1.shape
        
        # Calculate output dimensions
        out_h = (H - 1) // self.stride[0] + 1
        out_w = (W - 1) // self.stride[1] + 1
        
        # Initialize output - format (B, P_h, P_w, out_h, out_w)
        P_h, P_w = self.patch_size
        output = torch.zeros(B, P_h, P_w, out_h, out_w,
                            device=input1.device, dtype=input1.dtype)
        
        # Process each output position
        for y in range(out_h):
            for x in range(out_w):
                y_coord = y * self.stride[0]
                x_coord = x * self.stride[1]
                
                # Get patch from input1 centered at (y_coord, x_coord)
                patch1 = self._extract_patch(input1, y_coord, x_coord, P_h, P_w)
                
                # Get corresponding patch from input2
                patch2 = self._extract_patch(input2, y_coord, x_coord, P_h, P_w)
                
                if patch1 is not None and patch2 is not None:
                    # Compute element-wise correlation
                    corr = patch1 * patch2  # (B, C, P_h, P_w)
                    corr = corr.sum(dim=1)  # (B, P_h, P_w)
                    output[:, :, :, y, x] = corr
        
        # Reshape to (B, P_h*P_w, H, W) for compatibility
        return output.view(B, P_h * P_w, out_h, out_w)
    
    def _extract_patch(self, input: torch.Tensor, center_y: int, center_x: int,
                       patch_h: int, patch_w: int) -> Optional[torch.Tensor]:
        """Extract a patch centered at given coordinates with dilation"""
        B, C, H, W = input.shape
        
        # Calculate patch boundaries with dilation
        half_patch_h = patch_h // 2
        half_patch_w = patch_w // 2
        
        # Initialize patch tensor
        patch = torch.zeros(B, C, patch_h, patch_w, 
                           device=input.device, dtype=input.dtype)
        
        for ph in range(patch_h):
            for pw in range(patch_w):
                # Calculate actual position with dilation
                y = center_y + (ph - half_patch_h) * self.dilation_patch[0]
                x = center_x + (pw - half_patch_w) * self.dilation_patch[1]
                
                # Check boundaries
                if 0 <= y < H and 0 <= x < W:
                    patch[:, :, ph, pw] = input[:, :, y, x]
        
        return patch
    
    def _correlation_general(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        General implementation for arbitrary kernel_size and patch_size
        """
        B, C, H, W = input1.shape
        
        # Calculate output dimensions
        out_h = (H - 1) // self.stride[0] + 1
        out_w = (W - 1) // self.stride[1] + 1
        
        K_h, K_w = self.kernel_size
        P_h, P_w = self.patch_size
        
        # Initialize output
        output = torch.zeros(B, K_h * K_w * P_h * P_w, out_h, out_w,
                            device=input1.device, dtype=input1.dtype)
        
        idx = 0
        for ky in range(K_h):
            for kx in range(K_w):
                for py in range(P_h):
                    for px in range(P_w):
                        for y in range(out_h):
                            for x in range(out_w):
                                y_coord = y * self.stride[0]
                                x_coord = x * self.stride[1]
                                
                                # Position in input1
                                y1 = y_coord + (py - P_h // 2) * self.dilation_patch[0]
                                x1 = x_coord + (px - P_w // 2) * self.dilation_patch[1]
                                
                                # Position in input2 (with kernel offset)
                                y2 = y_coord + (ky - K_h // 2) * self.dilation[0] + (py - P_h // 2) * self.dilation_patch[0]
                                x2 = x_coord + (kx - K_w // 2) * self.dilation[1] + (px - P_w // 2) * self.dilation_patch[1]
                                
                                if 0 <= y1 < H and 0 <= x1 < W and 0 <= y2 < H and 0 <= x2 < W:
                                    corr = (input1[:, :, y1, x1] * input2[:, :, y2, x2]).sum(dim=1)
                                    output[:, idx, y, x] = corr
                        
                        idx += 1
        
        return output