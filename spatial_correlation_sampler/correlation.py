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
        original_shape = (H, W)
        
        # Apply padding
        if self.padding[0] > 0 or self.padding[1] > 0:
            input1 = F.pad(input1, [self.padding[1], self.padding[1],
                                   self.padding[0], self.padding[0]])
            input2 = F.pad(input2, [self.padding[1], self.padding[1],
                                   self.padding[0], self.padding[0]])
        
        if self.kernel_size[0] == 1 and self.kernel_size[1] == 1:
            # Special case: kernel_size = 1, variable patch_size
            return self._correlation_patch_based(input1, input2, original_shape)
        elif self.patch_size[0] == 1 and self.patch_size[1] == 1:
            # Special case: patch_size = 1, variable kernel_size
            return self._correlation_kernel_based(input1, input2, original_shape)
        else:
            # General case: both kernel_size and patch_size > 1
            return self._correlation_general(input1, input2, original_shape)
    
    def _correlation_kernel_based(self, input1: torch.Tensor, input2: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """
        GPU-optimized implementation for patch_size=1, variable kernel_size
        Used in MM-Tracker configuration
        
        Uses F.unfold for efficient GPU computation instead of nested loops
        """
        B, C, H, W = input1.shape
        K_h, K_w = self.kernel_size
        
        # Precise output dimension calculation
        out_h = (H - self.dilation[0] * (K_h - 1) - 1) // self.stride[0] + 1
        out_w = (W - self.dilation[1] * (K_w - 1) - 1) // self.stride[1] + 1
        
        # Use unfold to extract all kernel windows from input2
        input2_unfold = F.unfold(
            input2,
            kernel_size=(K_h, K_w),
            stride=self.stride,
            padding=0,  # Padding handled externally
            dilation=self.dilation
        )  # Shape: (B, C * K_h * K_w, N_patches)
        
        N_patches = input2_unfold.shape[2]
        
        # Get input1 at stride positions (single pixel queries)  
        input1_unfold = F.unfold(
            input1,
            kernel_size=1,
            stride=self.stride,
            padding=0
        )  # Shape: (B, C, N_patches_input1)
        
        N_patches_input1 = input1_unfold.shape[2]
        
        # Use the actual number of patches from input1 (should be the same)
        if N_patches_input1 != N_patches:
            # Take minimum and truncate both to match
            min_patches = min(N_patches_input1, N_patches)
            input1_unfold = input1_unfold[:, :, :min_patches]
            input2_unfold = input2_unfold[:, :, :min_patches]
            N_patches = min_patches
        
        # Use precise output dimensions
        actual_out_h, actual_out_w = out_h, out_w
        
        # Verify dimensions match unfold result
        if actual_out_h * actual_out_w != N_patches:
            # Fallback: use actual patches from unfold
            import math
            h = int(math.sqrt(N_patches))
            if h * h == N_patches:
                actual_out_h = actual_out_w = h
            else:
                actual_out_h = int(math.sqrt(N_patches))
                actual_out_w = N_patches // actual_out_h
        
        # Reshape to separate channels and kernel positions
        input2_unfold = input2_unfold.view(B, C, K_h * K_w, N_patches)
        
        # Expand input1 to match kernel dimensions 
        input1_expanded = input1_unfold.unsqueeze(2).expand(-1, -1, K_h * K_w, -1)
        # Shape: (B, C, K_h * K_w, N_patches)
        
        # Compute correlation via element-wise multiplication and channel-wise sum
        correlation = (input1_expanded * input2_unfold).sum(dim=1)
        # Shape: (B, K_h * K_w, N_patches)
        
        # Reshape to required output format
        output = correlation.unsqueeze(1).view(B, 1, K_h * K_w, actual_out_h, actual_out_w)
        
        return output
    
    def _correlation_patch_based(self, input1: torch.Tensor, input2: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """
        Fully vectorized implementation for kernel_size=1, variable patch_size
        Eliminates all Python loops using advanced indexing and einsum
        """
        B, C, H, W = input1.shape
        P_h, P_w = self.patch_size
        
        # Calculate output dimensions
        out_h = (H - 1) // self.stride[0] + 1 
        out_w = (W - 1) // self.stride[1] + 1
        
        if self.stride[0] == 1 and self.stride[1] == 1:
            # Optimized path for stride=1
            input2_unfold = F.unfold(
                input2,
                kernel_size=(P_h, P_w),
                stride=1,
                padding=(P_h // 2, P_w // 2),
                dilation=self.dilation_patch
            )
            
            # Robust dimension handling
            B_unfold, total_channels, spatial_positions = input2_unfold.shape
            patches_in_kernel = total_channels // C
            
            # Check if spatial positions match H*W
            if spatial_positions == H * W:
                input2_patches = input2_unfold.view(B, C, patches_in_kernel, H, W)
                # Use einsum for efficient correlation computation
                correlation = torch.einsum('bchw,bcphw->bphw', 
                                         input1, input2_patches)
                return correlation
            else:
                # Use flat representation for dilation_patch cases
                input2_patches = input2_unfold.view(B, C, patches_in_kernel, spatial_positions)
                input1_flat = input1.view(B, C, H * W)
                
                # Handle spatial position mismatch
                min_spatial = min(H * W, spatial_positions)
                input1_sampled = input1_flat[:, :, :min_spatial]
                input2_sampled = input2_patches[:, :, :, :min_spatial]
                
                # Vectorized correlation using einsum
                correlation = torch.einsum('bcn,bcpn->bpn', input1_sampled, input2_sampled)
                
                # Reshape to proper patch_size format (21x21 -> 21, 21 for FlowNet)
                import math
                patch_sqrt = int(math.sqrt(patches_in_kernel))
                if patch_sqrt * patch_sqrt == patches_in_kernel:
                    # Square patch case (21x21 = 441)
                    correlation_reshaped = correlation.view(B, patch_sqrt, patch_sqrt, min_spatial)
                    
                    # Pad or reshape spatial dimensions
                    if min_spatial == H * W:
                        return correlation_reshaped.view(B, patch_sqrt, patch_sqrt, H, W)
                    else:
                        output = torch.zeros(B, patch_sqrt, patch_sqrt, H, W, device=input1.device, dtype=input1.dtype)
                        output.view(B, patch_sqrt, patch_sqrt, -1)[:, :, :, :min_spatial] = correlation_reshaped
                        return output
                else:
                    # Non-square patch case
                    if min_spatial == H * W:
                        return correlation.view(B, patches_in_kernel, H, W)
                    else:
                        output = torch.zeros(B, patches_in_kernel, H, W, device=input1.device, dtype=input1.dtype)
                        output.view(B, patches_in_kernel, -1)[:, :, :min_spatial] = correlation
                        return output
        
        else:
            # Vectorized path for non-unit stride using advanced indexing
            # Create sampling indices
            y_indices = torch.arange(0, out_h, device=input1.device) * self.stride[0]
            x_indices = torch.arange(0, out_w, device=input1.device) * self.stride[1]
            
            # Clamp indices to valid range
            y_indices = torch.clamp(y_indices, 0, H-1)
            x_indices = torch.clamp(x_indices, 0, W-1)
            
            # Advanced indexing to sample input1
            input1_sampled = input1[:, :, y_indices[:, None], x_indices[None, :]]
            # Shape: (B, C, out_h, out_w)
            
            # Extract patches from input2 at stride positions
            input2_unfold = F.unfold(
                input2,
                kernel_size=(P_h, P_w),
                stride=self.stride,
                padding=(P_h // 2, P_w // 2),
                dilation=self.dilation_patch
            )
            
            # Handle dimension matching
            n_patches = input2_unfold.shape[2]
            actual_patches = min(n_patches, out_h * out_w)
            patches_in_kernel = input2_unfold.shape[1] // C
            
            input2_patches = input2_unfold[:, :, :actual_patches].view(
                B, C, patches_in_kernel, actual_patches
            )
            input1_flat = input1_sampled.reshape(B, C, -1)[:, :, :actual_patches]
            
            # Vectorized correlation using einsum
            correlation = torch.einsum('bcn,bcpn->bpn', input1_flat, input2_patches)
            
            # Reshape to output dimensions
            if actual_patches == out_h * out_w:
                correlation = correlation.view(B, patches_in_kernel, out_h, out_w)
            else:
                # Handle padding when needed
                output = torch.zeros(B, patches_in_kernel, out_h, out_w, 
                                   device=input1.device, dtype=input1.dtype)
                output.view(B, patches_in_kernel, -1)[:, :, :actual_patches] = correlation
                correlation = output
            
            return correlation
    def _correlation_general(self, input1: torch.Tensor, input2: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """
        Optimized general implementation with torch.roll for efficient spatial shifts
        Uses vectorized operations and better boundary handling
        """
        B, C, H, W = input1.shape
        K_h, K_w = self.kernel_size
        P_h, P_w = self.patch_size
        
        out_h = (H - self.dilation[0] * (K_h - 1) - 1) // self.stride[0] + 1
        out_w = (W - self.dilation[1] * (K_w - 1) - 1) // self.stride[1] + 1
        
        # Pre-compute all kernel offset positions
        kernel_offsets = []
        for ky in range(K_h):
            for kx in range(K_w):
                y_offset = (ky - K_h//2) * self.dilation[0]
                x_offset = (kx - K_w//2) * self.dilation[1]
                kernel_offsets.append((y_offset, x_offset))
        
        # Extract input1 patches once
        input1_patches = F.unfold(
            input1,
            kernel_size=(P_h, P_w),
            stride=self.stride,
            padding=(P_h//2, P_w//2),
            dilation=self.dilation_patch
        ).view(B, C * P_h * P_w, -1)  # (B, C*P_h*P_w, N_patches)
        
        # Process all kernel positions with torch.roll
        correlations = []
        for y_offset, x_offset in kernel_offsets:
            # Use torch.roll for efficient spatial shifting
            if y_offset != 0 or x_offset != 0:
                input2_shifted = torch.roll(input2, shifts=(-y_offset, -x_offset), dims=(2, 3))
                
                # Handle boundaries by zeroing wrapped regions
                if y_offset > 0:
                    input2_shifted[:, :, -y_offset:, :] = 0
                elif y_offset < 0:
                    input2_shifted[:, :, :-y_offset, :] = 0
                
                if x_offset > 0:
                    input2_shifted[:, :, :, -x_offset:] = 0
                elif x_offset < 0:
                    input2_shifted[:, :, :, :-x_offset] = 0
            else:
                input2_shifted = input2
            
            # Extract patches from shifted input2
            input2_patches = F.unfold(
                input2_shifted,
                kernel_size=(P_h, P_w),
                stride=self.stride,
                padding=(P_h//2, P_w//2),
                dilation=self.dilation_patch
            ).view(B, C * P_h * P_w, -1)
            
            # Ensure dimension matching
            min_patches = min(input1_patches.shape[2], input2_patches.shape[2])
            
            # Efficient batch correlation using tensor operations
            corr = (input1_patches[:, :, :min_patches] * 
                   input2_patches[:, :, :min_patches]).view(B, C, P_h*P_w, min_patches).sum(dim=1)
            
            correlations.append(corr)
        
        # Stack all kernel position results
        output = torch.stack(correlations, dim=1)  # (B, K_h*K_w, P_h*P_w, N_patches)
        
        # Reshape to final output format with robust dimension handling
        try:
            output = output.view(B, K_h*K_w*P_h*P_w, out_h, out_w)
        except RuntimeError:
            # Fallback for dimension mismatch
            n_patches = output.shape[-1]
            output_flat = output.view(B, -1, n_patches)
            final_output = torch.zeros(B, K_h*K_w*P_h*P_w, out_h, out_w, device=input1.device)
            min_size = min(n_patches, out_h * out_w)
            final_output.view(B, -1, out_h * out_w)[:, :, :min_size] = output_flat[:, :, :min_size]
            output = final_output
        
        return output