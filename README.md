# Spatial Correlation Sampler - Pure PyTorch

A pure PyTorch implementation of Spatial Correlation Sampler, optimized for GPU performance. This module computes correlation between feature maps for motion estimation, commonly used in optical flow estimation (FlowNet) and object tracking (MM-Tracker).

## Features

✅ **Pure PyTorch** - No C++ compilation required  
✅ **GPU Optimized** - Efficient CUDA operations using unfold  
✅ **API Compatible** - Drop-in replacement for the C++ extension  
✅ **Easy to Install** - Just `pip install`  
✅ **Well Tested** - Comprehensive test suite  
✅ **MM-Tracker Ready** - Optimized for tracking applications  

## Installation

```bash
pip install spatial-correlation-sampler-pytorch
```

## How It Works

The Spatial Correlation Sampler computes correlation between two feature maps to measure similarity at different spatial displacements. This is essential for motion estimation:

1. **Input**: Two feature maps from consecutive frames
2. **Process**: For each pixel, compute correlation with nearby pixels in the second frame
3. **Output**: Correlation volume showing motion likelihood at different displacements

### Mathematical Foundation

For feature maps `F1` and `F2`, correlation at position `(i,j)` with displacement `(u,v)`:

```
Correlation(i,j,u,v) = Σ F1(i+x, j+y) * F2(i+x+u, j+y+v)
```

## API Reference

### Core Module
```python
from spatial_correlation_sampler import SpatialCorrelationSampler

sampler = SpatialCorrelationSampler(
    kernel_size=1,      # Search window size  
    patch_size=1,       # Patch size for comparison
    stride=1,           # Output stride
    padding=0,          # Input padding
    dilation=1,         # Kernel dilation
    dilation_patch=1    # Patch dilation
)

# Input: (B, C, H, W), Output: varies by configuration
correlation = sampler(feat1, feat2)
```

### Input/Output Shapes

| Configuration | Input | Output | Use Case |
|---------------|-------|--------|----------|
| `kernel_size=1, patch_size=K` | `(B,C,H,W)` | `(B,K²,H,W)` | FlowNet-style |
| `kernel_size=K, patch_size=1` | `(B,C,H,W)` | `(B,K²,H,W)` | MM-Tracker-style |

## Quick Start Examples

### 1. Basic Usage
```python
import torch
from spatial_correlation_sampler import SpatialCorrelationSampler

# Create feature maps
feat1 = torch.randn(1, 256, 64, 64, device='cuda')
feat2 = torch.randn(1, 256, 64, 64, device='cuda')

# Basic correlation with 3x3 search window
sampler = SpatialCorrelationSampler(kernel_size=3, patch_size=1, padding=1)
correlation = sampler(feat1, feat2)

print(f"Input: {feat1.shape} -> Output: {correlation.shape}")
# Output: torch.Size([1, 256, 64, 64]) -> torch.Size([1, 9, 64, 64])
```

### 2. FlowNet Configuration (Optical Flow)
```python
# FlowNet-style: search in 21x21 window around each pixel
sampler = SpatialCorrelationSampler(
    kernel_size=1,
    patch_size=21,      # ±10 pixel search range
    stride=1,
    padding=0,
    dilation_patch=2    # Sparse sampling for efficiency
)

feat1 = torch.randn(1, 256, 64, 64, device='cuda')
feat2 = torch.randn(1, 256, 64, 64, device='cuda')
flow_correlation = sampler(feat1, feat2)
# Output: (1, 441, 64, 64) - 441 = 21×21 displacement channels
```

### 3. MM-Tracker Configuration (Object Tracking)
```python
# MM-Tracker-style: compare pixels in 9x9 search area
max_displacement = 4
sampler = SpatialCorrelationSampler(
    kernel_size=2*max_displacement+1,  # 9x9 search window
    patch_size=1,
    stride=1,
    padding=max_displacement
)

# Typical MM-Tracker feature size (1/32 scale)
feat1 = torch.randn(1, 512, 19, 34, device='cuda') 
feat2 = torch.randn(1, 512, 19, 34, device='cuda')
tracking_correlation = sampler(feat1, feat2)
# Output: (1, 81, 19, 34) - 81 = 9×9 displacement channels
```

### 4. Understanding Output Channels

Each output channel represents correlation at a specific displacement:

```python
# For 3x3 search (9 channels):
# Channel 0: displacement (-1, -1) ← top-left
# Channel 1: displacement (-1,  0) ← top-center  
# Channel 2: displacement (-1, +1) ← top-right
# ...
# Channel 4: displacement ( 0,  0) ← center (no motion)
# ...
# Channel 8: displacement (+1, +1) ← bottom-right

# Find dominant motion direction
max_channel = correlation.argmax(dim=1)  # (B, H, W)
displacement_y = max_channel // 3 - 1    # Convert to displacement
displacement_x = max_channel % 3 - 1
```
