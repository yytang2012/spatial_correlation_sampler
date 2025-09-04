# Technical Guide: Spatial Correlation Sampler

## Working Principle

The Spatial Correlation Sampler computes correlation between two feature maps, which is essential for motion estimation in computer vision tasks like optical flow and object tracking.

### Core Concept

Given two feature maps from consecutive frames, the correlation sampler measures the similarity between patches in the first feature map and corresponding regions in the second feature map. This helps identify where objects have moved between frames.

### Mathematical Foundation

For two input feature maps `F1(x,y)` and `F2(x,y)`, the correlation at position `(i,j)` with displacement `(u,v)` is:

```
Correlation(i,j,u,v) = Σ F1(i+x, j+y) * F2(i+x+u, j+y+v)
```

Where the summation is over the patch area defined by `patch_size`.

## Algorithm Implementation

### Two Main Modes

1. **Patch-based Mode** (`kernel_size=1`): Fixed query point, variable search area
2. **Kernel-based Mode** (`patch_size=1`): Variable query area, fixed comparison point

### Processing Steps

1. **Input Padding**: Apply padding to handle border cases
2. **Patch Extraction**: Extract patches based on `patch_size` and `dilation_patch`
3. **Correlation Computation**: Compute dot product between corresponding patches
4. **Output Formatting**: Reshape results to match expected API format

## API Reference

### Input/Output Specification

```python
SpatialCorrelationSampler(
    kernel_size=1,      # Search window size
    patch_size=1,       # Patch size for comparison
    stride=1,           # Output stride
    padding=0,          # Input padding
    dilation=1,         # Kernel dilation
    dilation_patch=1    # Patch dilation
)
```

#### Input Tensors
- **input1**: `(B, C, H, W)` - First feature map
- **input2**: `(B, C, H, W)` - Second feature map

#### Output Tensor
- **kernel_size=1**: `(B, patch_size², H_out, W_out)` - Patch-based correlation
- **patch_size=1**: `(B, kernel_size², H_out, W_out)` - Kernel-based correlation

Where:
- `H_out = (H + 2*padding - kernel_size) // stride + 1`
- `W_out = (W + 2*padding - kernel_size) // stride + 1`

## Common Use Cases

### 1. FlowNet Configuration (Optical Flow)
```python
# Search in 21x21 window around each pixel
sampler = SpatialCorrelationSampler(
    kernel_size=1,
    patch_size=21,
    stride=1,
    padding=0,
    dilation_patch=2  # Sparse sampling
)
```

### 2. MM-Tracker Configuration (Object Tracking)
```python
# Compare single pixels in 9x9 search area
sampler = SpatialCorrelationSampler(
    kernel_size=9,
    patch_size=1,
    stride=1,
    padding=4
)
```

## Detailed Example

Let's walk through a concrete example with small feature maps:

```python
import torch
from spatial_correlation_sampler import SpatialCorrelationSampler

# Create sample feature maps
B, C, H, W = 1, 2, 5, 5
feat1 = torch.randn(B, C, H, W)
feat2 = torch.randn(B, C, H, W)

# Configuration: 3x3 search window, single pixel comparison
sampler = SpatialCorrelationSampler(
    kernel_size=3,
    patch_size=1,
    stride=1,
    padding=1
)

output = sampler(feat1, feat2)
print(f"Input: {feat1.shape} -> Output: {output.shape}")
# Output: torch.Size([1, 2, 5, 5]) -> torch.Size([1, 9, 5, 5])
```

### Understanding the Output

For each pixel in the output:
- **9 channels**: Represent correlations at 9 different displacements in a 3x3 grid
- **Channel 0**: Correlation with displacement (-1, -1)
- **Channel 4**: Correlation with displacement (0, 0) - center
- **Channel 8**: Correlation with displacement (1, 1)

### Visualization of Correlation Process

```
Feature Map 1 (5x5):          Feature Map 2 (5x5):
┌─────────────────┐          ┌─────────────────┐
│ a b c d e       │          │ a' b' c' d' e'  │
│ f g h i j       │          │ f' g' h' i' j'  │
│ k l m n o   ──► │          │ k' l' m' n' o'  │  ──► Correlation
│ p q r s t       │          │ p' q' r' s' t'  │
│ u v w x y       │          │ u' v' w' x' y'  │
└─────────────────┘          └─────────────────┘

For pixel 'm' at (2,2), compute correlation with:
- Displacement (-1,-1): m * h'  (channel 0)
- Displacement ( 0, 0): m * m'  (channel 4)  
- Displacement ( 1, 1): m * r'  (channel 8)
... and so on for all 9 displacements
```

## Performance Characteristics

### Memory Complexity
- **O(B × C × H × W × patch_size²)** for patch-based mode
- **O(B × C × H × W × kernel_size²)** for kernel-based mode

### Computational Complexity
- Linear in input size and patch/kernel area
- GPU-optimized using PyTorch operations
- No custom CUDA kernels required

## Common Pitfalls and Solutions

### 1. Parameter Naming
**Issue**: Original C++ extension uses `patch_dilation`, ours uses `dilation_patch`
**Solution**: Use `dilation_patch` parameter in our implementation

### 2. Output Format Confusion
**Issue**: Different output shapes for different configurations
**Solution**: Check documentation for expected output format based on `kernel_size` and `patch_size`

### 3. GPU Memory Usage
**Issue**: Large correlation volumes can consume significant GPU memory
**Solution**: Use smaller batch sizes or reduce `patch_size`/`kernel_size`

## Integration Guide

### Replacing Original C++ Extension

```python
# Original (C++ extension)
from spatial_correlation_sampler import SpatialCorrelationSampler

# Our pure PyTorch version (same import)
from spatial_correlation_sampler import SpatialCorrelationSampler

# Usage remains identical except parameter naming:
# patch_dilation -> dilation_patch
```

### Gradient Flow
The implementation fully supports gradient computation and backpropagation, making it suitable for end-to-end training of neural networks.

### Multi-GPU Support
The pure PyTorch implementation automatically works with DataParallel and DistributedDataParallel without additional configuration.