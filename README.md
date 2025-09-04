# Spatial Correlation Sampler - Pure PyTorch

A pure PyTorch implementation of Spatial Correlation Sampler, optimized for GPU performance. This module is commonly used in optical flow estimation (FlowNet) and object tracking (MM-Tracker).

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

# Quick Start
## Basic Usage
```python
import torch
from spatial_correlation_sampler import spatial_correlation_sample

# Create feature maps
feat1 = torch.randn(1, 256, 32, 32, device='cuda')
feat2 = torch.randn(1, 256, 32, 32, device='cuda')

# Compute correlation
correlation = spatial_correlation_sample(
    feat1, feat2,
    kernel_size=9,  # Search window
    patch_size=1,   # Patch size
    padding=4       # Padding
)
```

## MM-Tracker Configuration

```python
from spatial_correlation_sampler import OptimizedSpatialCorrelationSampler

# Optimized for MM-Tracker
sampler = OptimizedSpatialCorrelationSampler(
    kernel_size=9,
    padding=4,
    aggregation='max'
)

# 1/32 scale features
feat1 = torch.randn(8, 512, 19, 34, device='cuda')
feat2 = torch.randn(8, 512, 19, 34, device='cuda')

motion_features = sampler(feat1, feat2)
```
