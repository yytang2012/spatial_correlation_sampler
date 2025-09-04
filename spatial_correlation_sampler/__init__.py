"""
Spatial Correlation Sampler - Pure PyTorch Implementation

A GPU-optimized implementation of spatial correlation sampling for optical flow
and motion estimation tasks, commonly used in FlowNet and MM-Tracker.

Features:
- Drop-in replacement for C++ extensions  
- GPU-optimized using F.unfold operations
- Compatible with MMTracker and FlowNet configurations
"""

from .correlation import SpatialCorrelationSampler

__version__ = "1.0.0"
__all__ = ["SpatialCorrelationSampler"]