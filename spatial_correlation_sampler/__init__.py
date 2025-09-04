"""
Spatial Correlation Sampler - Pure PyTorch Implementation

A GPU-optimized implementation of spatial correlation sampling for optical flow
and motion estimation tasks, commonly used in FlowNet and MM-Tracker.
"""

from .correlation import SpatialCorrelationSampler

__version__ = "1.0.0"
__all__ = ["SpatialCorrelationSampler"]