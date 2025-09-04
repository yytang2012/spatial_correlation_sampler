"""
Spatial Correlation Sampler - Pure PyTorch Implementation

A GPU-optimized implementation of spatial correlation sampling for optical flow
and motion estimation tasks, commonly used in FlowNet and MM-Tracker.
"""

from .correlation import SpatialCorrelationSampler
from .optimized import OptimizedSpatialCorrelationSampler
from .functional import (
    spatial_correlation_sample,
    correlation_sample_optimized
)

__version__ = "1.0.0"
__all__ = [
    "SpatialCorrelationSampler",
    "OptimizedSpatialCorrelationSampler",
    "spatial_correlation_sample",
    "correlation_sample_optimized",
]