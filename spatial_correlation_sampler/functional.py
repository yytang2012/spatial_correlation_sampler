"""
Functional interface for correlation sampling
"""

import torch
from .correlation import SpatialCorrelationSampler
from .optimized import OptimizedSpatialCorrelationSampler


def spatial_correlation_sample(
        input1: torch.Tensor,
        input2: torch.Tensor,
        kernel_size: int = 1,
        patch_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        dilation_patch: int = 1
) -> torch.Tensor:
    """
    Functional interface for spatial correlation sampling

    Args:
        input1: First input tensor [B, C, H, W]
        input2: Second input tensor [B, C, H, W]
        kernel_size: Size of the search window
        patch_size: Size of the patch for comparison
        stride: Stride for output
        padding: Padding for input
        dilation: Dilation for kernel
        dilation_patch: Dilation for patch

    Returns:
        Correlation tensor [B, PatchH, PatchW, oH, oW]

    Examples:
        >>> import torch
        >>> from spatial_correlation_sampler import spatial_correlation_sample
        >>>
        >>> feat1 = torch.randn(1, 256, 32, 32)
        >>> feat2 = torch.randn(1, 256, 32, 32)
        >>> corr = spatial_correlation_sample(
        ...     feat1, feat2,
        ...     kernel_size=9,
        ...     patch_size=1,
        ...     padding=4
        ... )
    """
    sampler = SpatialCorrelationSampler(
        kernel_size=kernel_size,
        patch_size=patch_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dilation_patch=dilation_patch
    )
    return sampler(input1, input2)


def correlation_sample_optimized(
        input1: torch.Tensor,
        input2: torch.Tensor,
        kernel_size: int = 9,
        aggregation: str = 'max'
) -> torch.Tensor:
    """
    Optimized correlation sampling for patch_size=1 case

    Args:
        input1: First input tensor [B, C, H, W]
        input2: Second input tensor [B, C, H, W]
        kernel_size: Size of search window
        aggregation: Aggregation method ('max', 'mean', 'sum')

    Returns:
        Correlation features [B, C, H, W]
    """
    padding = (kernel_size - 1) // 2
    sampler = OptimizedSpatialCorrelationSampler(
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        aggregation=aggregation
    )
    return sampler(input1, input2)