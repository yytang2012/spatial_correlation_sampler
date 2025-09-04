"""
Utility functions for correlation sampling
"""

from typing import Union, Tuple

import torch


def to_tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Convert single value to tuple"""
    if isinstance(value, tuple):
        return value
    return (value, value)


def calculate_output_size(
    input_size: int,
    kernel_size: int,
    padding: int,
    stride: int
) -> int:
    """Calculate output size after convolution/correlation"""
    return (input_size + 2 * padding - kernel_size) // stride + 1


def check_input_compatibility(input1: torch.Tensor, input2: torch.Tensor):
    """Check if two inputs are compatible for correlation"""
    assert input1.dim() == 4, f"Expected 4D tensor, got {input1.dim()}D"
    assert input2.dim() == 4, f"Expected 4D tensor, got {input2.dim()}D"
    assert input1.shape == input2.shape, \
        f"Input shapes must match: {input1.shape} vs {input2.shape}"
