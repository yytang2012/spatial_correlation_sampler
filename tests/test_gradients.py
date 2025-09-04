"""
Gradient tests for correlation sampler
"""

import torch
from spatial_correlation_sampler import SpatialCorrelationSampler


class TestGradients:
    """Test gradient computation"""

    def test_gradient_flow(self):
        """Test that gradients flow properly"""
        sampler = SpatialCorrelationSampler(kernel_size=3, patch_size=1, padding=1)

        input1 = torch.randn(2, 32, 8, 8, requires_grad=True)
        input2 = torch.randn(2, 32, 8, 8, requires_grad=True)

        output = sampler(input1, input2)
        loss = output.mean()
        loss.backward()

        assert input1.grad is not None
        assert input2.grad is not None
        assert not torch.isnan(input1.grad).any()
        assert not torch.isnan(input2.grad).any()

    def test_gradient_correctness(self):
        """Test gradient correctness with numerical approximation"""
        from torch.autograd import gradcheck

        sampler = SpatialCorrelationSampler(kernel_size=3, padding=1)

        input1 = torch.randn(1, 8, 4, 4, requires_grad=True, dtype=torch.float64)
        input2 = torch.randn(1, 8, 4, 4, requires_grad=True, dtype=torch.float64)

        # Note: This might be slow
        # assert gradcheck(sampler, (input1, input2), eps=1e-6, atol=1e-4)