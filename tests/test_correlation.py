"""
Unit tests for spatial correlation sampler
"""

import torch
from spatial_correlation_sampler import (
    SpatialCorrelationSampler,
    OptimizedSpatialCorrelationSampler,
    spatial_correlation_sample
)


class TestSpatialCorrelationSampler:
    """Test suite for correlation sampler"""

    @pytest.fixture
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_basic_correlation(self, device):
        """Test basic correlation computation"""
        sampler = SpatialCorrelationSampler(
            kernel_size=3,
            patch_size=1,
            padding=1
        )

        input1 = torch.randn(2, 64, 10, 10, device=device)
        input2 = torch.randn(2, 64, 10, 10, device=device)

        output = sampler(input1, input2)

        assert output.shape == (2, 1, 1, 10, 10)
        assert not torch.isnan(output).any()

    def test_mm_tracker_config(self, device):
        """Test MM-Tracker configuration"""
        sampler = SpatialCorrelationSampler(
            kernel_size=9,
            patch_size=1,
            stride=1,
            padding=4
        )

        input1 = torch.randn(1, 512, 19, 34, device=device)
        input2 = torch.randn(1, 512, 19, 34, device=device)

        output = sampler(input1, input2)

        assert output.shape == (1, 1, 1, 19, 34)

    def test_flownet_config(self, device):
        """Test FlowNet configuration"""
        sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=21,
            stride=1,
            padding=0,
            dilation=1,
            dilation_patch=2
        )

        input1 = torch.randn(4, 256, 48, 64, device=device)
        input2 = torch.randn(4, 256, 48, 64, device=device)

        output = sampler(input1, input2)

        # Check output shape matches expected
        assert output.dim() == 5

    def test_optimized_version(self, device):
        """Test optimized implementation"""
        optimized = OptimizedSpatialCorrelationSampler(
            kernel_size=9,
            aggregation='max'
        )

        input1 = torch.randn(2, 256, 20, 20, device=device)
        input2 = torch.randn(2, 256, 20, 20, device=device)

        output = optimized(input1, input2)

        assert output.shape == (2, 256, 20, 20)

    def test_functional_interface(self, device):
        """Test functional interface"""
        input1 = torch.randn(1, 128, 15, 15, device=device)
        input2 = torch.randn(1, 128, 15, 15, device=device)

        output = spatial_correlation_sample(
            input1, input2,
            kernel_size=5,
            patch_size=1,
            padding=2
        )

        assert output.shape[3:] == (15, 15)