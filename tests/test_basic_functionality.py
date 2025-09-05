"""
Basic functionality tests
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial_correlation_sampler import SpatialCorrelationSampler


def test_mmtracker_config():
    """Test MMTracker configuration"""
    print("Testing MMTracker configuration...")
    
    # Correct MMTracker parameters
    sampler = SpatialCorrelationSampler(
        kernel_size=9,
        patch_size=1, 
        stride=1,
        padding=4,
        dilation=1
    )
    
    # Test data
    B, C, H, W = 2, 256, 30, 40
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)
    
    output = sampler(x, y)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    
    # Verify output
    assert len(output.shape) == 5, f"Expected 5D output, got {len(output.shape)}D"
    assert output.shape == (B, 1, 81, H, W), f"Expected shape {(B, 1, 81, H, W)}, got {output.shape}"
    
    # Convert to 4D format (expected by MMTracker)
    output_4d = output[:, 0, :, :, :]
    print(f"4D format: {output_4d.shape}")
    assert output_4d.shape == (B, 81, H, W)
    
    print("✓ MMTracker configuration test passed")


def test_flownet_config():
    """Test FlowNet configuration"""
    print("\nTesting FlowNet configuration...")
    
    # FlowNet configuration
    sampler = SpatialCorrelationSampler(
        kernel_size=1,
        patch_size=21,
        stride=1,
        padding=0,
        dilation=1,
        dilation_patch=2
    )
    
    # Test data
    B, C, H, W = 1, 64, 32, 32
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)
    
    output = sampler(x, y)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    
    # FlowNet output format check
    if len(output.shape) == 5:
        # (B, PatchH, PatchW, H, W) format
        expected_patch_h, expected_patch_w = 21, 21
        assert output.shape[1:3] == (expected_patch_h, expected_patch_w), f"Expected patch size {(expected_patch_h, expected_patch_w)}, got {output.shape[1:3]}"
    elif len(output.shape) == 4:
        # (B, PatchH*PatchW, H, W) format
        expected_channels = 21 * 21  # 441
        assert output.shape[1] == expected_channels, f"Expected {expected_channels} channels, got {output.shape[1]}"
    
    print("✓ FlowNet configuration test passed")


def test_basic_correlation():
    """Test basic correlation computation"""
    print("\nTesting basic correlation computation...")
    
    # Simple configuration test
    sampler = SpatialCorrelationSampler(
        kernel_size=3,
        patch_size=1,
        stride=1,
        padding=1,
        dilation=1
    )
    
    B, C, H, W = 1, 64, 16, 16
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)
    
    output = sampler(x, y)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    
    # Numerical check
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert torch.isfinite(output).all(), "Output contains infinite values"
    
    print(f"Value range: [{output.min():.4f}, {output.max():.4f}]")
    
    print("✓ Basic correlation computation test passed")


if __name__ == "__main__":
    print("Spatial Correlation Sampler Basic Functionality Test")
    print("=" * 50)
    
    try:
        test_mmtracker_config()
        test_flownet_config()
        test_basic_correlation()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()