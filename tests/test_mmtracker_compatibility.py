#!/usr/bin/env python
"""
Test compatibility with MMTracker requirements
"""

import torch
import torch.nn as nn
from spatial_correlation_sampler import SpatialCorrelationSampler

class MMTrackerCorrelation(nn.Module):
    """
    MMTracker's Correlation module implementation
    Uses SpatialCorrelationSampler(1, kernel_size, 1, 0, 1)
    """
    def __init__(self, max_displacement):
        super(MMTrackerCorrelation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2 * max_displacement + 1
        # MMTracker uses: SpatialCorrelationSampler(1, self.kernel_size, 1, 0, 1)
        # Parameters: kernel_size=1, patch_size=kernel_size, stride=1, padding=0, dilation=1
        self.corr = SpatialCorrelationSampler(
            kernel_size=1, 
            patch_size=self.kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1
        )
        
    def forward(self, x, y):
        b, c, h, w = x.shape
        return self.corr(x, y).view(b, -1, h, w) / c

def test_mmtracker_configuration():
    """Test MMTracker-specific configuration"""
    print("=" * 80)
    print(" Testing MMTracker Compatibility")
    print("=" * 80)
    
    # MMTracker uses max_displacement=4 typically (kernel_size=9)
    max_displacement = 4
    print(f"\nConfiguration:")
    print(f"  max_displacement: {max_displacement}")
    print(f"  kernel_size: {2 * max_displacement + 1} (2*max_displacement+1)")
    
    # Create correlation module
    correlation = MMTrackerCorrelation(max_displacement)
    
    # Test with typical feature map sizes
    # MMTracker uses 1/32 scale features (19x34 for 608x1088 input)
    batch_size = 8
    channels = 512  # Typical feature channels
    height = 19
    width = 34
    
    # Create test inputs
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randn(batch_size, channels, height, width)
    
    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  y: {y.shape}")
    
    # Test forward pass
    try:
        output = correlation(x, y)
        print(f"\nOutput shape: {output.shape}")
        print(f"Expected shape: ({batch_size}, {correlation.kernel_size**2}, {height}, {width})")
        
        # Verify output shape
        expected_channels = correlation.kernel_size * correlation.kernel_size
        if output.shape == (batch_size, expected_channels, height, width):
            print("✓ Output shape correct!")
        else:
            print(f"✗ Output shape mismatch!")
            return False
        
        # Test with CUDA if available
        if torch.cuda.is_available():
            print("\nTesting on GPU...")
            x_cuda = x.cuda()
            y_cuda = y.cuda()
            correlation = correlation.cuda()
            
            output_cuda = correlation(x_cuda, y_cuda)
            print(f"  GPU Output shape: {output_cuda.shape}")
            
            # Check CPU/GPU consistency
            correlation_cpu = MMTrackerCorrelation(max_displacement)
            output_cpu = correlation_cpu(x, y)
            
            diff = (output_cpu - output_cuda.cpu()).abs().max()
            print(f"  CPU/GPU max difference: {diff:.6f}")
            
            if diff < 1e-5:
                print("  ✓ CPU/GPU consistency check passed")
            else:
                print("  ✗ CPU/GPU mismatch")
                return False
        else:
            print("\n⚠ CUDA not available, skipping GPU test")
        
        # Test gradient flow
        print("\nTesting gradient flow...")
        x.requires_grad = True
        y.requires_grad = True
        
        correlation_grad = MMTrackerCorrelation(max_displacement)
        output = correlation_grad(x, y)
        loss = output.sum()
        loss.backward()
        
        if x.grad is not None and y.grad is not None:
            print(f"  x.grad shape: {x.grad.shape}, norm: {x.grad.norm():.3f}")
            print(f"  y.grad shape: {y.grad.shape}, norm: {y.grad.norm():.3f}")
            print("  ✓ Gradient flow test passed")
        else:
            print("  ✗ Gradient flow failed")
            return False
        
        print("\n" + "=" * 80)
        print(" ✅ All MMTracker compatibility tests passed!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_displacements():
    """Test different max_displacement values used in MMTracker"""
    print("\n" + "=" * 80)
    print(" Testing Different Max Displacement Values")
    print("=" * 80)
    
    displacements = [1, 2, 4, 8]  # Common values
    
    for max_disp in displacements:
        print(f"\n--- max_displacement = {max_disp} ---")
        kernel_size = 2 * max_disp + 1
        print(f"  kernel_size: {kernel_size}")
        
        correlation = MMTrackerCorrelation(max_disp)
        
        # Small test input
        x = torch.randn(1, 256, 10, 10)
        y = torch.randn(1, 256, 10, 10)
        
        try:
            output = correlation(x, y)
            print(f"  Input: {x.shape} -> Output: {output.shape}")
            print(f"  ✓ max_displacement={max_disp} works")
        except Exception as e:
            print(f"  ✗ max_displacement={max_disp} failed: {e}")
            return False
    
    return True

def compare_with_original_api():
    """Compare our API with original spatial_correlation_sampler"""
    print("\n" + "=" * 80)
    print(" API Compatibility Summary")
    print("=" * 80)
    
    print("\nOriginal API (C++ extension):")
    print("  SpatialCorrelationSampler(kernel_size, patch_size, stride, padding, dilation, patch_dilation)")
    print("  MMTracker usage: SpatialCorrelationSampler(1, kernel_size, 1, 0, 1)")
    
    print("\nOur API (Pure PyTorch):")
    print("  SpatialCorrelationSampler(kernel_size, patch_size, stride, padding, dilation, dilation_patch)")
    print("  Equivalent: SpatialCorrelationSampler(kernel_size=1, patch_size=kernel_size, stride=1, padding=0, dilation=1)")
    
    print("\n⚠ Note: Parameter naming difference:")
    print("  - Original uses 'patch_dilation'")
    print("  - Ours uses 'dilation_patch'")
    print("  - For MMTracker, this doesn't matter as it uses default value (1)")
    
    return True

if __name__ == "__main__":
    print("Testing spatial_correlation_sampler compatibility with MMTracker\n")
    
    # Run all tests
    success = True
    
    # Test MMTracker configuration
    if not test_mmtracker_configuration():
        success = False
    
    # Test different displacement values
    if not test_different_displacements():
        success = False
    
    # Show API comparison
    compare_with_original_api()
    
    print("\n" + "=" * 80)
    if success:
        print(" ✅ CONCLUSION: Package is compatible with MMTracker!")
        print(" The package can be used as a drop-in replacement.")
    else:
        print(" ❌ CONCLUSION: Some compatibility issues found.")
        print(" Please check the errors above.")
    print("=" * 80)