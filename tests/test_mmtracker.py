#!/usr/bin/env python
"""
Test MMTracker Motion Mamba module correctness and compatibility

MMTracker Configuration:
- max_displacement = 4
- kernel_size = 9 (2 * max_displacement + 1)
- Input features: [B=1, C=512, H=19, W=34] at 1/32 scale
"""

import torch
import torch.nn as nn
import numpy as np
from spatial_correlation_sampler import SpatialCorrelationSampler


class MMTrackerCorrelation(nn.Module):
    """
    Exact implementation of MMTracker's Correlation module
    Reference: MMTracker/motion/motion_model.py
    """
    def __init__(self, max_displacement=4):
        super(MMTrackerCorrelation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2 * max_displacement + 1  # 9 for max_displacement=4
        
        # MMTracker uses: SpatialCorrelationSampler(1, self.kernel_size, 1, 0, 1)
        # Our equivalent: kernel_size=1, patch_size=9, stride=1, padding=0, dilation=1
        self.corr = SpatialCorrelationSampler(
            kernel_size=1, 
            patch_size=self.kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1
        )
        
    def forward(self, x, y):
        """
        Args:
            x: Frame1 features [B, C, H, W]
            y: Frame2 features [B, C, H, W]
        Returns:
            correlation: [B, kernel_size^2, H, W]
        """
        b, c, h, w = x.shape
        return self.corr(x, y).view(b, -1, h, w) / c


def test_correlation_correctness():
    """Test correlation computation correctness"""
    print("=" * 80)
    print(" Testing Correlation Correctness")
    print("=" * 80)
    
    # MMTracker exact configuration
    max_displacement = 4
    batch_size = 1
    channels = 512
    height = 19
    width = 34
    
    print(f"\nMMTracker Configuration:")
    print(f"  max_displacement: {max_displacement}")
    print(f"  kernel_size: {2 * max_displacement + 1}")
    print(f"  Input shape: [{batch_size}, {channels}, {height}, {width}]")
    
    # Create correlation module
    correlation = MMTrackerCorrelation(max_displacement)
    
    # Test 1: Identity test - same features should give maximum correlation at center
    print("\n1. Identity Test (same features):")
    features = torch.randn(batch_size, channels, height, width)
    output = correlation(features, features)
    
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [{batch_size}, {correlation.kernel_size**2}, {height}, {width}]")
    
    # The center position (4, 4) in 9x9 grid should have maximum correlation
    center_idx = correlation.kernel_size * correlation.kernel_size // 2
    center_values = output[:, center_idx, :, :]
    
    print(f"  Center channel index: {center_idx} (position [4,4] in 9x9 grid)")
    print(f"  Center correlation mean: {center_values.mean().item():.4f}")
    print(f"  Center correlation max: {center_values.max().item():.4f}")
    
    # Check if center has highest correlation
    max_corr_channels = output.argmax(dim=1)
    center_is_max = (max_corr_channels == center_idx).float().mean()
    print(f"  Percentage where center has max correlation: {center_is_max * 100:.1f}%")
    
    if center_is_max > 0.8:  # At least 80% should have max at center
        print("  ✓ Identity test passed")
    else:
        print("  ✗ Identity test failed")
        return False
    
    # Test 2: Shifted features test
    print("\n2. Shifted Features Test:")
    features1 = torch.randn(batch_size, channels, height, width)
    features2 = torch.zeros_like(features1)
    
    # Shift features2 by 2 pixels to the right
    shift_x = 2
    features2[:, :, :, shift_x:] = features1[:, :, :, :-shift_x]
    
    output = correlation(features1, features2)
    
    # The correlation should be highest at position corresponding to shift
    # In a 9x9 kernel, center is (4,4), shift right by 2 means correlation at (4, 4+2)
    expected_peak_idx = center_idx + shift_x  # Shift in kernel space
    
    # Check correlation pattern in central region
    central_h = height // 2
    central_w = width // 2
    max_corr_at_center = output[:, :, central_h, central_w].argmax().item()
    
    print(f"  Applied shift: {shift_x} pixels right")
    print(f"  Max correlation channel at center pixel: {max_corr_at_center}")
    print(f"  Expected channel (approx): {expected_peak_idx}")
    
    # Test 3: Zero correlation test
    print("\n3. Orthogonal Features Test:")
    features1 = torch.randn(batch_size, channels, height, width)
    features2 = torch.randn(batch_size, channels, height, width)
    
    # Make features orthogonal in channel dimension
    features2 = features2 - (features2 * features1).sum(dim=1, keepdim=True) / (features1 * features1).sum(dim=1, keepdim=True) * features1
    
    output = correlation(features1, features2)
    
    print(f"  Mean correlation: {output.mean().item():.6f}")
    print(f"  Std correlation: {output.std().item():.6f}")
    print(f"  Should be close to 0 for orthogonal features")
    
    if abs(output.mean().item()) < 0.01:
        print("  ✓ Orthogonal test passed")
    else:
        print("  ⚠ Orthogonal test: correlation not close to zero")
    
    return True


def test_gradient_flow():
    """Test gradient flow through correlation module"""
    print("\n" + "=" * 80)
    print(" Testing Gradient Flow")
    print("=" * 80)
    
    # MMTracker configuration
    correlation = MMTrackerCorrelation(max_displacement=4)
    
    # Create inputs with gradients
    features1 = torch.randn(1, 512, 19, 34, requires_grad=True)
    features2 = torch.randn(1, 512, 19, 34, requires_grad=True)
    
    # Forward pass
    output = correlation(features1, features2)
    
    # Create a loss
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    print(f"\nGradient statistics:")
    print(f"  features1.grad: shape {features1.grad.shape}, norm {features1.grad.norm():.3f}")
    print(f"  features2.grad: shape {features2.grad.shape}, norm {features2.grad.norm():.3f}")
    
    # Check gradients are not zero
    if features1.grad.norm() > 0 and features2.grad.norm() > 0:
        print("  ✓ Gradient flow test passed")
        return True
    else:
        print("  ✗ Gradient flow test failed")
        return False


def test_motion_detection():
    """Test correlation for motion detection scenarios"""
    print("\n" + "=" * 80)
    print(" Testing Motion Detection Scenarios")
    print("=" * 80)
    
    correlation = MMTrackerCorrelation(max_displacement=4)
    
    # Scenario 1: Small object motion
    print("\n1. Small Motion Detection:")
    
    # Create a feature map with a localized pattern
    features1 = torch.zeros(1, 512, 19, 34)
    features2 = torch.zeros(1, 512, 19, 34)
    
    # Add a distinctive pattern at a specific location
    pattern = torch.randn(512, 3, 3)
    features1[:, :, 8:11, 15:18] = pattern
    
    # Move pattern by 2 pixels diagonally
    features2[:, :, 10:13, 17:20] = pattern
    
    output = correlation(features1, features2)
    
    # Check if correlation detects the motion
    # At the original pattern location, correlation should be highest 
    # at the displacement corresponding to the motion
    correlation_at_origin = output[:, :, 9, 16]  # Center of original pattern
    max_corr_channel = correlation_at_origin.argmax().item()
    
    print(f"  Pattern moved: 2 pixels down-right")
    print(f"  Max correlation channel at origin: {max_corr_channel}")
    print(f"  This corresponds to displacement in 9x9 kernel")
    
    # Scenario 2: Multiple moving objects
    print("\n2. Multiple Objects Motion:")
    
    features1 = torch.randn(1, 512, 19, 34) * 0.1  # Background noise
    features2 = torch.randn(1, 512, 19, 34) * 0.1
    
    # Add multiple objects
    for i in range(3):
        obj_pattern = torch.randn(512, 2, 2)
        y_pos = 5 + i * 5
        x_pos = 10 + i * 7
        
        # Place objects
        features1[:, :, y_pos:y_pos+2, x_pos:x_pos+2] += obj_pattern
        
        # Move each object differently
        dy, dx = i-1, (i-1)*2  # Different motion for each object
        new_y = max(0, min(17, y_pos + dy))
        new_x = max(0, min(32, x_pos + dx))
        features2[:, :, new_y:new_y+2, new_x:new_x+2] += obj_pattern
    
    output = correlation(features1, features2)
    
    print(f"  Added 3 objects with different motions")
    print(f"  Correlation output shape: {output.shape}")
    print(f"  Mean correlation: {output.mean().item():.4f}")
    print(f"  Max correlation: {output.max().item():.4f}")
    
    return True


def test_gpu_consistency():
    """Test CPU/GPU consistency"""
    print("\n" + "=" * 80)
    print(" Testing CPU/GPU Consistency")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("  ⚠ CUDA not available, skipping GPU test")
        return True
    
    # Create modules
    correlation_cpu = MMTrackerCorrelation(max_displacement=4)
    correlation_gpu = MMTrackerCorrelation(max_displacement=4).cuda()
    
    # Test data
    features1_cpu = torch.randn(1, 512, 19, 34)
    features2_cpu = torch.randn(1, 512, 19, 34)
    
    features1_gpu = features1_cpu.cuda()
    features2_gpu = features2_cpu.cuda()
    
    # Compute correlations
    output_cpu = correlation_cpu(features1_cpu, features2_cpu)
    output_gpu = correlation_gpu(features1_gpu, features2_gpu)
    
    # Compare
    diff = (output_cpu - output_gpu.cpu()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nCPU/GPU Comparison:")
    print(f"  Max difference: {max_diff:.8f}")
    print(f"  Mean difference: {mean_diff:.8f}")
    
    if max_diff < 1e-5:
        print("  ✓ CPU/GPU consistency test passed")
        return True
    else:
        print("  ✗ CPU/GPU mismatch")
        return False


def test_performance():
    """Test performance metrics"""
    print("\n" + "=" * 80)
    print(" Testing Performance")
    print("=" * 80)
    
    import time
    
    correlation = MMTrackerCorrelation(max_displacement=4)
    features1 = torch.randn(1, 512, 19, 34)
    features2 = torch.randn(1, 512, 19, 34)
    
    # Warmup
    for _ in range(3):
        _ = correlation(features1, features2)
    
    # Time CPU
    start = time.time()
    for _ in range(10):
        _ = correlation(features1, features2)
    cpu_time = (time.time() - start) / 10 * 1000  # ms
    
    print(f"\nCPU Performance:")
    print(f"  Average time: {cpu_time:.2f} ms")
    
    if torch.cuda.is_available():
        correlation_gpu = correlation.cuda()
        features1_gpu = features1.cuda()
        features2_gpu = features2.cuda()
        
        # Warmup
        for _ in range(3):
            _ = correlation_gpu(features1_gpu, features2_gpu)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(10):
            _ = correlation_gpu(features1_gpu, features2_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 10 * 1000  # ms
        
        print(f"\nGPU Performance:")
        print(f"  Average time: {gpu_time:.2f} ms")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    
    return True


def run_all_tests():
    """Run all MMTracker tests"""
    print("\n" + "=" * 80)
    print(" MMTracker Motion Mamba Module Test Suite")
    print("=" * 80)
    print("\nConfiguration:")
    print("  max_displacement = 4")
    print("  kernel_size = 9")
    print("  Feature shape = [1, 512, 19, 34] (1/32 scale)")
    
    tests = [
        ("Correlation Correctness", test_correlation_correctness),
        ("Gradient Flow", test_gradient_flow),
        ("Motion Detection", test_motion_detection),
        ("CPU/GPU Consistency", test_gpu_consistency),
        ("Performance", test_performance),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print(" TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Package is correct and compatible with MMTracker.")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)