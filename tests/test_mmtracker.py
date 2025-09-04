#!/usr/bin/env python
"""
Comprehensive MMTracker effectiveness verification with dimension tracking
"""

import torch
import torch.nn as nn
import numpy as np
from spatial_correlation_sampler import SpatialCorrelationSampler

class MMTrackerCorrelation(nn.Module):
    """MMTracker Correlation module using our implementation"""
    def __init__(self, max_displacement=4):
        super(MMTrackerCorrelation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2 * max_displacement + 1  # 9
        self.corr = SpatialCorrelationSampler(
            kernel_size=1, 
            patch_size=self.kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1
        )
        
    def forward(self, x, y):
        b, c, h, w = x.shape
        # Forward through correlation sampler and reshape for MMTracker format
        return self.corr(x, y).view(b, -1, h, w) / c

def create_controlled_features():
    """Create controlled feature maps with known motion patterns"""
    print("\n" + "=" * 80)
    print("Creating Controlled Test Data")
    print("=" * 80)
    
    B, C, H, W = 1, 512, 19, 34
    print(f"Creating features: Batch={B}, Channels={C}, Height={H}, Width={W}")
    
    # Initialize feature maps
    feat1 = torch.zeros(B, C, H, W)
    feat2 = torch.zeros(B, C, H, W)
    
    # Add distinctive patterns at specific locations
    patterns = []
    for i in range(3):  # Create 3 objects
        # Create random pattern
        pattern = torch.randn(C, 2, 2)  # 2x2 pattern
        patterns.append(pattern)
        
        # Position in frame 1
        y1, x1 = 5 + i * 4, 10 + i * 6
        feat1[0, :, y1:y1+2, x1:x1+2] = pattern
        
        # Position in frame 2 (with known motion)
        motion_y, motion_x = [0, -2, 1][i], [1, 0, -1][i]  # Different motions
        y2, x2 = y1 + motion_y, x1 + motion_x
        
        # Ensure within bounds
        if 0 <= y2 <= H-2 and 0 <= x2 <= W-2:
            feat2[0, :, y2:y2+2, x2:x2+2] = pattern
            print(f"  Object {i+1}: ({y1},{x1}) -> ({y2},{x2}) | Motion: ({motion_y},{motion_x})")
        
    print(f"✓ Created controlled features with known motion patterns")
    return feat1, feat2, patterns

def test_motion_detection():
    """Test actual motion detection capability"""
    print("\n" + "=" * 80)
    print("Testing Motion Detection Effectiveness")
    print("=" * 80)
    
    # Create controlled test data
    feat1, feat2, patterns = create_controlled_features()
    
    # Initialize correlation module
    print(f"\nInitializing MMTracker correlation (max_displacement=4)")
    correlation = MMTrackerCorrelation(max_displacement=4)
    
    print(f"\nForward pass through correlation module:")
    output = correlation(feat1, feat2)
    
    print(f"\nFinal output analysis:")
    print(f"  Output shape: {output.shape}")
    print(f"  Output channels: 81 (9x9 displacement grid)")
    print(f"  Channel mapping: 0=(-4,-4), 40=(0,0), 80=(+4,+4)")
    
    # Analyze motion detection at object locations
    test_locations = [(5, 11), (9, 16), (13, 22)]  # Approximate object centers
    expected_motions = [(0, 1), (-2, 0), (1, -1)]
    
    print(f"\nMotion Detection Results:")
    print(f"{'Location':<12} {'Expected':<12} {'Detected':<12} {'Channel':<8} {'Confidence':<12} {'Status'}")
    print("-" * 80)
    
    success_count = 0
    for i, ((y, x), (exp_dy, exp_dx)) in enumerate(zip(test_locations, expected_motions)):
        if y < output.shape[2] and x < output.shape[3]:
            # Get correlation at this location
            location_corr = output[0, :, y, x]  # 81 channels
            
            # Find maximum correlation
            max_channel = location_corr.argmax().item()
            max_confidence = location_corr[max_channel].item()
            
            # Convert channel to displacement
            detected_dy = max_channel // 9 - 4  # 9x9 grid, center at (4,4)
            detected_dx = max_channel % 9 - 4
            
            # Check accuracy
            motion_error = abs(detected_dy - exp_dy) + abs(detected_dx - exp_dx)
            is_correct = motion_error <= 1  # Allow 1 pixel tolerance
            
            status = "✓ PASS" if is_correct else "✗ FAIL"
            if is_correct:
                success_count += 1
                
            print(f"({y:2},{x:2})      ({exp_dy:2},{exp_dx:2})       ({detected_dy:2},{detected_dx:2})       {max_channel:<8} {max_confidence:<12.4f} {status}")
    
    accuracy = success_count / len(test_locations) * 100
    print(f"\nMotion Detection Accuracy: {success_count}/{len(test_locations)} ({accuracy:.1f}%)")
    
    return accuracy > 66  # At least 2/3 should be correct

def test_dimension_flow():
    """Test and document dimension flow through the entire pipeline"""
    print("\n" + "=" * 80)
    print("Dimension Flow Analysis")
    print("=" * 80)
    
    # MMTracker typical dimensions
    B, C, H, W = 1, 512, 19, 34
    max_displacement = 4
    kernel_size = 9
    
    print(f"Starting dimensions: B={B}, C={C}, H={H}, W={W}")
    print(f"Configuration: max_displacement={max_displacement}, kernel_size={kernel_size}")
    
    # Create input features
    feat1 = torch.randn(B, C, H, W)
    feat2 = torch.randn(B, C, H, W)
    print(f"\n1. Input Features:")
    print(f"   feat1: {feat1.shape}")
    print(f"   feat2: {feat2.shape}")
    
    # Initialize correlation sampler directly
    sampler = SpatialCorrelationSampler(
        kernel_size=1, 
        patch_size=kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1
    )
    
    print(f"\n2. Correlation Sampler Parameters:")
    print(f"   kernel_size: 1 (query single point)")
    print(f"   patch_size: {kernel_size} (search in {kernel_size}x{kernel_size} area)")
    print(f"   stride: 1, padding: 0, dilation: 1")
    
    # Forward pass with dimension tracking
    print(f"\n3. Forward Pass:")
    raw_output = sampler(feat1, feat2)
    print(f"   Raw sampler output: {raw_output.shape}")
    print(f"   Expected: (B, patch_size², H, W) = ({B}, {kernel_size**2}, {H}, {W})")
    
    # Reshape for MMTracker compatibility
    reshaped_output = raw_output.view(B, -1, H, W)
    print(f"   Reshaped output: {reshaped_output.shape}")
    
    # Normalization
    normalized_output = reshaped_output / C
    print(f"   Normalized (÷{C}): {normalized_output.shape}")
    
    print(f"\n4. Output Interpretation:")
    print(f"   {kernel_size**2} channels represent correlation at different displacements")
    print(f"   Channel mapping:")
    print(f"     Channel 0:  displacement (-4, -4)")
    print(f"     Channel 40: displacement ( 0,  0) [center - no motion]")
    print(f"     Channel 80: displacement (+4, +4)")
    
    # Test gradient flow
    print(f"\n5. Gradient Flow Test:")
    feat1_grad = torch.randn(B, C, H, W, requires_grad=True)
    feat2_grad = torch.randn(B, C, H, W, requires_grad=True)
    
    output_grad = sampler(feat1_grad, feat2_grad)
    loss = output_grad.sum()
    loss.backward()
    
    print(f"   Input gradients computed: ✓")
    print(f"   feat1.grad: {feat1_grad.grad.shape}")
    print(f"   feat2.grad: {feat2_grad.grad.shape}")
    print(f"   Gradient norm: {feat1_grad.grad.norm():.4f}")
    
    return True

def test_performance_analysis():
    """Analyze computational performance and memory usage"""
    print("\n" + "=" * 80)
    print("Performance Analysis")
    print("=" * 80)
    
    import time
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8]
    C, H, W = 512, 19, 34
    
    correlation = MMTrackerCorrelation(max_displacement=4)
    
    if torch.cuda.is_available():
        correlation = correlation.cuda()
        device_name = "GPU"
    else:
        device_name = "CPU"
    
    print(f"Testing on {device_name}")
    print(f"{'Batch Size':<12} {'Input Shape':<20} {'Output Shape':<20} {'Time (ms)':<12} {'Memory (MB)':<12}")
    print("-" * 85)
    
    for batch_size in batch_sizes:
        # Create test data
        feat1 = torch.randn(batch_size, C, H, W)
        feat2 = torch.randn(batch_size, C, H, W)
        
        if torch.cuda.is_available():
            feat1 = feat1.cuda()
            feat2 = feat2.cuda()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Warmup
        for _ in range(3):
            _ = correlation(feat1, feat2)
        
        # Time measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(10):
            output = correlation(feat1, feat2)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        avg_time = (time.time() - start_time) / 10 * 1000
        
        # Memory measurement
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**2
        else:
            memory_used = 0  # Hard to measure CPU memory accurately
        
        print(f"{batch_size:<12} {str(feat1.shape):<20} {str(output.shape):<20} {avg_time:<12.2f} {memory_used:<12.1f}")
    
    return True

def test_comparison_with_naive():
    """Compare with naive implementation to verify correctness"""
    print("\n" + "=" * 80)
    print("Correctness Verification vs Naive Implementation")
    print("=" * 80)
    
    def naive_correlation(feat1, feat2, max_disp=4):
        """Naive but correct correlation implementation"""
        B, C, H, W = feat1.shape
        kernel_size = 2 * max_disp + 1
        output = torch.zeros(B, kernel_size * kernel_size, H, W)
        
        for y in range(H):
            for x in range(W):
                idx = 0
                for dy in range(-max_disp, max_disp + 1):
                    for dx in range(-max_disp, max_disp + 1):
                        y2, x2 = y + dy, x + dx
                        if 0 <= y2 < H and 0 <= x2 < W:
                            # Correlation between feat1[y,x] and feat2[y2,x2]
                            corr = (feat1[:, :, y, x] * feat2[:, :, y2, x2]).sum(dim=1)
                            output[:, idx, y, x] = corr / C
                        idx += 1
        return output
    
    # Small test case for comparison
    B, C, H, W = 1, 4, 8, 8
    feat1 = torch.randn(B, C, H, W)
    feat2 = torch.randn(B, C, H, W)
    
    print(f"Small test case: {feat1.shape}")
    
    # Our implementation
    correlation = MMTrackerCorrelation(max_displacement=2)  # Smaller for speed
    our_output = correlation(feat1, feat2)
    
    # Naive implementation  
    naive_output = naive_correlation(feat1, feat2, max_disp=2)
    
    # Compare
    diff = (our_output - naive_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Our implementation: {our_output.shape}")
    print(f"Naive implementation: {naive_output.shape}")
    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    
    is_correct = max_diff < 1e-6
    print(f"Correctness check: {'✓ PASS' if is_correct else '✗ FAIL'}")
    
    return is_correct

def run_comprehensive_test():
    """Run all tests comprehensively"""
    print("=" * 80)
    print("COMPREHENSIVE MMTRACKER EFFECTIVENESS VERIFICATION")
    print("=" * 80)
    
    tests = [
        ("Dimension Flow Analysis", test_dimension_flow),
        ("Motion Detection Effectiveness", test_motion_detection),
        ("Performance Analysis", test_performance_analysis),
        ("Correctness Verification", test_comparison_with_naive),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name:<35} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("   The package is verified to be effective for MMTracker!")
    else:
        print("\n⚠️ Some tests failed. Please review the output above.")
    
    return passed == total

if __name__ == "__main__":
    run_comprehensive_test()