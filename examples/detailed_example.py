#!/usr/bin/env python
"""
Detailed example demonstrating Spatial Correlation Sampler usage

This example shows how to use the correlation sampler for different scenarios:
1. Basic usage with small feature maps
2. FlowNet-style optical flow estimation  
3. MM-Tracker-style object tracking
4. Understanding output interpretation
"""

import torch
import numpy as np
from spatial_correlation_sampler import SpatialCorrelationSampler

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def basic_example():
    """Basic usage example with visualization"""
    print("=" * 60)
    print("Basic Example: Understanding Correlation Output")
    print("=" * 60)
    
    # Create simple 5x5 feature maps
    batch_size, channels, height, width = 1, 1, 5, 5
    
    # First feature map with a central peak
    feat1 = torch.zeros(batch_size, channels, height, width)
    feat1[0, 0, 2, 2] = 1.0  # Peak at center (2,2)
    
    # Second feature map with the same peak shifted by (1,1)
    feat2 = torch.zeros(batch_size, channels, height, width)
    feat2[0, 0, 3, 3] = 1.0  # Peak at (3,3)
    
    print(f"Feature map 1 (peak at center):")
    print(feat1[0, 0].numpy())
    print(f"\nFeature map 2 (peak shifted to (3,3)):")
    print(feat2[0, 0].numpy())
    
    # Configure correlation sampler
    # 3x3 search window, single pixel comparison
    sampler = SpatialCorrelationSampler(
        kernel_size=3,
        patch_size=1,
        stride=1,
        padding=1
    )
    
    # Compute correlation
    correlation = sampler(feat1, feat2)
    
    print(f"\nInput shape: {feat1.shape}")
    print(f"Output shape: {correlation.shape}")
    print(f"Output channels: 9 (3x3 search window)")
    
    # Analyze correlation at center pixel (2,2)
    center_correlations = correlation[0, :, 2, 2]
    print(f"\nCorrelation at center pixel (2,2):")
    print(f"Displacement grid (3x3):")
    displacement_grid = center_correlations.view(3, 3)
    print(displacement_grid.numpy())
    
    # Find maximum correlation
    max_channel = center_correlations.argmax().item()
    max_displacement = divmod(max_channel, 3)  # Convert to (row, col)
    actual_displacement = (max_displacement[0] - 1, max_displacement[1] - 1)  # Center at (1,1)
    
    print(f"\nMaximum correlation at channel {max_channel}")
    print(f"This corresponds to displacement: {actual_displacement}")
    print(f"Expected displacement: (1, 1) ✓" if actual_displacement == (1, 1) else "❌")


def flownet_example():
    """FlowNet-style optical flow estimation"""
    print("\n" + "=" * 60)
    print("FlowNet Example: Optical Flow Estimation")
    print("=" * 60)
    
    # FlowNet typical configuration
    batch_size = 1
    channels = 256
    height, width = 64, 64
    
    # Create feature maps (simulating CNN features)
    feat1 = torch.randn(batch_size, channels, height, width)
    feat2 = torch.randn(batch_size, channels, height, width)
    
    # FlowNet correlation configuration
    # Search in 21x21 window with sparse sampling
    sampler = SpatialCorrelationSampler(
        kernel_size=1,          # Single query point
        patch_size=21,          # 21x21 search window
        stride=1,
        padding=0,
        dilation=1,
        dilation_patch=2        # Sparse sampling (every 2nd pixel)
    )
    
    correlation = sampler(feat1, feat2)
    
    print(f"FlowNet Configuration:")
    print(f"  Input features: {feat1.shape}")
    print(f"  Search window: 21x21 with dilation=2")
    print(f"  Output correlation: {correlation.shape}")
    print(f"  Correlation channels: {21 * 21} = 441")
    
    # The output represents correlation at different displacements
    # Channel mapping: [-10, -10] to [+10, +10] in pixel space
    print(f"\nDisplacement mapping:")
    print(f"  Channel 0: displacement (-10, -10)")
    print(f"  Channel 220: displacement (0, 0) - center")
    print(f"  Channel 440: displacement (+10, +10)")
    
    # Simulate finding maximum correlation
    center_pixel_corr = correlation[0, :, height//2, width//2]
    max_channel = center_pixel_corr.argmax().item()
    
    # Convert channel index to displacement
    displacement_y = max_channel // 21 - 10
    displacement_x = max_channel % 21 - 10
    
    print(f"\nExample correlation at center pixel:")
    print(f"  Max correlation channel: {max_channel}")
    print(f"  Inferred motion: ({displacement_x}, {displacement_y}) pixels")


def mmtracker_example():
    """MM-Tracker-style object tracking"""
    print("\n" + "=" * 60)
    print("MM-Tracker Example: Object Tracking")
    print("=" * 60)
    
    # MM-Tracker typical configuration (1/32 scale features)
    batch_size = 1
    channels = 512
    height, width = 19, 34  # 608x1088 input -> 19x34 features
    
    # Create feature maps
    feat1 = torch.randn(batch_size, channels, height, width)
    feat2 = torch.randn(batch_size, channels, height, width)
    
    # MM-Tracker correlation configuration
    max_displacement = 4
    kernel_size = 2 * max_displacement + 1  # 9
    
    sampler = SpatialCorrelationSampler(
        kernel_size=kernel_size,  # 9x9 search area
        patch_size=1,             # Single pixel comparison
        stride=1,
        padding=max_displacement  # Padding to preserve spatial size
    )
    
    correlation = sampler(feat1, feat2)
    
    print(f"MM-Tracker Configuration:")
    print(f"  Max displacement: ±{max_displacement} pixels")
    print(f"  Search window: {kernel_size}x{kernel_size}")
    print(f"  Input features: {feat1.shape}")
    print(f"  Output correlation: {correlation.shape}")
    print(f"  Spatial size preserved: {height}x{width}")
    
    # Create Motion Mamba style correlation module
    class MotionCorrelation:
        def __init__(self, max_displacement):
            self.max_displacement = max_displacement
            self.kernel_size = 2 * max_displacement + 1
            self.sampler = SpatialCorrelationSampler(
                kernel_size=1,
                patch_size=self.kernel_size,
                stride=1,
                padding=0
            )
        
        def __call__(self, x, y):
            b, c, h, w = x.shape
            return self.sampler(x, y).view(b, -1, h, w) / c
    
    # Use MM-Tracker style
    mm_correlation = MotionCorrelation(max_displacement=4)
    mm_output = mm_correlation(feat1, feat2)
    
    print(f"\nMM-Tracker Motion Mamba style:")
    print(f"  Output shape: {mm_output.shape}")
    print(f"  Normalized by channel count: /{channels}")
    
    # Analyze motion at a specific location
    y_pos, x_pos = height//2, width//2
    motion_vector = mm_output[0, :, y_pos, x_pos]
    max_motion_channel = motion_vector.argmax().item()
    
    # Convert to displacement
    displacement_y = max_motion_channel // 9 - 4
    displacement_x = max_motion_channel % 9 - 4
    
    print(f"\nMotion analysis at pixel ({y_pos}, {x_pos}):")
    print(f"  Strongest correlation at channel: {max_motion_channel}")
    print(f"  Inferred motion: ({displacement_x}, {displacement_y}) pixels")


def performance_comparison():
    """Compare different configurations and their performance"""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    import time
    
    configs = [
        ("Small FlowNet", {"kernel_size": 1, "patch_size": 9}),
        ("Large FlowNet", {"kernel_size": 1, "patch_size": 21}),
        ("MM-Tracker", {"kernel_size": 9, "patch_size": 1}),
        ("Dense Search", {"kernel_size": 5, "patch_size": 5}),
    ]
    
    # Test input
    feat1 = torch.randn(1, 256, 64, 64)
    feat2 = torch.randn(1, 256, 64, 64)
    
    if torch.cuda.is_available():
        feat1 = feat1.cuda()
        feat2 = feat2.cuda()
        device_name = "GPU"
    else:
        device_name = "CPU"
    
    print(f"Testing on {device_name} with input shape: {feat1.shape}")
    print()
    
    for name, config in configs:
        sampler = SpatialCorrelationSampler(**config)
        if torch.cuda.is_available():
            sampler = sampler.cuda()
        
        # Warmup
        for _ in range(3):
            _ = sampler(feat1, feat2)
        
        # Time measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(5):
            output = sampler(feat1, feat2)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        avg_time = (time.time() - start_time) / 5 * 1000  # ms
        
        print(f"{name:15} | Output: {str(output.shape):20} | Time: {avg_time:.2f}ms")


def visualization_example():
    """Create visualizations to understand correlation"""
    print("\n" + "=" * 60)
    print("Visualization Example")
    print("=" * 60)
    
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return
    
    try:
        
        # Create feature maps with known motion
        height, width = 32, 32
        feat1 = torch.zeros(1, 1, height, width)
        feat2 = torch.zeros(1, 1, height, width)
        
        # Add a distinctive pattern
        pattern = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        feat1[0, 0, 15:17, 15:17] = pattern
        feat2[0, 0, 17:19, 13:15] = pattern  # Moved down 2, left 2
        
        # Compute correlation
        sampler = SpatialCorrelationSampler(kernel_size=1, patch_size=7, stride=1, padding=0)
        correlation = sampler(feat1, feat2)
        
        # Find correlation at the original pattern location
        corr_at_pattern = correlation[0, :, 15, 15].view(7, 7).detach()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Show input feature maps
        axes[0].imshow(feat1[0, 0], cmap='viridis')
        axes[0].set_title('Frame 1 (Original Pattern)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        axes[1].imshow(feat2[0, 0], cmap='viridis')
        axes[1].set_title('Frame 2 (Moved Pattern)')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        
        # Show correlation map
        im = axes[2].imshow(corr_at_pattern, cmap='hot')
        axes[2].set_title('Correlation Map (7x7 search)')
        axes[2].set_xlabel('X displacement')
        axes[2].set_ylabel('Y displacement')
        
        # Add displacement labels
        ticks = range(7)
        labels = [str(i-3) for i in ticks]  # -3 to +3
        axes[2].set_xticks(ticks)
        axes[2].set_xticklabels(labels)
        axes[2].set_yticks(ticks)
        axes[2].set_yticklabels(labels)
        
        plt.colorbar(im, ax=axes[2])
        plt.tight_layout()
        plt.savefig('correlation_visualization.png', dpi=150, bbox_inches='tight')
        
        print("Visualization saved as 'correlation_visualization.png'")
        print("The correlation map shows highest values where the pattern matches")
        print("Expected peak should be at displacement (+2, -2)")
        
        # Find actual peak
        max_pos = torch.unravel_index(corr_at_pattern.argmax(), corr_at_pattern.shape)
        displacement = (max_pos[0].item() - 3, max_pos[1].item() - 3)
        print(f"Actual peak at displacement: {displacement}")
        
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    print("Spatial Correlation Sampler - Detailed Examples")
    print("=" * 60)
    
    # Run all examples
    basic_example()
    flownet_example()
    mmtracker_example()
    performance_comparison()
    visualization_example()
    
    print("\n" + "=" * 60)
    print("Examples completed! Check the technical guide for more details.")
    print("=" * 60)