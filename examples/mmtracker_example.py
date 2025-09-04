"""
Example usage for MM-Tracker
"""

import torch
import torch.nn as nn
from spatial_correlation_sampler import OptimizedSpatialCorrelationSampler


class MMTrackerMotionModule(nn.Module):
    """
    Motion module for MM-Tracker using correlation sampling
    """

    def __init__(self, feature_channels=512):
        super().__init__()

        # Correlation sampler with MM-Tracker configuration
        self.correlation = OptimizedSpatialCorrelationSampler(
            kernel_size=9,
            stride=1,
            padding=4,
            aggregation='max'
        )

        # Motion prediction head
        self.motion_head = nn.Sequential(
            nn.Conv2d(feature_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1)  # Output 2D motion vectors
        )

    def forward(self, feat_current, feat_previous):
        """
        Compute motion features

        Args:
            feat_current: Current frame features [B, C, H, W]
            feat_previous: Previous frame features [B, C, H, W]

        Returns:
            motion_vectors: 2D motion field [B, 2, H, W]
        """
        # Compute correlation
        correlation = self.correlation(feat_current, feat_previous)

        # Predict motion vectors
        motion_vectors = self.motion_head(correlation)

        return motion_vectors


def main():
    """Example usage"""

    # Create model
    model = MMTrackerMotionModule(feature_channels=512)

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create example inputs (1/32 scale features)
    batch_size = 1
    feat_current = torch.randn(batch_size, 512, 19, 34, device=device)
    feat_previous = torch.randn(batch_size, 512, 19, 34, device=device)

    # Compute motion
    with torch.no_grad():
        motion = model(feat_current, feat_previous)

    print(f"Input shape: {feat_current.shape}")
    print(f"Output shape: {motion.shape}")
    print(f"Motion range: [{motion.min():.3f}, {motion.max():.3f}]")

    # Convert to pixel displacements (scale up by 32)
    pixel_motion = motion * 32
    print(f"Pixel motion range: [{pixel_motion.min():.1f}, {pixel_motion.max():.1f}]")


if __name__ == "__main__":
    main()