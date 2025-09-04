"""
基本功能测试
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatial_correlation_sampler import SpatialCorrelationSampler


def test_mmtracker_config():
    """测试MMTracker配置"""
    print("测试MMTracker配置...")
    
    # 正确的MMTracker参数
    sampler = SpatialCorrelationSampler(
        kernel_size=9,
        patch_size=1, 
        stride=1,
        padding=4,
        dilation=1
    )
    
    # 测试数据
    B, C, H, W = 2, 256, 30, 40
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)
    
    output = sampler(x, y)
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    
    # 验证输出
    assert len(output.shape) == 5, f"期望5D输出，得到{len(output.shape)}D"
    assert output.shape == (B, 1, 81, H, W), f"期望形状{(B, 1, 81, H, W)}，得到{output.shape}"
    
    # 转换为4D格式（MMTracker期望）
    output_4d = output[:, 0, :, :, :]
    print(f"4D格式: {output_4d.shape}")
    assert output_4d.shape == (B, 81, H, W)
    
    print("✓ MMTracker配置测试通过")


def test_flownet_config():
    """测试FlowNet配置"""
    print("\n测试FlowNet配置...")
    
    # FlowNet配置
    sampler = SpatialCorrelationSampler(
        kernel_size=1,
        patch_size=21,
        stride=1,
        padding=0,
        dilation=1,
        dilation_patch=2
    )
    
    # 测试数据
    B, C, H, W = 1, 64, 32, 32
    x = torch.randn(B, C, H, W)
    y = torch.randn(B, C, H, W)
    
    output = sampler(x, y)
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    
    # FlowNet输出格式检查
    if len(output.shape) == 5:
        # (B, PatchH, PatchW, H, W) 格式
        expected_patch_h, expected_patch_w = 21, 21
        assert output.shape[1:3] == (expected_patch_h, expected_patch_w), f"期望patch尺寸{(expected_patch_h, expected_patch_w)}，得到{output.shape[1:3]}"
    elif len(output.shape) == 4:
        # (B, PatchH*PatchW, H, W) 格式
        expected_channels = 21 * 21  # 441
        assert output.shape[1] == expected_channels, f"期望{expected_channels}通道，得到{output.shape[1]}"
    
    print("✓ FlowNet配置测试通过")


def test_basic_correlation():
    """测试基本相关计算"""
    print("\n测试基本相关计算...")
    
    # 简单配置
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
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    
    # 验证输出合理性
    assert not torch.isnan(output).any(), "输出包含NaN值"
    assert torch.isfinite(output).all(), "输出包含无穷值"
    
    print(f"数值范围: [{output.min():.4f}, {output.max():.4f}]")
    print("✓ 基本相关计算测试通过")


if __name__ == "__main__":
    print("Spatial Correlation Sampler 基本功能测试")
    print("=" * 50)
    
    try:
        test_mmtracker_config()
        test_flownet_config()
        test_basic_correlation()
        
        print("\n" + "=" * 50)
        print("✓ 所有测试通过！")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()