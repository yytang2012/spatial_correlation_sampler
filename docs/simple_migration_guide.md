# 简单迁移指南

## 概述

基于分析，MMTracker_original 的参数顺序存在问题，但修改接口比添加复杂的自动修正机制更直接。以下是针对两个项目的简单修改方案。

## MMTracker_original 修改方案

### 问题
```python
# MMTracker_original/motion/motion_model.py:53
self.corr = SpatialCorrelationSampler(1, self.kernel_size, 1, 0, 1)
```

这里 `kernel_size=1, patch_size=9` 的语义是错误的。

### 解决方案：修正参数顺序

**文件**: `MMTracker_original/motion/motion_model.py`

```python
class Correlation(nn.Module):
    def __init__(self, max_displacement):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2*max_displacement+1
        
        # 修正：使用正确的参数顺序和添加padding
        self.padding = (self.kernel_size - 1) // 2
        
        # 方案1：导入外部包
        from spatial_correlation_sampler import SpatialCorrelationSampler
        self.corr = SpatialCorrelationSampler(
            kernel_size=self.kernel_size,  # 9 (搜索窗口)
            patch_size=1,                  # 1 (单像素查询)
            stride=1,
            padding=self.padding,          # 4 (保持空间尺寸)
            dilation=1
        )
        
        # 或方案2：如果要保持现有导入
        # from spatial_correlation_sampler import SpatialCorrelationSampler
        # self.corr = SpatialCorrelationSampler(self.kernel_size, 1, 1, self.padding, 1)
        
    def forward(self, x, y):
        b, c, h, w = x.shape
        corr_output = self.corr(x, y)
        
        # 处理5D输出 (B, 1, kernel_size^2, H, W) -> (B, kernel_size^2, H, W)
        if len(corr_output.shape) == 5:
            corr_output = corr_output[:, 0, :, :, :]
        
        return corr_output.view(b, -1, h, w) / c
```

## MMTracker 修改方案

MMTracker 已经有适配器处理各种情况，可以直接替换导入：

**文件**: `MMTracker/motion/motion_model.py`

```python
# 原来：
# from yolox.utils import SpatialCorrelationSampler

# 修改为：
from spatial_correlation_sampler import SpatialCorrelationSampler

# Correlation 类保持不变，现有的参数已经是正确的
class Correlation(nn.Module):
    def __init__(self, max_displacement):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2*max_displacement+1
        # 现有参数是正确的，但需要添加padding
        self.padding = (self.kernel_size - 1) // 2
        self.corr = SpatialCorrelationSampler(
            self.kernel_size, 1, 1, self.padding, 1  # 添加padding
        )
    
    def forward(self, x, y):
        b, c, h, w = x.shape
        corr_output = self.corr(x, y)
        
        # 处理5D输出格式
        if len(corr_output.shape) == 5:
            corr_output = corr_output[:, 0, :, :, :]
        
        # 处理可能的批次大小不匹配
        actual_batch_size = corr_output.shape[0]
        if actual_batch_size != b:
            return corr_output.view(actual_batch_size, -1, corr_output.shape[-2], corr_output.shape[-1]) / c
        return corr_output.view(b, -1, h, w) / c
```

## 核心修改点总结

### MMTracker_original
1. **参数顺序修正**: `(1, 9, 1, 0, 1)` → `(9, 1, 1, 4, 1)`
2. **添加padding**: `padding = (kernel_size - 1) // 2`
3. **处理5D输出**: 添加维度压缩逻辑

### MMTracker  
1. **导入替换**: 从内部utils导入改为外部包导入
2. **添加padding**: 现有参数正确，只需添加padding
3. **处理5D输出**: 添加维度处理逻辑

## 验证修改

修改后可以运行以下测试验证：

```python
# 测试脚本
import torch
from spatial_correlation_sampler import SpatialCorrelationSampler

# 测试MMTracker配置
sampler = SpatialCorrelationSampler(kernel_size=9, patch_size=1, stride=1, padding=4, dilation=1)

# 测试数据
x = torch.randn(2, 256, 30, 40)
y = torch.randn(2, 256, 30, 40)

output = sampler(x, y)
print(f"输出形状: {output.shape}")  # 应该是 (2, 1, 81, 30, 40)

# 处理为MMTracker期望的4D格式
if len(output.shape) == 5:
    output = output[:, 0, :, :, :]
print(f"处理后形状: {output.shape}")  # 应该是 (2, 81, 30, 40)
```

## 优势

1. **简单直接**: 直接修改错误的参数，不引入复杂逻辑
2. **语义正确**: 使用正确的搜索窗口和模板大小语义
3. **性能最优**: 避免自动检测和修正的开销
4. **维护性好**: 代码逻辑清晰，易于理解和调试

这样修改后，两个项目都能正确使用 spatial_correlation_sampler，同时保持代码的简洁性。