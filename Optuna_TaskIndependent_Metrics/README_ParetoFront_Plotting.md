# Pareto Front Plotting Module

这个模块提供了用于可视化Pareto前沿和相关优化数据点的功能。它可以读取CSV数据文件并创建全面的图表，显示Pareto前沿和被支配的点。

## 功能特性

- **2D Pareto前沿可视化**: 绘制CQ vs MC的2D散点图，显示Pareto前沿和接近前沿的点
- **3D可视化**: 在CQ-MC基础上添加第三个参数的3D可视化
- **参数分布分析**: 比较Pareto前沿点与所有试验点的参数分布
- **接近前沿点识别**: 自动识别和标记距离Pareto前沿最近的支配点
- **详细统计报告**: 生成关于Pareto前沿分析的汇总统计信息

## 文件结构

```
Optuna_TaskIndependent_Metrics/
├── ParetoFront_Plotting.py      # 主绘图模块
├── example_pareto_plotting.py   # 使用示例脚本
└── README_ParetoFront_Plotting.md  # 本文档
```

## 数据要求

该模块需要从以下位置读取CSV数据文件：
- `../ParetoFront_CQandMC/data/pareto_front.csv` - Pareto前沿点数据
- `../ParetoFront_CQandMC/data/all_trials.csv` - 所有试验数据

CSV文件应包含以下列：
- `number`: 试验编号
- `CQ`: 计算品质 (Computational Quality)
- `MC`: 记忆容量 (Memory Capacity)
- `gamma`, `theta`, `m0`, `h`, `beta_prime`, `Nvirt`: 优化参数

## 快速开始

### 基本使用

```python
from ParetoFront_Plotting import ParetoFrontPlotter

# 初始化绘图器
plotter = ParetoFrontPlotter()

# 加载数据
plotter.load_data()

# 创建2D Pareto前沿图
fig = plotter.plot_pareto_front_2d(
    distance_threshold=0.4,    # 控制显示多少接近前沿的点
    max_near_points=40,        # 最多显示的接近前沿点数量
    save_path="pareto_2d.png"  # 保存路径
)

# 显示图表
import matplotlib.pyplot as plt
plt.show()
```

### 运行示例脚本

```bash
cd Optuna_TaskIndependent_Metrics
python example_pareto_plotting.py
```

## 主要功能详解

### 1. ParetoFrontPlotter类

主要的绘图类，提供以下方法：

#### `load_data()`
从CSV文件加载Pareto前沿和所有试验数据。

#### `plot_pareto_front_2d(figsize, distance_threshold, max_near_points, save_path)`
创建2D Pareto前沿图，包含：
- 所有试验点（灰色背景）
- 接近Pareto前沿的点（橙色）
- Pareto前沿点（红色）
- 前沿点之间的连线

**参数：**
- `figsize`: 图像尺寸，默认 (12, 8)
- `distance_threshold`: 距离阈值，控制哪些点被认为"接近"前沿，默认 0.5
- `max_near_points`: 最大显示的接近前沿点数量，默认 50
- `save_path`: 图像保存路径，可选

#### `plot_parameter_distribution(parameters, figsize, save_path)`
绘制参数分布对比图，比较Pareto前沿点与所有试验点的参数分布。

**参数：**
- `parameters`: 要绘制的参数列表，默认所有数值参数
- `figsize`: 图像尺寸，默认 (15, 10)
- `save_path`: 图像保存路径，可选

#### `plot_3d_pareto(third_param, figsize, distance_threshold, save_path)`
创建3D可视化，显示CQ、MC和第三个参数的关系。

**参数：**
- `third_param`: 第三个维度的参数名，默认 'gamma'
- `figsize`: 图像尺寸，默认 (12, 10)
- `distance_threshold`: 距离阈值，默认 0.5
- `save_path`: 图像保存路径，可选

#### `find_near_pareto_points(distance_threshold, max_points)`
识别接近Pareto前沿的点。

**参数：**
- `distance_threshold`: 最大距离阈值
- `max_points`: 返回的最大点数量

#### `create_summary_report()`
打印详细的Pareto前沿分析统计报告。

## 自定义使用示例

### 调整距离阈值
```python
# 更严格的近似前沿点选择
plotter.plot_pareto_front_2d(distance_threshold=0.2, max_near_points=20)

# 更宽松的近似前沿点选择  
plotter.plot_pareto_front_2d(distance_threshold=0.8, max_near_points=100)
```

### 选择特定参数进行分布分析
```python
# 只分析关键参数
plotter.plot_parameter_distribution(
    parameters=['gamma', 'beta_prime', 'Nvirt']
)
```

### 不同的3D参数组合
```python
# 尝试不同的第三个参数
for param in ['gamma', 'theta', 'beta_prime', 'h']:
    plotter.plot_3d_pareto(
        third_param=param,
        save_path=f"3d_plot_{param}.png"
    )
```

### 直接访问数据进行自定义分析
```python
# 获取数据进行自定义分析
pareto_df = plotter.pareto_front_df
all_trials_df = plotter.all_trials_df

# 找到最佳CQ的试验
best_cq = all_trials_df.loc[all_trials_df['CQ'].idxmax()]
print(f"最佳CQ试验: {best_cq['number']}, CQ = {best_cq['CQ']}")

# 找到接近前沿的点
near_points = plotter.find_near_pareto_points(distance_threshold=0.3)
print(f"找到 {len(near_points)} 个接近前沿的点")
```

## 输出文件

该模块可以生成以下类型的图像文件：
- `pareto_front_2d.png` - 2D Pareto前沿图
- `parameter_distributions.png` - 参数分布对比图
- `pareto_front_3d_[parameter].png` - 3D可视化图

所有图像默认以300 DPI的高质量保存。

## 故障排除

### 常见问题

1. **FileNotFoundError**: 确保CSV数据文件存在于正确的路径
2. **缺少依赖包**: 安装必要的Python包：
   ```bash
   pip install pandas numpy matplotlib seaborn scipy
   ```

3. **图像显示问题**: 如果在Jupyter notebook中使用，添加：
   ```python
   %matplotlib inline
   ```

### 性能注意事项

- 对于大型数据集，可以调整 `max_near_points` 参数来限制显示的点数量
- 3D图表在数据量很大时可能渲染较慢
- 距离计算使用归一化的欧几里得距离，适合大多数用例

## 扩展功能

该模块设计为可扩展的。您可以：

1. 继承 `ParetoFrontPlotter` 类添加新的绘图方法
2. 修改距离计算方法（在 `find_near_pareto_points` 中）
3. 添加新的可视化类型
4. 集成其他多目标优化分析工具

## 联系与支持

如有问题或建议，请联系项目维护者或提交issue。 