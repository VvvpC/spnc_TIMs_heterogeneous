# SPNC Task-Independent Metrics - Modular Framework

## 概述

这是一个重新设计的模块化框架，用于评估超顺磁纳米点储层计算系统(SPNC)的任务无关性能指标。该框架将原始的单文件代码重构为高内聚、低耦合的模块化架构。

## 目录结构

```
├── core/                          # 核心组件
│   ├── __init__.py               
│   ├── reservoir.py              # ReservoirParams, RunSpnc
│   └── base_utils.py             # 通用工具函数
├── tasks/                         # 评估任务模块
│   ├── __init__.py
│   ├── memory_capacity/          # 记忆容量评估
│   │   ├── __init__.py
│   │   ├── signals.py            # 信号生成和处理
│   │   ├── processing.py         # MC专用处理函数
│   │   └── evaluator.py          # 主要评估函数
│   ├── kr_gr/                    # KR&GR评估
│   │   ├── __init__.py
│   │   ├── signals.py            # KR&GR信号生成
│   │   ├── processing.py         # 排序计算和分析
│   │   └── evaluator.py          # 主要评估函数
│   ├── narma10/                  # NARMA10评估
│   │   ├── __init__.py
│   │   └── evaluator.py          # NARMA10评估
│   └── ti46/                     # TI46语音识别
│       ├── __init__.py
│       └── evaluator.py          # TI46评估
├── framework/                     # 评估框架
│   ├── __init__.py
│   ├── evaluator.py              # ReservoirPerformanceEvaluator
│   └── runner.py                 # run_evaluation统一接口
├── config/                        # 配置管理
│   ├── __init__.py
│   └── paths.py                  # 环境和路径设置
├── formal_Parameter_Dynamics_Preformance.py  # 原始文件(保留)
├── example_usage.py              # 使用示例
├── test_basic.py                 # 基础测试
└── README_MODULAR.md             # 本文档
```

## 主要特性

### 1. 完全模块化
- 每个评估任务是完整的自包含模块
- 任务专用的信号生成和处理函数
- 清晰的模块间依赖关系

### 2. 高内聚性
- MC模块：包含信号生成、线性记忆容量计算、Ridge回归等全部相关功能
- KR&GR模块：包含特殊输入生成、SVD分析、排序计算等专用功能
- 每个模块都是完整的"业务单元"

### 3. 故障隔离
- 单个任务的修改不会影响其他任务
- 完善的错误处理机制
- 参数验证和默认值处理

### 4. 灵活的参数扫描
- 支持单参数范围扫描
- 支持多参数网格搜索
- 自动化结果保存和管理

## 使用方法

### 基本使用

```python
# 设置环境
from config import setup_environment
setup_environment()

# 导入核心组件
from core import ReservoirParams
from framework import run_evaluation

# 创建储层参数
reservoir_params = ReservoirParams(
    beta_prime=30,
    Nvirt=40,
    m0=0.008,
    params={'theta': 0.3, 'gamma': 0.12}
)

# 运行单个评估
result = run_evaluation(
    task_type='MC',
    param_name='beta_prime',
    param_range=[20, 30, 40, 50],
    reservoir_params=reservoir_params,
    reservoir_tag='my_mc_evaluation'
)

print(f"Memory Capacity results: {result['MC']}")
```

### 多参数网格搜索

```python
import numpy as np

# 定义参数网格
param_grid = {
    'm0': np.linspace(0.005, 0.012, 5),
    'gamma': np.linspace(0.08, 0.15, 4)
}

# 运行网格搜索
result = run_evaluation(
    task_type='KRANDGR',
    param_grid=param_grid,
    reservoir_params=reservoir_params,
    reservoir_tag='kr_gr_grid_search'
)

print(f"Best CQ: {max(result['CQ'])}")
```

### 运行所有任务

```python
task_types = ['MC', 'KRANDGR', 'NARMA10', 'TI46']

for task in task_types:
    result = run_evaluation(
        task_type=task,
        param_name='beta_prime',
        param_range=[30],  # 单点评估
        reservoir_params=reservoir_params,
        reservoir_tag=f'all_tasks_{task.lower()}'
    )
    # 处理结果...
```

## 可用的评估任务

1. **MC** - 线性记忆容量评估
   - 结果键: `['MC']`
   - 描述: 使用延迟信号重构评估线性记忆能力

2. **KRANDGR** - 计算质量评估 
   - 结果键: `['KR', 'GR', 'CQ']`
   - 描述: 通过核排序和泛化排序分析计算质量

3. **NARMA10** - 非线性系统识别
   - 结果键: `['NRMSE', 'y_test', 'pred']`
   - 描述: NARMA10非线性基准测试

4. **TI46** - 语音识别
   - 结果键: `['acc']`
   - 描述: TI46数字语音识别任务

## 与原始框架的对比

### 原始框架的问题
- 单文件包含所有功能（692行）
- 函数间紧耦合
- 修改任一功能可能影响整体稳定性
- 难以扩展和维护

### 新模块化框架的优势
- 功能模块完全隔离
- 每个任务是自包含的业务单元
- 修改单个任务不影响其他功能
- 易于添加新的评估任务
- 更好的代码组织和可读性

## 兼容性说明

- 原始的 `formal_Parameter_Dynamics_Preformance.py` 文件已保留
- 新框架提供相同的功能接口
- 结果格式与原框架完全兼容
- 支持所有原有的参数配置

## 环境依赖

模块化框架依赖以下外部库：
- `spnc` - SPNC物理模型
- `single_node_res` - 单节点储层
- `deterministic_mask` - 确定性掩码
- `spnc_ml` - SPNC机器学习功能
- 其他原框架依赖的库

## 测试

运行基础结构测试：
```bash
python test_basic.py
```

运行使用示例：
```bash
python example_usage.py
```

## 扩展指南

### 添加新评估任务

1. 在 `tasks/` 下创建新目录
2. 实现以下文件：
   - `__init__.py` - 模块初始化
   - `signals.py` - 任务专用信号处理
   - `processing.py` - 任务专用算法
   - `evaluator.py` - 主评估函数

3. 在 `framework/runner.py` 的 `TASK_REGISTRY` 中注册

### 修改现有任务

每个任务模块完全独立，可以安全地修改任一模块而不影响其他功能。

## 小结

这个模块化设计实现了你提出的目标：
- ✅ 避免细微修改导致框架崩溃
- ✅ 将信号生成和处理函数与任务放在一起，形成完整体系
- ✅ 保持原有功能和接口的完整性
- ✅ 提供更好的可扩展性和维护性