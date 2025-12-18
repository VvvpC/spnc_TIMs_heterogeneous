import matplotlib.pyplot as plt
import matplotlib as mpl

# ==========================================
#  1. 物理期刊通用尺寸定义 (单位: 英寸)
#  参考 APS/Nature 标准:
#  Single column width: ~3.375 inch (8.5 cm)
#  Double column width: ~6.75 inch (17 cm)
# ==========================================

# 宽度常量
WIDTH_SINGLE = 3.5  # 单栏宽度 (稍微留点余量)
WIDTH_DOUBLE = 7.0  # 双栏宽度

# 高宽比常量
RATIO_GOLDEN = 1.618         # 黄金比例 (经典美感)
RATIO_SQUARE = 1.0           # 正方形
RATIO_WIDE   = 2.5           # 长条形 (适合时间序列 Trace)

# ==========================================
#  2. 绘图细节参数 (Fine-tuning)
# ==========================================

STYLE_PARAMS = {
    # 字体大小
    'font_size_title': 10,      # 标题
    'font_size_label': 10,      # 轴标签 (x_label, y_label)
    'font_size_tick': 9,        # 刻度标签
    'font_size_legend': 6,      # 图例
    
    # 线条与标记
    'linewidth_grid': 0.5,      # 网格线宽
    'linewidth_plot': 1.5,      # 数据线宽 (标准)
    'linewidth_thick': 2.0,     # 加粗线宽 (强调)
    'markersize': 4,            # 标记点大小
    'markeredgewidth': 0.8,     # 标记边缘线宽
    
    # 颜色 (推荐高对比度配色)
    'color_cycle': [
        '#1f77b4', # Muted Blue
        '#ff7f0e', # Safety Orange
        '#2ca02c', # Cooked Asparagus Green
        '#d62728', # Brick Red
        '#9467bd', # Muted Purple
        '#8c564b', # Chestnut Brown
    ]
}

# ==========================================
#  3. 初始化函数: 应用物理期刊风格
# ==========================================
def set_pub_style(width_mode: str = 'single', scale: float | None = None):
    """
    配置 Matplotlib 的 rcParams 以符合物理期刊发表标准。
    关键：根据图宽自动缩放字体/刻度/线宽，保证 double 与 single 视觉比例一致。
    :param width_mode: 'single' 或 'double'
    :param scale: 可手动指定缩放倍率；None 时按 width_mode 自动计算
    """
    # 重置为默认，避免叠加干扰
    plt.rcParams.update(plt.rcParamsDefault)

    # --- 计算缩放倍率：single=1，double=WIDTH_DOUBLE/WIDTH_SINGLE (=2) ---
    if scale is None:
        width = WIDTH_DOUBLE if width_mode == 'double' else WIDTH_SINGLE
        scale = width / WIDTH_SINGLE
        scale = scale * 0.7

    # --- 字体设置 ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # --- 颜色循环 ---
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=STYLE_PARAMS['color_cycle'])

    # --- 刻度设置 (Ticks) ---
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False

    # 刻度长度/粗细/间距：按 scale 放大
    plt.rcParams['xtick.major.size'] = 4 * scale
    plt.rcParams['xtick.minor.size'] = 2 * scale
    plt.rcParams['ytick.major.size'] = 4 * scale
    plt.rcParams['ytick.minor.size'] = 2 * scale
    plt.rcParams['xtick.major.width'] = 0.8 * scale
    plt.rcParams['xtick.minor.width'] = 0.6 * scale
    plt.rcParams['ytick.major.width'] = 0.8 * scale
    plt.rcParams['ytick.minor.width'] = 0.6 * scale
    plt.rcParams['xtick.major.pad'] = 3.5 * scale
    plt.rcParams['ytick.major.pad'] = 3.5 * scale

    # --- 字体大小：按 scale 放大 ---
    plt.rcParams['font.size'] = STYLE_PARAMS['font_size_label'] * scale
    plt.rcParams['axes.labelsize'] = STYLE_PARAMS['font_size_label'] * scale
    plt.rcParams['axes.titlesize'] = STYLE_PARAMS['font_size_title'] * scale
    plt.rcParams['xtick.labelsize'] = STYLE_PARAMS['font_size_tick'] * scale
    plt.rcParams['ytick.labelsize'] = STYLE_PARAMS['font_size_tick'] * scale
    plt.rcParams['legend.fontsize'] = STYLE_PARAMS['font_size_legend'] * scale

    # --- 线条与标记：按 scale 放大（保证视觉比例一致）---
    plt.rcParams['lines.linewidth'] = STYLE_PARAMS['linewidth_plot'] * scale
    plt.rcParams['lines.markersize'] = STYLE_PARAMS['markersize'] * scale
    plt.rcParams['lines.markeredgewidth'] = STYLE_PARAMS['markeredgewidth'] * scale
    plt.rcParams['grid.linewidth'] = STYLE_PARAMS['linewidth_grid'] * scale

    # --- 边框/布局 ---
    plt.rcParams['axes.linewidth'] = 0.8 * scale
    plt.rcParams['grid.alpha'] = 0
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.05 * scale

def get_figsize(width_mode='single', aspect_ratio=RATIO_GOLDEN):
    """
    计算图片尺寸
    :param width_mode: 'single' (单栏) 或 'double' (双栏)
    :param aspect_ratio: 宽/高 比 (例如 1.618)
    :return: (width, height)
    """
    if width_mode == 'double':
        width = WIDTH_DOUBLE
    else:
        width = WIDTH_SINGLE
        
    height = width / aspect_ratio
    return (width, height)