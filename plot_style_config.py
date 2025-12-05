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
def set_pub_style():
    """
    配置 Matplotlib 的 rcParams 以符合物理期刊发表标准。
    特点: Times New Roman 字体, 向内的刻度, LaTex 数学公式支持
    """
    # 重置为默认，避免叠加干扰
    plt.rcParams.update(plt.rcParamsDefault)
    
    # --- 字体设置 ---
    # 优先使用 Times New Roman (衬线体)，如果没有则回退到 DejaVu Serif
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    # 开启数学公式渲染 (Mathtext)，使其看起来像 LaTeX
    plt.rcParams['mathtext.fontset'] = 'stix' 
    
    # --- 刻度设置 (Ticks) ---
    plt.rcParams['xtick.direction'] = 'in'  # 刻度朝内
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = False        # 上方显示刻度
    plt.rcParams['ytick.right'] = False      # 右侧显示刻度
    plt.rcParams['xtick.major.size'] = 4    # 主刻度长度
    plt.rcParams['xtick.minor.size'] = 2    # 次刻度长度
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['ytick.minor.size'] = 2
    
    # --- 字体大小全局默认 ---
    plt.rcParams['font.size'] = STYLE_PARAMS['font_size_label']
    plt.rcParams['axes.labelsize'] = STYLE_PARAMS['font_size_label']
    plt.rcParams['axes.titlesize'] = STYLE_PARAMS['font_size_title']
    plt.rcParams['xtick.labelsize'] = STYLE_PARAMS['font_size_tick']
    plt.rcParams['ytick.labelsize'] = STYLE_PARAMS['font_size_tick']
    plt.rcParams['legend.fontsize'] = STYLE_PARAMS['font_size_legend']
    
    # --- 布局与边框 ---
    plt.rcParams['axes.linewidth'] = 0.8    # 坐标轴线宽
    plt.rcParams['grid.alpha'] = 0        # 网格透明度
    plt.rcParams['figure.autolayout'] = False # 我们手动控制 layout
    plt.rcParams['savefig.bbox'] = 'tight'  # 保存时去除多余白边
    plt.rcParams['savefig.pad_inches'] = 0.05

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