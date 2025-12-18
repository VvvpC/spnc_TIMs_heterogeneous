# 这个文档中函数作用是辅助绘图

import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np

# 获取当前文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from Plotting.plot_style_config import set_pub_style, get_figsize

def plot_scatter(x, y,x_label, y_label):
    set_pub_style()
    plt.figure(figsize=get_figsize('single'))

    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_bar(filename, feature_list):
    # 使用set_pub_style()
    set_pub_style()
    df = pd.read_csv(filename)
    df["reservoir"] = [f"R{i+1}" for i in range(len(df))]

    features = feature_list
    z = df[features]
    z["reservoir"] = df["reservoir"]

    # 宽表：index=reservoir，columns=feature，values=zscore
    wide = z.set_index("reservoir")[features]

    set_pub_style()
    fig, ax = plt.subplots(figsize=get_figsize("double"))

    x = np.arange(len(wide.index))
    m = len(features)
    width = min(0.8 / max(m, 1), 0.25)  # 每组总宽<=0.8，每根柱<=0.25

    for i, feat in enumerate(features):
        offset = (i - (m - 1) / 2) * width
        ax.bar(x + offset, wide[feat].to_numpy(), width=width, label=feat)

    ax.set_xticks(x)
    ax.set_xticklabels(wide.index, rotation=45, ha="right")
    ax.set_xlabel("Reservoir")
    ax.set_ylabel("Feature Value")
    ax.legend(title="feature", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    fig.tight_layout()
    # 保存图片
    fig.savefig('bar_plot.png', dpi=300, bbox_inches='tight')
    plt.show()