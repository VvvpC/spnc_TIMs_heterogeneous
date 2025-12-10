# 这个文档中函数作用是辅助绘图

import matplotlib.pyplot as plt
import os
import sys

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



