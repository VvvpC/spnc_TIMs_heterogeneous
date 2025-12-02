import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def parse_cq_data(cq_string):
    """
    解析CQ数据格式，例如：143(161,18) -> CQ=143, CQ_KR=161, CQ_GR=18
    
    Args:
        cq_string (str): CQ数据字符串，格式如"143(161,18)"
    
    Returns:
        tuple: (CQ值, CQ_KR值, CQ_GR值)
    """
    if pd.isna(cq_string) or cq_string == '':
        return np.nan, np.nan, np.nan
    
    # 使用正则表达式解析格式：数字(数字,数字)
    pattern = r'(\d+)\((\d+),(\d+)\)'
    match = re.match(pattern, str(cq_string))
    
    if match:
        cq_val = int(match.group(1))
        cq_kr = int(match.group(2))
        cq_gr = int(match.group(3))
        return cq_val, cq_kr, cq_gr
    else:
        return np.nan, np.nan, np.nan


def load_and_process_pareto_data(csv_file='Pareto_information.csv'):
    """
    加载并处理Pareto前沿数据，解析CQ1和CQ2的复合数据格式
    
    Args:
        csv_file (str): CSV文件路径
        
    Returns:
        pd.DataFrame: 处理后的数据，包含分离的CQ1, CQ1_KR, CQ1_GR, CQ2, CQ2_KR, CQ2_GR列
    """
    # 加载Pareto信息
    pareto_data = pd.read_csv(csv_file)
    
    # 解析CQ1数据
    cq1_parsed = pareto_data['CQ1'].apply(parse_cq_data)
    pareto_data['CQ1'] = [x[0] if isinstance(x, tuple) else np.nan for x in cq1_parsed]
    pareto_data['CQ1_KR'] = [x[1] if isinstance(x, tuple) else np.nan for x in cq1_parsed]
    pareto_data['CQ1_GR'] = [x[2] if isinstance(x, tuple) else np.nan for x in cq1_parsed]
    
    # 解析CQ2数据
    cq2_parsed = pareto_data['CQ2'].apply(parse_cq_data)
    pareto_data['CQ2'] = [x[0] if isinstance(x, tuple) else np.nan for x in cq2_parsed]
    pareto_data['CQ2_KR'] = [x[1] if isinstance(x, tuple) else np.nan for x in cq2_parsed]
    pareto_data['CQ2_GR'] = [x[2] if isinstance(x, tuple) else np.nan for x in cq2_parsed]
    
    return pareto_data

def get_parameter_metrics(data):
    """
    提取参数指标
    """
    metrics = {
        'trial': data['trial'].values,
        'gamma': data['gamma'].values,
        'theta': data['theta'].values,
        'm0': data['m0'].values,
        'beta_prime': data['beta_prime'].values,
    }
    return metrics
        
        


def get_task_performance_metrics(data):
    """
    提取任务表现指标
    
    Args:
        data (pd.DataFrame): 处理后的数据
        
    Returns:
        dict: 任务表现指标字典
    """
    metrics = {
        'NARMA10': data['NARMA10'].values,
        'TI46': data['TI46'].values,
    }
    return metrics


def get_computational_metrics(data):
    """
    提取计算性能指标
    
    Args:
        data (pd.DataFrame): 处理后的数据
        
    Returns:
        dict: 计算性能指标字典
    """
    metrics = {
        'MC': data['MC'].values,
        'CQ1': data['CQ1'].values,
        'CQ1_KR': data['CQ1_KR'].values,
        'CQ1_GR': data['CQ1_GR'].values,
        'CQ2': data['CQ2'].values,
        'CQ2_KR': data['CQ2_KR'].values,
        'CQ2_GR': data['CQ2_GR'].values,
    }
    return metrics


def calculate_linear_correlation(x, y, x_label=None, y_label=None, plot=False, save_path=None):
    """
    计算两个变量之间的皮尔逊线性相关性，可选择绘制散点图
    
    Args:
        x (array-like): 第一个变量的数据
        y (array-like): 第二个变量的数据
        x_label (str, optional): x轴标签
        y_label (str, optional): y轴标签
        plot (bool): 是否绘制散点图
        save_path (str, optional): 图片保存路径
        
    Returns:
        dict: 包含相关系数和p值的字典
    """
    # 移除缺失值
    x = np.array(x)
    y = np.array(y)
    
    # 找到两个数组都不是NaN的索引
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if len(x_clean) < 2:
        return {'correlation': np.nan, 'p_value': np.nan, 'n_samples': len(x_clean)}
    
    # 计算皮尔逊相关系数
    correlation, p_value = pearsonr(x_clean, y_clean)
    
    # 绘制散点图（如果需要）
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(x_clean, y_clean, alpha=1, color='green', s=200, marker='*')
        
        # 添加回归线
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        plt.plot(x_clean, p(x_clean), "--", alpha=0.8, linewidth=3)
        
        # 设置标签和标题
        plt.xlabel(x_label if x_label else 'X Variable', fontsize=12)
        plt.ylabel(y_label if y_label else 'Y Variable', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(f'Correlation: r={correlation:.3f}, p={p_value:.3f}')
        plt.grid(True, alpha=0.3)
        
        # 将相关性统计信息放到图外（图下方）
        plt.gcf().text(0.5, -0.08, f'r = {correlation:.3f}, p = {p_value:.3f}', 
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=12)
        plt.tight_layout()
        
        # 保存图片（如果指定路径）
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'n_samples': len(x_clean)
    }


def calculate_statistics(data):
    """
    计算数据的平均值和标准差，可选择绘制直方图
    
    Args:
        data (array-like): 数据数组
        label (str, optional): 数据标签，用于图表标题和轴标签
        plot (bool): 是否绘制直方图
        save_path (str, optional): 图片保存路径
        bins (int): 直方图的箱数
        
    Returns:
        dict: 包含平均值、标准差、样本数等统计信息的字典
    """
    # 移除缺失值
    data = np.array(data)
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'n_samples': 0
        }
    
    # 计算统计量
    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean, ddof=1)  # 使用样本标准差
    median_val = np.median(data_clean)
    min_val = np.min(data_clean)
    max_val = np.max(data_clean)
    
    return {
        'mean': mean_val,
        'std': std_val,
        'median': median_val,
        'min': min_val,
        'max': max_val,
        'n_samples': len(data_clean)
    }

def histogram(datalist, labels=None, bins=10, alpha=0.5, title=None, xlabel=None, ylabel='Count'):
    """
    绘制多组数据的对比直方图

    Args:
        datalist (list or array-like): 多组数据，每组为一list或np.array。形如 [data1, data2, ...]
        labels (list of str, optional): 每组数据的标签
        bins (int): 直方图的箱数
        alpha (float): 透明度
        title (str, optional): 图表标题
        xlabel (str, optional): x轴标签
        ylabel (str, optional): y轴标签
    """
    if not isinstance(datalist, (list, tuple)):
        datalist = [datalist]
    if labels is None:
        labels = [f'data{i+1}' for i in range(len(datalist))]
    plt.figure(figsize=(8, 6))
    for data, label in zip(datalist, labels):
        data = np.array(data)
        data = data[~np.isnan(data)]
        plt.hist(data, bins=bins, alpha=alpha, label=label)
    plt.legend()
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return

def plot_boxplot(metric, labels, xlabel='Group'):
    """
    绘制MC的分组箱线图

    Args:
        mc_data (list of array-like): 每组MC数据的列表
        labels (list of str): 每组的标签
    """
    data = [up_data[metric].values, mid_data[metric].values, bot_data[metric].values]
    plt.figure(figsize=(6, 5))
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.xlabel(xlabel)
    plt.ylabel(metric)
    plt.tight_layout()
    # save
    plt.savefig(f'Correlation_Plotting/{xlabel}_{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_ti46_cq_contribution(data, plot=True, save_path=None):
    """
    使用标准化多元线性回归分析TI46与CQ指标的关系，评估KR和GR的贡献率
    
    Args:
        data (pd.DataFrame): 包含TI46和CQ指标的数据
        plot (bool): 是否绘制结果可视化
        save_path (str, optional): 图片保存路径
        
    Returns:
        dict: 包含回归结果和贡献率分析的字典
    """
    # 提取变量
    y = data['TI46'].values  # 因变量
    X_vars = ['CQ1_KR', 'CQ1_GR']  # 只分析CQ1的KR和GR
    X = data[X_vars].values
    
    # 移除包含NaN的行
    valid_mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    y_clean = y[valid_mask]
    X_clean = X[valid_mask]
    
    if len(y_clean) < len(X_vars) + 1:
        return {'error': 'Insufficient data for regression analysis'}
    
    # 标准化数据
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_standardized = scaler_X.fit_transform(X_clean)
    y_standardized = scaler_y.fit_transform(y_clean.reshape(-1, 1)).ravel()
    
    # 执行多元线性回归
    model = LinearRegression(fit_intercept=True)
    model.fit(X_standardized, y_standardized)
    
    # 预测值
    y_pred_std = model.predict(X_standardized)
    y_pred = scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).ravel()
    
    # 计算R²
    r2 = r2_score(y_clean, y_pred)
    r2_std = r2_score(y_standardized, y_pred_std)
    
    # 标准化回归系数（beta coefficients）
    beta_coefficients = model.coef_
    
    # 计算各变量的相对重要性（绝对贡献率）
    abs_beta = np.abs(beta_coefficients)
    relative_importance = abs_beta / np.sum(abs_beta) * 100
    
    # 按类型分组分析（KR vs GR）
    kr_indices = [0]  # CQ1_KR
    gr_indices = [1]  # CQ1_GR
    
    kr_contribution = np.sum(abs_beta[kr_indices]) / np.sum(abs_beta) * 100
    gr_contribution = np.sum(abs_beta[gr_indices]) / np.sum(abs_beta) * 100
    
    # 结果字典
    results = {
        'r_squared': r2,
        'r_squared_standardized': r2_std,
        'beta_coefficients': dict(zip(X_vars, beta_coefficients)),
        'relative_importance': dict(zip(X_vars, relative_importance)),
        'kr_total_contribution': kr_contribution,
        'gr_total_contribution': gr_contribution,
        'n_samples': len(y_clean)
    }
    
    # 可视化结果
    if plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Beta系数条形图
        ax1.bar(X_vars, beta_coefficients, alpha=0.7, 
               color=['red' if 'KR' in var else 'blue' for var in X_vars])
        ax1.set_title('Standardized Beta Coefficients')
        ax1.set_ylabel('Beta Coefficient')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 2. 相对重要性饼图
        ax2.pie(relative_importance, labels=X_vars, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Relative Importance of Variables')
        
        # 3. CQ1_KR vs CQ1_GR贡献率对比
        contributions = [kr_contribution, gr_contribution]
        labels = ['CQ1_KR', 'CQ1_GR']
        colors = ['red', 'blue']
        ax3.bar(labels, contributions, color=colors, alpha=0.7)
        ax3.set_title('CQ1_KR vs CQ1_GR Contribution')
        ax3.set_ylabel('Contribution Rate (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. 实际值 vs 预测值散点图
        ax4.scatter(y_clean, y_pred, alpha=0.7, color='green', s=50)
        ax4.plot([y_clean.min(), y_clean.max()], [y_clean.min(), y_clean.max()], 
                'r--', alpha=0.8, linewidth=2)
        ax4.set_xlabel('Actual TI46')
        ax4.set_ylabel('Predicted TI46')
        ax4.set_title(f'Actual vs Predicted (R² = {r2:.3f})')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # 打印详细结果
        print("=== TI46 与 CQ指标多元回归分析结果 ===")
        print(f"R² = {r2:.3f}")
        print(f"样本数量: {len(y_clean)}")
        print("\n标准化Beta系数:")
        for var, beta in results['beta_coefficients'].items():
            print(f"  {var}: {beta:.3f}")
        print("\n相对重要性:")
        for var, importance in results['relative_importance'].items():
            print(f"  {var}: {importance:.1f}%")
        print(f"\nCQ1_KR贡献率: {kr_contribution:.1f}%")
        print(f"CQ1_GR贡献率: {gr_contribution:.1f}%")
        
        if kr_contribution > gr_contribution:
            print(f"\n结论: TI46的提升主要由CQ1_KR增加驱动 ({kr_contribution:.1f}% vs {gr_contribution:.1f}%)")
        else:
            print(f"\n结论: TI46的提升主要由CQ1_GR降低驱动 ({gr_contribution:.1f}% vs {kr_contribution:.1f}%)")
    
    return results


def Pareto_color(csv_file='Pareto_information.csv', cq_type='CQ1', save_path=None, 
                title="Pareto Front with Position Colors", figsize=(15, 5)):
    """
    从CSV文件中提取帕累托前沿点数据并绘制散点图，使用颜色表示点在前沿上的相对位置
    包含三个子图：1) CQ vs MC散点图 2) NARMA-10性能散点图 3) TI46性能散点图，颜色保持一致
    
    Args:
        csv_file (str): CSV文件路径，默认为'Pareto_information.csv'
        cq_type (str): 使用的CQ类型，'CQ1'或'CQ2'，默认为'CQ1'
        save_path (str, optional): 图片保存路径
        title (str): 图表标题
        figsize (tuple): 图片尺寸
        
    Returns:
        None: 显示并保存图片
    """
    # 加载并处理帕累托数据
    try:
        data = load_and_process_pareto_data(csv_file)
        print(f"Successfully loaded data from {csv_file}")
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 提取CQ、MC、NARMA10和TI46数据
    if cq_type not in ['CQ1', 'CQ2']:
        print("Error: cq_type must be 'CQ1' or 'CQ2'")
        return
        
    CQ = data[cq_type].values
    MC = data['MC'].values
    NARMA10 = data['NARMA10'].values
    TI46 = data['TI46'].values
    
    # 移除NaN值（需要所有变量都有效）
    valid_mask = ~(np.isnan(CQ) | np.isnan(MC) | np.isnan(NARMA10) | np.isnan(TI46))
    CQ_clean = CQ[valid_mask]
    MC_clean = MC[valid_mask]
    NARMA10_clean = NARMA10[valid_mask]
    TI46_clean = TI46[valid_mask]
    
    if len(CQ_clean) < 2:
        print("Error: Insufficient valid data points for plotting")
        return
    
    print(f"Valid data points: {len(CQ_clean)}")
    print(f"{cq_type} range: [{CQ_clean.min():.1f}, {CQ_clean.max():.1f}]")
    print(f"MC range: [{MC_clean.min():.3f}, {MC_clean.max():.3f}]")
    print(f"NARMA10 range: [{NARMA10_clean.min():.3f}, {NARMA10_clean.max():.3f}]")
    print(f"TI46 range: [{TI46_clean.min():.3f}, {TI46_clean.max():.3f}]")
    
    # 创建内部编号系统：按照CQ越小、MC越大的优先级排序
    # 首先按CQ升序排序，然后按MC降序排序
    # 创建数据框便于排序
    df_temp = pd.DataFrame({
        'CQ': CQ_clean,
        'MC': MC_clean,
        'NARMA10': NARMA10_clean,
        'TI46': TI46_clean,
        'original_index': np.arange(len(CQ_clean))
    })
    
    # 排序：CQ升序，MC降序
    df_sorted = df_temp.sort_values(['CQ', 'MC'], ascending=[True, False]).reset_index(drop=True)
    
    # 创建内部编号（从1开始）
    internal_numbers = np.arange(1, len(df_sorted) + 1)
    df_sorted['internal_number'] = internal_numbers
    
    # 获取排序后的数据
    CQ_sorted = df_sorted['CQ'].values
    MC_sorted = df_sorted['MC'].values
    NARMA10_sorted = df_sorted['NARMA10'].values
    TI46_sorted = df_sorted['TI46'].values
    TI46_error_sorted = (1 - TI46_sorted) * 100  # 误差率，百分比
    internal_nums = df_sorted['internal_number'].values
    
    # 计算每个点在前沿上的相对位置（用于颜色映射）
    # 对于帕累托前沿，理想情况是CQ小、MC大，所以我们创建一个综合指标
    # 归一化CQ和MC
    CQ_norm = (CQ_sorted - CQ_sorted.min()) / (CQ_sorted.max() - CQ_sorted.min()) if CQ_sorted.max() != CQ_sorted.min() else np.zeros_like(CQ_sorted)
    MC_norm = (MC_sorted - MC_sorted.min()) / (MC_sorted.max() - MC_sorted.min()) if MC_sorted.max() != MC_sorted.min() else np.zeros_like(MC_sorted)
    
    # 计算相对位置：CQ越小越好，MC越大越好
    # 所以好的点应该有低CQ_norm和高MC_norm
    position_metric = 0.5 * (1 - CQ_norm) + 0.5 * MC_norm
    
    # 创建1x3子图布局
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ax1, ax2, ax3 = axes

    # 调整子图间距
    plt.subplots_adjust(left=0.06, right=0.92, top=0.88, bottom=0.15, wspace=0.28)

    # 子图1: CQ vs MC散点图（使用排序后的数据）
    scatter1 = ax1.scatter(CQ_sorted, MC_sorted, c=position_metric, 
                          cmap='viridis', s=120, alpha=0.8)
    ax1.set_xlabel('CQ (Computational Quality)', fontsize=14)
    ax1.set_ylabel('MC (Memory Capacity)', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # 子图2: NARMA-10性能散点图（使用内部编号作为横坐标）
    scatter2 = ax2.scatter(internal_nums, NARMA10_sorted, c=position_metric,
                          cmap='viridis', s=120, alpha=0.8)
    ax2.set_ylabel('NRMSE(NARMA-10)', fontsize=14)
    ax2.tick_params(axis='y', which='major', labelsize=12)
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # 子图3: TI46性能散点图（使用内部编号作为横坐标）
    scatter3 = ax3.scatter(internal_nums, TI46_error_sorted, c=position_metric,
                          cmap='viridis', s=120, alpha=0.8)

    ax3.set_ylabel('Error Rate(%, TI46)', fontsize=14)
    ax3.tick_params(axis='y', which='major', labelsize=12)
    ax3.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # 在右侧添加颜色条
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax1)
    cbar_ax = divider.append_axes("right", size="10%", pad=0.15)
    cbar = fig.colorbar(scatter3, cax=cbar_ax)
    cbar.set_label('Relative Position ', fontsize=14, rotation=90)
    cbar.set_ticks([])  # 不显示刻度数字，只显示颜色

    # 设置总标题


    # 保存图片（如果指定路径）
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # 显示图片
    plt.show()
    
    return


if __name__ == "__main__":
    # # Test data processing functionality
    # data = load_and_process_pareto_data()


    # # 分离参数,性能,和任务表现
    # params = get_parameter_metrics(data)
    # tasks = get_task_performance_metrics(data)
    # metrics = get_computational_metrics(data)

    # up = [91,29,320,254]
    # mid = [193,326,250,103,280,217,116,288,290,283]
    # bot = [301,277,94,157,121]
    
    
    # # 筛选特定trial的数据
    # up_data = data[data['trial'].isin(up)]
    # mid_data = data[data['trial'].isin(mid)]
    # bot_data = data[data['trial'].isin(bot)]

    # labels = ['UP', 'MID', 'BOT']


    # # 调用函数
    # # plot_boxplot('CQ2_GR', labels)
    
    # # 相关性
    # MC = mid_data['MC'].values
    # NARMA10 = mid_data['NARMA10'].values
    # TI46 = mid_data['TI46'].values
    # CQ1 = mid_data['CQ1'].values
    # CQ2 = mid_data['CQ2'].values
    # CQ1_KR = mid_data['CQ1_KR'].values
    # CQ2_KR = mid_data['CQ2_KR'].values
    # CQ1_GR = mid_data['CQ1_GR'].values
    # CQ2_GR = mid_data['CQ2_GR'].values

    # # 计算相关性
    # correlation = calculate_linear_correlation(MC, CQ1_GR, x_label='MC', y_label='CQ1_GR', plot=True, save_path='Correlation_Plotting/Correlation_MC_CQ1_GR.png')
    # print(correlation)
    
    # # 测试TI46与CQ指标的多元回归分析
    # print("\n=== 测试标准化多元线性回归分析 ===")
    # regression_results = analyze_ti46_cq_contribution(mid_data, plot=True, save_path='Correlation_Plotting/TI46_CQ_Regression_Analysis.png')
    # print(f"回归分析完成，R² = {regression_results['r_squared']:.3f}")

    Pareto_color(save_path='Correlation_Plotting/Pareto_Front_with_Position_Colors.png')