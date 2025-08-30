import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def create_boxplot_comparison(data, grade_groups, save_path='../res/boxplot_comparison.png'):
    """
    创建更美观高级的箱线图比较，采用seaborn风格，增加分布、均值、样式美化等
    """
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False

    grade_names = ['高一', '高二', '高三']
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    # 准备数据
    word_data = [grade_groups[grade]['单词数量-DESWC-03'] for grade in grade_names]
    sent_data = [grade_groups[grade]['句子数量-DESSC-02'] for grade in grade_names]

    # 构造DataFrame用于seaborn
    import pandas as pd
    df_word = pd.DataFrame({
        '年级': np.repeat(grade_names, [len(x) for x in word_data]),
        '单词数量': np.concatenate(word_data)
    })
    df_sent = pd.DataFrame({
        '年级': np.repeat(grade_names, [len(x) for x in sent_data]),
        '句子数量': np.concatenate(sent_data)
    })

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=120, constrained_layout=True)

    # 单词数量箱线图+小提琴图
    sns.violinplot(
        x='年级', y='单词数量', data=df_word, ax=axes[0],
        inner=None, palette=colors, linewidth=0, alpha=0.18
    )
    sns.boxplot(
        x='年级', y='单词数量', data=df_word, ax=axes[0],
        width=0.25, palette=colors, boxprops=dict(alpha=0.7)
    )
    # 均值点
    means = df_word.groupby('年级')['单词数量'].mean()
    axes[0].scatter(range(len(grade_names)), means, color='#E74C3C', s=120, marker='D', edgecolor='white', zorder=10, label='均值')
    # 美化
    axes[0].set_title('单词数量年级间分布对比', fontsize=20, fontweight='bold', color='#222')
    axes[0].set_xlabel('年级', fontsize=16, fontweight='bold')
    axes[0].set_ylabel('单词数量', fontsize=16, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.25)
    axes[0].set_axisbelow(True)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # 句子数量箱线图+小提琴图
    sns.violinplot(
        x='年级', y='句子数量', data=df_sent, ax=axes[1],
        inner=None, palette=colors, linewidth=0, alpha=0.18
    )
    sns.boxplot(
        x='年级', y='句子数量', data=df_sent, ax=axes[1],
        width=0.25, palette=colors, boxprops=dict(alpha=0.7)
    )
    # 均值点
    means = df_sent.groupby('年级')['句子数量'].mean()
    axes[1].scatter(range(len(grade_names)), means, color='#E74C3C', s=120, marker='D', edgecolor='white', zorder=10, label='均值')
    # 美化
    axes[1].set_title('句子数量年级间分布对比', fontsize=20, fontweight='bold', color='#222')
    axes[1].set_xlabel('年级', fontsize=16, fontweight='bold')
    axes[1].set_ylabel('句子数量', fontsize=16, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.25)
    axes[1].set_axisbelow(True)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # 自定义图例
    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='高一', alpha=0.7),
        Patch(facecolor='#A23B72', edgecolor='#A23B72', label='高二', alpha=0.7),
        Patch(facecolor='#F18F01', edgecolor='#F18F01', label='高三', alpha=0.7),
        Line2D([0], [0], marker='D', color='w', label='均值', markerfacecolor='#E74C3C', markeredgecolor='white', markersize=12)
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    axes[1].legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)

    # 去除多余边距
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"箱线图已保存为: {save_path}")
    
def create_density_distribution(data, grade_groups, save_path='../res/density_distribution.png'):
    """创建专业美观的密度分布图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 单词数量密度图
    for i, grade in enumerate(grade_names):
        grade_data = grade_groups[grade]['单词数量-DESWC-03']
        ax1.hist(grade_data, bins=20, alpha=0.6, density=True, 
                color=colors[i], edgecolor='white', linewidth=1.5, label=grade)
        
        # 添加核密度估计曲线
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(grade_data)
        x_range = np.linspace(grade_data.min(), grade_data.max(), 100)
        ax1.plot(x_range, kde(x_range), color=colors[i], linewidth=3, alpha=0.8)
    
    ax1.set_title('单词数量分布密度图', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax1.set_xlabel('单词数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_ylabel('密度', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 句子数量密度图
    for i, grade in enumerate(grade_names):
        grade_data = grade_groups[grade]['句子数量-DESSC-02']
        ax2.hist(grade_data, bins=15, alpha=0.6, density=True,
                color=colors[i], edgecolor='white', linewidth=1.5, label=grade)
        
        # 添加核密度估计曲线
        kde = gaussian_kde(grade_data)
        x_range = np.linspace(grade_data.min(), grade_data.max(), 100)
        ax2.plot(x_range, kde(x_range), color=colors[i], linewidth=3, alpha=0.8)
    
    ax2.set_title('句子数量分布密度图', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax2.set_xlabel('句子数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_ylabel('密度', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"密度分布图已保存为: {save_path}")



def create_density_distribution(data, grade_groups, save_path='../res/微观叙事词法变异系数热力图.png'):
    """创建微观叙事词法变异系数热力图"""
    print("生成微观叙事词法变异系数热力图...")
    
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False
    
    # 定义指标
    indicators = {
        '多样性': '多样性-LDVOCDa（51）',
        '密度': '密度-WRDFRQC（94）',
        '复杂度-K1': '复杂度-K1',
        '复杂度-K2': '复杂度-K2', 
        '复杂度-K3': '复杂度-K3',
        '复杂度-K4': '复杂度-K4',
        '复杂度-K5': '复杂度-K5',
        '复杂度-K6+': '复杂度-K6+'
    }
    
    grade_names = ['高一', '高二', '高三']
    
    # 计算各年级各指标的变异系数
    cv_matrix = []
    for grade in grade_names:
        grade_data = grade_groups[grade]
        cv_row = []
        for name, col in indicators.items():
            cv = grade_data[col].std() / grade_data[col].mean() * 100
            cv_row.append(cv)
        cv_matrix.append(cv_row)
    
    cv_matrix = np.array(cv_matrix)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=120, constrained_layout=True)
    
    # 创建热力图
    im = sns.heatmap(
        cv_matrix, 
        annot=True, 
        cmap='RdYlBu_r', 
        center=cv_matrix.mean(),
        square=True, 
        linewidths=2, 
        cbar_kws={"shrink": .8, "label": "变异系数 (%)"},
        ax=ax, 
        fmt='.1f', 
        annot_kws={'size': 14, 'weight': 'bold'},
        xticklabels=list(indicators.keys()),
        yticklabels=grade_names
    )
    
    # 设置标题和标签
    ax.set_title('各年级各指标变异系数热力图', fontsize=20, fontweight='bold', color='#222', pad=20)
    ax.set_xlabel('指标', fontsize=16, fontweight='bold', color='#222')
    ax.set_ylabel('年级', fontsize=16, fontweight='bold', color='#222')
    
    # 美化坐标轴
    ax.tick_params(axis='both', which='major', labelsize=12, labelcolor='#222')
    
    # 旋转x轴标签以提高可读性
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"变异系数热力图已保存为: {save_path}")