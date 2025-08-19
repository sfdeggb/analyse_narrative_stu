from sklearn.decomposition import PCA
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

# 创建复杂度主成分分析
def create_complexity_pca(df,cols_name,n_components=1,target_col_name='复杂度-K'):
    k_cols = cols_name
    pca = PCA(n_components=n_components)
    df[target_col_name] = pca.fit_transform(df[k_cols])
    # 删除原来的复杂度列
    df = df.drop(columns=k_cols)
    return df

#算数平均求综合指标
def create_complexity_composite_mean(df,cols_name,target_col_name='综合复杂度'):
    k_cols = cols_name
    df[target_col_name] = df[k_cols].mean(axis=1)
    #删除原来的列
    df = df.drop(columns=k_cols)
    return df


def plot_boxplots_by_grade2(file_name, df, n):
    """
    输入文件名、对应的DataFrame和要可视化的特征数n，输出该DataFrame的箱线图（plotly），
    一个特征一个子图，最后是一个大图。
    箱线图横坐标为年级，纵坐标为特征值，子图标题不要了，
    大图的名称为文件名。
    每行最多放三个子图，多余的自动换行。
    """
    # 识别数值列（排除年级列）
    if '年级' not in df.columns:
        print("该数据集不包含'年级'列，无法绘制箱线图。")
        return

    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = [col for col in numeric_cols if col != '年级']
    if len(numeric_cols) == 0:
        print("没有可用的数值特征。")
        return

    # 只取前n个特征
    selected_cols = numeric_cols[:n]
    num_features = len(selected_cols)

    # 每行最多3个子图
    max_cols = 3
    n_cols = min(num_features, max_cols)
    n_rows = (num_features + max_cols - 1) // max_cols

    # 创建子图（不设置subplot_titles，即无子图标题）
    fig = make_subplots(rows=n_rows, cols=n_cols)

    for idx, col in enumerate(selected_cols):
        row = idx // max_cols + 1
        col_pos = idx % max_cols + 1
        # 按年级分组绘制箱线图
        grades = sorted(df['年级'].dropna().unique())
        for grade in grades:
            grade_data = df[df['年级'] == grade][col].dropna()
            fig.add_trace(
                go.Box(
                    y=grade_data,
                    name=f"{int(grade)}年级",
                    boxmean='sd',
                    showlegend=(idx == 0),  # 只在第一个子图显示图例
                ),
                row=row, col=col_pos
            )
        # 设置y轴标题
        fig.update_yaxes(title_text=col.split('-', 1)[0] if '-' in col else col, row=row, col=col_pos)
        # 设置x轴为年级
        fig.update_xaxes(title_text="年级", row=row, col=col_pos)

    fig.update_layout(
        #title_text=f"{file_name} 的特征箱线图",
        height=400 * n_rows,
        width=350 * n_cols,
        showlegend=True
    )
    fig.show()

def create_density_plots(df, feature_cols):
    """
    创建密度图展示年级间分布差异（美观高大上的版本）
    参数:
        df: 包含'年级'列和若干特征列的数据框
        feature_cols: 需要展示密度的特征列名列表
    """
    import matplotlib as mpl
    # 设置全局美化参数
    sns.set(palette="Set2", font_scale=1.2, rc={
        'axes.titlesize': 18,
        'axes.labelsize': 15,
        'legend.fontsize': 13,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'axes.titleweight': 'bold',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'figure.facecolor': 'white',
        'axes.facecolor': '#f7f7f7'
    })
    #mpl.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 支持中文
    mpl.rcParams['font.sans-serif'] = ['PingFang HK']  # 支持中文
    mpl.rcParams['axes.unicode_minus'] = False

    n_features = len(feature_cols)
    n_cols = 2 if n_features <= 2 else 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows))
    axes = axes.ravel()

    color_palette = sns.color_palette("husl", n_colors=len(df['年级'].dropna().unique()))

    for idx, feature in enumerate(feature_cols):
        if feature not in df.columns:
            continue
        ax = axes[idx]
        for i, grade in enumerate(sorted(df['年级'].dropna().unique())):
            grade_data = df[df['年级'] == grade][feature].dropna()
            if len(grade_data) > 0:
                sns.kdeplot(
                    data=grade_data,
                    label=f'{int(grade)}年级',
                    ax=ax,
                    fill=True,
                    linewidth=2.5,
                    alpha=0.7,
                    color=color_palette[i]
                )
        sub_title = feature.split('-')[0]
        ax.set_title(f'{sub_title}', fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('数值', fontsize=15)
        ax.set_ylabel('密度', fontsize=15)
        ax.legend(title='年级', loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, linestyle='--', alpha=0.3)
        # 去除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 隐藏多余的子图
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    #fig.suptitle('年级间分布密度对比', fontsize=22, fontweight='bold', color='#222222', y=1.03)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

def merge_data_dict_to_csv(data_dict, save_path="./pre_data/全部特征合并表.csv"):
    """
    将data_dict中的所有dataframe按照“文本编号”和“年级”两列进行横向合并，并保存为csv文件

    参数:
        data_dict: dict，key为表名，value为DataFrame
        save_path: str，保存路径
    """
    import functools
    import pandas as pd

    # 先将所有dataframe的“文本编号”和“年级”列名标准化，避免后续合并出错
    def standardize_id_grade(df):
        df = df.copy()
        # 统一列名
        for col in df.columns:
            if col.strip() in ['文本编号', '编号', 'file', 'File', 'filename']:
                df.rename(columns={col: '文本编号'}, inplace=True)
            if col.strip() in ['年级', 'grade', 'Grade', 'level']:
                df.rename(columns={col: '年级'}, inplace=True)
        return df

    data_dict_std = {k: standardize_id_grade(df) for k, df in data_dict.items()}

    # 合并时避免重复列名，给每个dataframe的特征列加前缀
    dfs_with_prefix = []
    for name, df in data_dict_std.items():
        # 除了“文本编号”和“年级”，其他列加前缀
        cols = df.columns.tolist()
        new_cols = []
        for col in cols:
            if col in ['文本编号', '年级']:
                new_cols.append(col)
            else:
                new_cols.append(f"{col}")
        df = df.copy()
        df.columns = new_cols
        dfs_with_prefix.append(df)

    # 依次按“文本编号”和“年级”做外连接
    df_merged = functools.reduce(lambda left, right: pd.merge(left, right, on=['文本编号', '年级'], how='outer'), dfs_with_prefix)

    # 保存合并后的大表
    df_merged.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"已将所有特征表合并并保存到: {save_path}")