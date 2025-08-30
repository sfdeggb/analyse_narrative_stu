# @pre_data/ 现在我要完成关于学生叙述能力的分析，请你分析统计要求内筒，结合pre_data的数据，分析是否可行，然后先给出一个分析方案。不需要代码实现
_Exported on 2025/8/30 at GMT+8 21:33:08 from Cursor (1.4.5)_

---

**User**

@pre_data/ 现在我要完成关于学生叙述能力的分析，请你分析统计要求内筒，结合pre_data的数据，分析是否可行，然后先给出一个分析方案。不需要代码实现

---

**Cursor**

我来分析统计要求文档和pre_data文件夹中的数据，评估分析可行性并给出分析方案。


基于对统计要求文档和pre_data文件夹中数据的分析，我来评估分析可行性并给出分析方案。

## 数据可行性分析

### 1. 数据完整性评估
✅ **数据充足**：pre_data文件夹包含7个CSV文件，每个文件都有302行数据（包含标题行），对应300个学生样本
✅ **年级分布均衡**：每个年级100个样本，共3个年级（高一、高二、高三）
✅ **特征维度丰富**：包含描述性统计、微观叙事（词法、句法）、宏观叙事（语篇连贯性、情境模型、段落长度）等多个维度

### 2. 数据质量评估
✅ **特征指标完整**：涵盖了统计要求中提到的所有分析维度
✅ **数据格式规范**：CSV格式，便于处理
✅ **指标命名清晰**：每个指标都有明确的含义和编号

## 分析方案

### 第一阶段：描述性统计分析
**目标**：了解各年级学生叙述能力的基本特征和分布情况

**分析内容**：
1. **基础统计量**：各年级在各项指标上的均值、标准差、中位数、四分位数
2. **分布特征**：通过直方图、密度图、箱线图展示各年级的分布差异
3. **年级间对比**：可视化展示三个年级在各维度上的表现差异

### 第二阶段：差异显著性分析
**目标**：检验不同年级间叙述能力是否存在统计学意义上的显著差异

**分析方法**：
1. **正态性检验**：使用Shapiro-Wilk检验判断数据是否符合正态分布
2. **方差齐性检验**：使用Levene检验判断各组方差是否相等
3. **单因素方差分析（ANOVA）**：如果数据符合正态分布且方差齐性，使用ANOVA
4. **非参数检验**：如果不符合正态分布，使用Kruskal-Wallis H检验
5. **事后检验**：如果存在显著差异，使用Tukey HSD或Mann-Whitney U检验进行两两比较

### 第三阶段：多维度综合分析
**目标**：从多个维度综合评估学生叙述能力的发展特点

**分析维度**：
1. **微观叙事能力**：
   - 词法维度：词汇多样性、密度、复杂度
   - 句法维度：短语长度、密度、句子结构

2. **宏观叙事能力**：
   - 语篇连贯性：局部连贯、整体连贯
   - 情境模型：因果衔接、意图衔接、时体衔接
   - 段落结构：段落长度及其变异性

3. **描述性统计**：
   - 文本长度：单词数量、句子数量
   - 结构特征：段落平均长度

### 第四阶段：发展模式分析
**目标**：识别学生叙述能力的发展规律和特点

**分析内容**：
1. **发展趋势**：分析各指标随年级升高的变化趋势
2. **发展速度**：比较不同维度的发展速度差异
3. **发展模式**：识别是否存在线性发展、阶段性发展等模式
4. **关键转折点**：识别能力发展的关键时期

### 第五阶段：相关性分析
**目标**：探索各维度指标间的相互关系

**分析内容**：
1. **维度内相关性**：分析同一维度内各指标的相关性
2. **维度间相关性**：分析不同维度间的相关性
3. **综合能力评估**：通过主成分分析等方法构建综合能力指标

## 预期成果

1. **统计报告**：详细的描述性统计和推断统计结果
2. **可视化图表**：箱线图、密度图、热力图、散点图等
3. **发展规律总结**：学生叙述能力的发展特点和规律
4. **教学建议**：基于分析结果的教学改进建议

这个分析方案充分利用了现有数据的优势，能够全面、系统地分析学生叙述能力的年级差异，为教学实践提供科学依据。

---

**User**

好的，首先我们完成统计要求中的第一部分：描述性统计
●年级间比较：“单词数量”和“句子数量”两个标准，在三个年级（高一、高二、高三）之间，三组数据是否有显著性差异，描述差异特征。
★年级内比较：内部每个人差异大小如何。请实现

---

**Cursor**

我来帮您完成统计要求中的第一部分：描述性统计分析。我将实现年级间比较和年级内比较两个部分。


现在我来创建一个完整的描述性统计分析脚本，实现年级间比较和年级内比较：

```python:analyse_v2/descriptive_analysis.py
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, tukey_hsd, levene, shapiro
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def load_and_prepare_data():
    """加载和准备数据"""
    # 读取数据
    data = pd.read_csv('../pre_data/描述性统计-grade.csv')
    
    # 将年级标签转换为更易理解的名称
    grade_mapping = {1: '高一', 2: '高二', 3: '高三'}
    data['年级名称'] = data['年级'].map(grade_mapping)
    
    # 按年级分组数据
    grade_groups = {grade: group for grade, group in data.groupby('年级名称')}
    
    return data, grade_groups

def descriptive_statistics(data, grade_groups):
    """描述性统计分析"""
    print("=" * 60)
    print("描述性统计分析报告")
    print("=" * 60)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(group_data)}")
        print(f"  单词数量 - 均值：{group_data['单词数量-DESWC-03'].mean():.2f}, 标准差：{group_data['单词数量-DESWC-03'].std():.2f}")
        print(f"  句子数量 - 均值：{group_data['句子数量-DESSC-02'].mean():.2f}, 标准差：{group_data['句子数量-DESSC-02'].std():.2f}")

def normality_test(data, grade_groups):
    """正态性检验"""
    print("\n3. 正态性检验：")
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        
        # 单词数量正态性检验
        stat_word, p_word = shapiro(group_data['单词数量-DESWC-03'])
        # 句子数量正态性检验
        stat_sent, p_sent = shapiro(group_data['句子数量-DESSC-02'])
        
        print(f"\n{grade_name}：")
        print(f"  单词数量 - Shapiro-Wilk检验：W={stat_word:.4f}, p={p_word:.4f} {'(正态分布)' if p_word > 0.05 else '(非正态分布)'}")
        print(f"  句子数量 - Shapiro-Wilk检验：W={stat_sent:.4f}, p={p_sent:.4f} {'(正态分布)' if p_sent > 0.05 else '(非正态分布)'}")

def variance_homogeneity_test(data, grade_groups):
    """方差齐性检验"""
    print("\n4. 方差齐性检验：")
    
    # 提取各年级数据
    grade1_words = grade_groups['高一']['单词数量-DESWC-03']
    grade2_words = grade_groups['高二']['单词数量-DESWC-03']
    grade3_words = grade_groups['高三']['单词数量-DESWC-03']
    
    grade1_sents = grade_groups['高一']['句子数量-DESSC-02']
    grade2_sents = grade_groups['高二']['句子数量-DESSC-02']
    grade3_sents = grade_groups['高三']['句子数量-DESSC-02']
    
    # Levene检验
    stat_word, p_word = levene(grade1_words, grade2_words, grade3_words)
    stat_sent, p_sent = levene(grade1_sents, grade2_sents, grade3_sents)
    
    print(f"单词数量 - Levene检验：F={stat_word:.4f}, p={p_word:.4f} {'(方差齐性)' if p_word > 0.05 else '(方差非齐性)'}")
    print(f"句子数量 - Levene检验：F={stat_sent:.4f}, p={p_sent:.4f} {'(方差齐性)' if p_sent > 0.05 else '(方差非齐性)'}")

def grade_comparison_analysis(data, grade_groups):
    """年级间比较分析"""
    print("\n5. 年级间比较分析：")
    
    # 提取各年级数据
    grade1_words = grade_groups['高一']['单词数量-DESWC-03']
    grade2_words = grade_groups['高二']['单词数量-DESWC-03']
    grade3_words = grade_groups['高三']['单词数量-DESWC-03']
    
    grade1_sents = grade_groups['高一']['句子数量-DESSC-02']
    grade2_sents = grade_groups['高二']['句子数量-DESSC-02']
    grade3_sents = grade_groups['高三']['句子数量-DESSC-02']
    
    # 单词数量分析
    print("\n单词数量年级间比较：")
    f_stat_word, p_word = f_oneway(grade1_words, grade2_words, grade3_words)
    print(f"  单因素方差分析：F={f_stat_word:.4f}, p={p_word:.4f}")
    
    if p_word < 0.05:
        print("  结果：存在显著性差异")
        # Tukey HSD事后检验
        tukey_result = tukey_hsd(grade1_words, grade2_words, grade3_words)
        print("  Tukey HSD事后检验结果：")
        print(f"    高一 vs 高二：p={tukey_result.pvalue[0]:.4f}")
        print(f"    高一 vs 高三：p={tukey_result.pvalue[1]:.4f}")
        print(f"    高二 vs 高三：p={tukey_result.pvalue[2]:.4f}")
    else:
        print("  结果：无显著性差异")
    
    # 句子数量分析
    print("\n句子数量年级间比较：")
    f_stat_sent, p_sent = f_oneway(grade1_sents, grade2_sents, grade3_sents)
    print(f"  单因素方差分析：F={f_stat_sent:.4f}, p={p_sent:.4f}")
    
    if p_sent < 0.05:
        print("  结果：存在显著性差异")
        # Tukey HSD事后检验
        tukey_result = tukey_hsd(grade1_sents, grade2_sents, grade3_sents)
        print("  Tukey HSD事后检验结果：")
        print(f"    高一 vs 高二：p={tukey_result.pvalue[0]:.4f}")
        print(f"    高一 vs 高三：p={tukey_result.pvalue[1]:.4f}")
        print(f"    高二 vs 高三：p={tukey_result.pvalue[2]:.4f}")
    else:
        print("  结果：无显著性差异")

def within_grade_analysis(data, grade_groups):
    """年级内比较分析"""
    print("\n6. 年级内比较分析：")
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        
        print(f"\n{grade_name}内部差异分析：")
        
        # 单词数量内部差异
        word_cv = group_data['单词数量-DESWC-03'].std() / group_data['单词数量-DESWC-03'].mean() * 100
        word_range = group_data['单词数量-DESWC-03'].max() - group_data['单词数量-DESWC-03'].min()
        word_iqr = group_data['单词数量-DESWC-03'].quantile(0.75) - group_data['单词数量-DESWC-03'].quantile(0.25)
        
        print(f"  单词数量：")
        print(f"    变异系数：{word_cv:.2f}%")
        print(f"    极差：{word_range}")
        print(f"    四分位距：{word_iqr:.2f}")
        
        # 句子数量内部差异
        sent_cv = group_data['句子数量-DESSC-02'].std() / group_data['句子数量-DESSC-02'].mean() * 100
        sent_range = group_data['句子数量-DESSC-02'].max() - group_data['句子数量-DESSC-02'].min()
        sent_iqr = group_data['句子数量-DESSC-02'].quantile(0.75) - group_data['句子数量-DESSC-02'].quantile(0.25)
        
        print(f"  句子数量：")
        print(f"    变异系数：{sent_cv:.2f}%")
        print(f"    极差：{sent_range}")
        print(f"    四分位距：{sent_iqr:.2f}")

def create_visualizations(data, grade_groups):
    """创建可视化图表"""
    print("\n7. 生成可视化图表...")
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('学生叙述能力描述性统计分析', fontsize=16, fontweight='bold')
    
    # 1. 单词数量箱线图
    word_data = [grade_groups['高一']['单词数量-DESWC-03'], 
                 grade_groups['高二']['单词数量-DESWC-03'], 
                 grade_groups['高三']['单词数量-DESWC-03']]
    axes[0, 0].boxplot(word_data, labels=['高一', '高二', '高三'])
    axes[0, 0].set_title('单词数量年级间比较')
    axes[0, 0].set_ylabel('单词数量')
    
    # 2. 句子数量箱线图
    sent_data = [grade_groups['高一']['句子数量-DESSC-02'], 
                 grade_groups['高二']['句子数量-DESSC-02'], 
                 grade_groups['高三']['句子数量-DESSC-02']]
    axes[0, 1].boxplot(sent_data, labels=['高一', '高二', '高三'])
    axes[0, 1].set_title('句子数量年级间比较')
    axes[0, 1].set_ylabel('句子数量')
    
    # 3. 单词数量密度图
    for grade_name in ['高一', '高二', '高三']:
        grade_data = grade_groups[grade_name]['单词数量-DESWC-03']
        axes[0, 2].hist(grade_data, alpha=0.6, label=grade_name, bins=15, density=True)
    axes[0, 2].set_title('单词数量分布密度图')
    axes[0, 2].set_xlabel('单词数量')
    axes[0, 2].set_ylabel('密度')
    axes[0, 2].legend()
    
    # 4. 句子数量密度图
    for grade_name in ['高一', '高二', '高三']:
        grade_data = grade_groups[grade_name]['句子数量-DESSC-02']
        axes[1, 0].hist(grade_data, alpha=0.6, label=grade_name, bins=15, density=True)
    axes[1, 0].set_title('句子数量分布密度图')
    axes[1, 0].set_xlabel('句子数量')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].legend()
    
    # 5. 散点图：单词数量 vs 句子数量
    colors = ['red', 'blue', 'green']
    for i, grade_name in enumerate(['高一', '高二', '高三']):
        grade_data = grade_groups[grade_name]
        axes[1, 1].scatter(grade_data['单词数量-DESWC-03'], 
                          grade_data['句子数量-DESSC-02'], 
                          alpha=0.6, label=grade_name, color=colors[i])
    axes[1, 1].set_title('单词数量与句子数量关系')
    axes[1, 1].set_xlabel('单词数量')
    axes[1, 1].set_ylabel('句子数量')
    axes[1, 1].legend()
    
    # 6. 均值比较条形图
    word_means = [grade_groups['高一']['单词数量-DESWC-03'].mean(),
                  grade_groups['高二']['单词数量-DESWC-03'].mean(),
                  grade_groups['高三']['单词数量-DESWC-03'].mean()]
    sent_means = [grade_groups['高一']['句子数量-DESSC-02'].mean(),
                  grade_groups['高二']['句子数量-DESSC-02'].mean(),
                  grade_groups['高三']['句子数量-DESSC-02'].mean()]
    
    x = np.arange(3)
    width = 0.35
    
    axes[1, 2].bar(x - width/2, word_means, width, label='单词数量', alpha=0.8)
    axes[1, 2].bar(x + width/2, sent_means, width, label='句子数量', alpha=0.8)
    axes[1, 2].set_title('各年级均值比较')
    axes[1, 2].set_xlabel('年级')
    axes[1, 2].set_ylabel('数量')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(['高一', '高二', '高三'])
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('../res/描述性统计分析结果.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存为 '../res/描述性统计分析结果.png'")

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 执行各项分析
    descriptive_statistics(data, grade_groups)
    normality_test(data, grade_groups)
    variance_homogeneity_test(data, grade_groups)
    grade_comparison_analysis(data, grade_groups)
    within_grade_analysis(data, grade_groups)
    
    # 创建可视化
    create_visualizations(data, grade_groups)
    
    print("\n" + "=" * 60)
    print("描述性统计分析完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

这个脚本实现了以下功能：

## 主要分析内容：

### 1. **年级间比较**（单词数量和句子数量）
- **正态性检验**：使用Shapiro-Wilk检验判断数据是否符合正态分布
- **方差齐性检验**：使用Levene检验判断各组方差是否相等
- **单因素方差分析（ANOVA）**：检验三个年级间是否存在显著性差异
- **事后检验**：如果存在显著差异，使用Tukey HSD进行两两比较

### 2. **年级内比较**（内部差异分析）
- **变异系数**：衡量相对变异程度
- **极差**：最大值与最小值的差
- **四分位距**：75%分位数与25%分位数的差

### 3. **可视化图表**
- 箱线图：显示年级间分布差异
- 密度图：显示各年级数据分布特征
- 散点图：显示单词数量与句子数量的关系
- 均值比较条形图：直观比较各年级均值

### 4. **统计报告输出**
- 详细的描述性统计信息
- 假设检验结果
- 显著性差异分析
- 内部差异特征描述

这个分析将全面回答统计要求中的问题：
- 三个年级在单词数量和句子数量上是否存在显著性差异
- 各年级内部学生之间的差异大小如何
- 提供详细的统计证据和可视化支持

您可以直接运行这个脚本来获得完整的分析结果。

```python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, tukey_hsd, levene, shapiro
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def load_and_prepare_data():
    """加载和准备数据"""
    # 读取数据
    data = pd.read_csv('../pre_data/描述性统计-grade.csv')
    
    # 将年级标签转换为更易理解的名称
    grade_mapping = {1: '高一', 2: '高二', 3: '高三'}
    data['年级名称'] = data['年级'].map(grade_mapping)
    
    # 按年级分组数据
    grade_groups = {grade: group for grade, group in data.groupby('年级名称')}
    
    return data, grade_groups

def descriptive_statistics(data, grade_groups):
    """描述性统计分析"""
    print("=" * 60)
    print("描述性统计分析报告")
    print("=" * 60)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(group_data)}")
        print(f"  单词数量 - 均值：{group_data['单词数量-DESWC-03'].mean():.2f}, 标准差：{group_data['单词数量-DESWC-03'].std():.2f}")
        print(f"  句子数量 - 均值：{group_data['句子数量-DESSC-02'].mean():.2f}, 标准差：{group_data['句子数量-DESSC-02'].std():.2f}")

def normality_test(data, grade_groups):
    """正态性检验"""
    print("\n3. 正态性检验：")
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        
        # 单词数量正态性检验
        stat_word, p_word = shapiro(group_data['单词数量-DESWC-03'])
        # 句子数量正态性检验
        stat_sent, p_sent = shapiro(group_data['句子数量-DESSC-02'])
        
        print(f"\n{grade_name}：")
        print(f"  单词数量 - Shapiro-Wilk检验：W={stat_word:.4f}, p={p_word:.4f} {'(正态分布)' if p_word > 0.05 else '(非正态分布)'}")
        print(f"  句子数量 - Shapiro-Wilk检验：W={stat_sent:.4f}, p={p_sent:.4f} {'(正态分布)' if p_sent > 0.05 else '(非正态分布)'}")

def variance_homogeneity_test(data, grade_groups):
    """方差齐性检验"""
    print("\n4. 方差齐性检验：")
    
    # 提取各年级数据
    grade1_words = grade_groups['高一']['单词数量-DESWC-03']
    grade2_words = grade_groups['高二']['单词数量-DESWC-03']
    grade3_words = grade_groups['高三']['单词数量-DESWC-03']
    
    grade1_sents = grade_groups['高一']['句子数量-DESSC-02']
    grade2_sents = grade_groups['高二']['句子数量-DESSC-02']
    grade3_sents = grade_groups['高三']['句子数量-DESSC-02']
    
    # Levene检验
    stat_word, p_word = levene(grade1_words, grade2_words, grade3_words)
    stat_sent, p_sent = levene(grade1_sents, grade2_sents, grade3_sents)
    
    print(f"单词数量 - Levene检验：F={stat_word:.4f}, p={p_word:.4f} {'(方差齐性)' if p_word > 0.05 else '(方差非齐性)'}")
    print(f"句子数量 - Levene检验：F={stat_sent:.4f}, p={p_sent:.4f} {'(方差齐性)' if p_sent > 0.05 else '(方差非齐性)'}")

def grade_comparison_analysis(data, grade_groups):
    """年级间比较分析"""
    print("\n5. 年级间比较分析：")
    
    # 提取各年级数据
    grade1_words = grade_groups['高一']['单词数量-DESWC-03']
    grade2_words = grade_groups['高二']['单词数量-DESWC-03']
    grade3_words = grade_groups['高三']['单词数量-DESWC-03']
    
    grade1_sents = grade_groups['高一']['句子数量-DESSC-02']
    grade2_sents = grade_groups['高二']['句子数量-DESSC-02']
    grade3_sents = grade_groups['高三']['句子数量-DESSC-02']
    
    # 单词数量分析
    print("\n单词数量年级间比较：")
    f_stat_word, p_word = f_oneway(grade1_words, grade2_words, grade3_words)
    print(f"  单因素方差分析：F={f_stat_word:.4f}, p={p_word:.4f}")
    
    if p_word < 0.05:
        print("  结果：存在显著性差异")
        # Tukey HSD事后检验
        tukey_result = tukey_hsd(grade1_words, grade2_words, grade3_words)
        print("  Tukey HSD事后检验结果：")
        print(f"    高一 vs 高二：p={tukey_result.pvalue[0]:.4f}")
        print(f"    高一 vs 高三：p={tukey_result.pvalue[1]:.4f}")
        print(f"    高二 vs 高三：p={tukey_result.pvalue[2]:.4f}")
    else:
        print("  结果：无显著性差异")
    
    # 句子数量分析
    print("\n句子数量年级间比较：")
    f_stat_sent, p_sent = f_oneway(grade1_sents, grade2_sents, grade3_sents)
    print(f"  单因素方差分析：F={f_stat_sent:.4f}, p={p_sent:.4f}")
    
    if p_sent < 0.05:
        print("  结果：存在显著性差异")
        # Tukey HSD事后检验
        tukey_result = tukey_hsd(grade1_sents, grade2_sents, grade3_sents)
        print("  Tukey HSD事后检验结果：")
        print(f"    高一 vs 高二：p={tukey_result.pvalue[0]:.4f}")
        print(f"    高一 vs 高三：p={tukey_result.pvalue[1]:.4f}")
        print(f"    高二 vs 高三：p={tukey_result.pvalue[2]:.4f}")
    else:
        print("  结果：无显著性差异")

def within_grade_analysis(data, grade_groups):
    """年级内比较分析"""
    print("\n6. 年级内比较分析：")
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        
        print(f"\n{grade_name}内部差异分析：")
        
        # 单词数量内部差异
        word_cv = group_data['单词数量-DESWC-03'].std() / group_data['单词数量-DESWC-03'].mean() * 100
        word_range = group_data['单词数量-DESWC-03'].max() - group_data['单词数量-DESWC-03'].min()
        word_iqr = group_data['单词数量-DESWC-03'].quantile(0.75) - group_data['单词数量-DESWC-03'].quantile(0.25)
        
        print(f"  单词数量：")
        print(f"    变异系数：{word_cv:.2f}%")
        print(f"    极差：{word_range}")
        print(f"    四分位距：{word_iqr:.2f}")
        
        # 句子数量内部差异
        sent_cv = group_data['句子数量-DESSC-02'].std() / group_data['句子数量-DESSC-02'].mean() * 100
        sent_range = group_data['句子数量-DESSC-02'].max() - group_data['句子数量-DESSC-02'].min()
        sent_iqr = group_data['句子数量-DESSC-02'].quantile(0.75) - group_data['句子数量-DESSC-02'].quantile(0.25)
        
        print(f"  句子数量：")
        print(f"    变异系数：{sent_cv:.2f}%")
        print(f"    极差：{sent_range}")
        print(f"    四分位距：{sent_iqr:.2f}")

def create_visualizations(data, grade_groups):
    """创建可视化图表"""
    print("\n7. 生成可视化图表...")
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('学生叙述能力描述性统计分析', fontsize=16, fontweight='bold')
    
    # 1. 单词数量箱线图
    word_data = [grade_groups['高一']['单词数量-DESWC-03'], 
                 grade_groups['高二']['单词数量-DESWC-03'], 
                 grade_groups['高三']['单词数量-DESWC-03']]
    axes[0, 0].boxplot(word_data, labels=['高一', '高二', '高三'])
    axes[0, 0].set_title('单词数量年级间比较')
    axes[0, 0].set_ylabel('单词数量')
    
    # 2. 句子数量箱线图
    sent_data = [grade_groups['高一']['句子数量-DESSC-02'], 
                 grade_groups['高二']['句子数量-DESSC-02'], 
                 grade_groups['高三']['句子数量-DESSC-02']]
    axes[0, 1].boxplot(sent_data, labels=['高一', '高二', '高三'])
    axes[0, 1].set_title('句子数量年级间比较')
    axes[0, 1].set_ylabel('句子数量')
    
    # 3. 单词数量密度图
    for grade_name in ['高一', '高二', '高三']:
        grade_data = grade_groups[grade_name]['单词数量-DESWC-03']
        axes[0, 2].hist(grade_data, alpha=0.6, label=grade_name, bins=15, density=True)
    axes[0, 2].set_title('单词数量分布密度图')
    axes[0, 2].set_xlabel('单词数量')
    axes[0, 2].set_ylabel('密度')
    axes[0, 2].legend()
    
    # 4. 句子数量密度图
    for grade_name in ['高一', '高二', '高三']:
        grade_data = grade_groups[grade_name]['句子数量-DESSC-02']
        axes[1, 0].hist(grade_data, alpha=0.6, label=grade_name, bins=15, density=True)
    axes[1, 0].set_title('句子数量分布密度图')
    axes[1, 0].set_xlabel('句子数量')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].legend()
    
    # 5. 散点图：单词数量 vs 句子数量
    colors = ['red', 'blue', 'green']
    for i, grade_name in enumerate(['高一', '高二', '高三']):
        grade_data = grade_groups[grade_name]
        axes[1, 1].scatter(grade_data['单词数量-DESWC-03'], 
                          grade_data['句子数量-DESSC-02'], 
                          alpha=0.6, label=grade_name, color=colors[i])
    axes[1, 1].set_title('单词数量与句子数量关系')
    axes[1, 1].set_xlabel('单词数量')
    axes[1, 1].set_ylabel('句子数量')
    axes[1, 1].legend()
    
    # 6. 均值比较条形图
    word_means = [grade_groups['高一']['单词数量-DESWC-03'].mean(),
                  grade_groups['高二']['单词数量-DESWC-03'].mean(),
                  grade_groups['高三']['单词数量-DESWC-03'].mean()]
    sent_means = [grade_groups['高一']['句子数量-DESSC-02'].mean(),
                  grade_groups['高二']['句子数量-DESSC-02'].mean(),
                  grade_groups['高三']['句子数量-DESSC-02'].mean()]
    
    x = np.arange(3)
    width = 0.35
    
    axes[1, 2].bar(x - width/2, word_means, width, label='单词数量', alpha=0.8)
    axes[1, 2].bar(x + width/2, sent_means, width, label='句子数量', alpha=0.8)
    axes[1, 2].set_title('各年级均值比较')
    axes[1, 2].set_xlabel('年级')
    axes[1, 2].set_ylabel('数量')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(['高一', '高二', '高三'])
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('../res/描述性统计分析结果.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图表已保存为 '../res/描述性统计分析结果.png'")

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 执行各项分析
    descriptive_statistics(data, grade_groups)
    normality_test(data, grade_groups)
    variance_homogeneity_test(data, grade_groups)
    grade_comparison_analysis(data, grade_groups)
    within_grade_analysis(data, grade_groups)
    
    # 创建可视化
    create_visualizations(data, grade_groups)
    
    print("\n" + "=" * 60)
    print("描述性统计分析完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

---

**User**

 22 tukey_result = tukey_hsd(grade1_words, grade2_words, grade3_words)
     23 print("  Tukey HSD事后检验结果：")
---> 24 print(f"    高一 vs 高二：p={tukey_result.pvalue[0]:.4f}")
     25 print(f"    高一 vs 高三：p={tukey_result.pvalue[1]:.4f}")
     26 print(f"    高二 vs 高三：p={tukey_result.pvalue[2]:.4f}")

TypeError: unsupported format string passed to numpy.ndarray.__format__ 这里运行会报错，为什么 如何解决

---

**Cursor**

这个错误是因为 `tukey_result.pvalue` 返回的是一个 numpy 数组，而 numpy 数组不支持直接使用 `.4f` 格式化。我们需要先提取数组中的具体数值，然后再进行格式化。

让我修复这个问题：

```python:analyse_v2/grade_analyse.ipynb
# ... existing code ...

def grade_comparison_analysis(data, grade_groups):
    """年级间比较分析"""
    print("\n5. 年级间比较分析：")
    
    # 提取各年级数据
    grade1_words = grade_groups['高一']['单词数量-DESWC-03']
    grade2_words = grade_groups['高二']['单词数量-DESWC-03']
    grade3_words = grade_groups['高三']['单词数量-DESWC-03']
    
    grade1_sents = grade_groups['高一']['句子数量-DESSC-02']
    grade2_sents = grade_groups['高二']['句子数量-DESSC-02']
    grade3_sents = grade_groups['高三']['句子数量-DESSC-02']
    
    # 单词数量分析
    print("\n单词数量年级间比较：")
    f_stat_word, p_word = f_oneway(grade1_words, grade2_words, grade3_words)
    print(f"  单因素方差分析：F={f_stat_word:.4f}, p={p_word:.4f}")
    
    if p_word < 0.05:
        print("  结果：存在显著性差异")
        # Tukey HSD事后检验
        tukey_result = tukey_hsd(grade1_words, grade2_words, grade3_words)
        print("  Tukey HSD事后检验结果：")
        # 修复：提取数组中的具体数值
        print(f"    高一 vs 高二：p={float(tukey_result.pvalue[0]):.4f}")
        print(f"    高一 vs 高三：p={float(tukey_result.pvalue[1]):.4f}")
        print(f"    高二 vs 高三：p={float(tukey_result.pvalue[2]):.4f}")
    else:
        print("  结果：无显著性差异")
    
    # 句子数量分析
    print("\n句子数量年级间比较：")
    f_stat_sent, p_sent = f_oneway(grade1_sents, grade2_sents, grade3_sents)
    print(f"  单因素方差分析：F={f_stat_sent:.4f}, p={p_sent:.4f}")
    
    if p_sent < 0.05:
        print("  结果：存在显著性差异")
        # Tukey HSD事后检验
        tukey_result = tukey_hsd(grade1_sents, grade2_sents, grade3_sents)
        print("  Tukey HSD事后检验结果：")
        # 修复：提取数组中的具体数值
        print(f"    高一 vs 高二：p={float(tukey_result.pvalue[0]):.4f}")
        print(f"    高一 vs 高三：p={float(tukey_result.pvalue[1]):.4f}")
        print(f"    高二 vs 高三：p={float(tukey_result.pvalue[2]):.4f}")
    else:
        print("  结果：无显著性差异")

# ... existing code ...
```

**问题原因：**
- `tukey_result.pvalue` 返回的是一个 numpy 数组，而不是单个数值
- numpy 数组不支持直接使用 `.4f` 格式化字符串
- 需要先提取数组中的具体数值，然后转换为 Python 的 float 类型

**解决方案：**
1. 使用 `float(tukey_result.pvalue[0])` 将 numpy 数组元素转换为 Python float
2. 然后就可以正常使用 `.4f` 格式化了

**其他可能的解决方案：**
```python
# 方案1：使用 float() 转换
print(f"    高一 vs 高二：p={float(tukey_result.pvalue[0]):.4f}")

# 方案2：使用 item() 方法
print(f"    高一 vs 高二：p={tukey_result.pvalue[0].item():.4f}")

# 方案3：使用 str() 和字符串格式化
print(f"    高一 vs 高二：p={str(tukey_result.pvalue[0])[:6]}")
```

推荐使用第一种方案（`float()` 转换），因为它最直观且兼容性最好。

```python
# ... existing code ...

def grade_comparison_analysis(data, grade_groups):
    """年级间比较分析"""
    print("\n5. 年级间比较分析：")
    
    # 提取各年级数据
    grade1_words = grade_groups['高一']['单词数量-DESWC-03']
    grade2_words = grade_groups['高二']['单词数量-DESWC-03']
    grade3_words = grade_groups['高三']['单词数量-DESWC-03']
    
    grade1_sents = grade_groups['高一']['句子数量-DESSC-02']
    grade2_sents = grade_groups['高二']['句子数量-DESSC-02']
    grade3_sents = grade_groups['高三']['句子数量-DESSC-02']
    
    # 单词数量分析
    print("\n单词数量年级间比较：")
    f_stat_word, p_word = f_oneway(grade1_words, grade2_words, grade3_words)
    print(f"  单因素方差分析：F={f_stat_word:.4f}, p={p_word:.4f}")
    
    if p_word < 0.05:
        print("  结果：存在显著性差异")
        # Tukey HSD事后检验
        tukey_result = tukey_hsd(grade1_words, grade2_words, grade3_words)
        print("  Tukey HSD事后检验结果：")
        # 修复：提取数组中的具体数值
        print(f"    高一 vs 高二：p={float(tukey_result.pvalue[0]):.4f}")
        print(f"    高一 vs 高三：p={float(tukey_result.pvalue[1]):.4f}")
        print(f"    高二 vs 高三：p={float(tukey_result.pvalue[2]):.4f}")
    else:
        print("  结果：无显著性差异")
    
    # 句子数量分析
    print("\n句子数量年级间比较：")
    f_stat_sent, p_sent = f_oneway(grade1_sents, grade2_sents, grade3_sents)
    print(f"  单因素方差分析：F={f_stat_sent:.4f}, p={p_sent:.4f}")
    
    if p_sent < 0.05:
        print("  结果：存在显著性差异")
        # Tukey HSD事后检验
        tukey_result = tukey_hsd(grade1_sents, grade2_sents, grade3_sents)
        print("  Tukey HSD事后检验结果：")
        # 修复：提取数组中的具体数值
        print(f"    高一 vs 高二：p={float(tukey_result.pvalue[0]):.4f}")
        print(f"    高一 vs 高三：p={float(tukey_result.pvalue[1]):.4f}")
        print(f"    高二 vs 高三：p={float(tukey_result.pvalue[2]):.4f}")
    else:
        print("  结果：无显著性差异")

# ... existing code ...
```

```python
# 方案1：使用 float() 转换
print(f"    高一 vs 高二：p={float(tukey_result.pvalue[0]):.4f}")

# 方案2：使用 item() 方法
print(f"    高一 vs 高二：p={tukey_result.pvalue[0].item():.4f}")

# 方案3：使用 str() 和字符串格式化
print(f"    高一 vs 高二：p={str(tukey_result.pvalue[0])[:6]}")
```

---

**User**

在上面的程序运行完之后我得到了的结果 @描述性统计分析结果.txt ，那么根据这个结果，该怎么回答●年级间比较：“单词数量”和“句子数量”两个标准，在三个年级（高一、高二、高三）之间，三组数据是否有显著性差异，描述差异特征。
★年级内比较：内部每个人差异大小这个问题呢

---

**Cursor**

根据您的描述性统计分析结果，我来帮您回答这两个问题：

## ● 年级间比较分析

### 单词数量年级间比较

**显著性差异检验结果：**
- **单因素方差分析：** F=23.7287, p=0.0000 < 0.05
- **结论：** 三个年级在单词数量上存在统计学意义上的显著性差异

**差异特征描述：**
1. **均值差异：**
   - 高一：199.85个单词
   - 高二：181.86个单词  
   - 高三：216.33个单词
   - 高三 > 高一 > 高二

2. **两两比较结果（Tukey HSD事后检验）：**
   - 高一 vs 高二：p=0.0011 < 0.05（显著差异）
   - 高一 vs 高三：p=0.0032 < 0.05（显著差异）
   - 高二 vs 高三：p=0.0000 < 0.05（显著差异）
   - **结论：** 任意两个年级之间都存在显著性差异

3. **发展特征：**
   - 从高一到高二，单词数量呈现下降趋势（199.85→181.86）
   - 从高二到高三，单词数量显著上升（181.86→216.33）
   - 高三学生的单词数量达到最高水平

### 句子数量年级间比较

**显著性差异检验结果：**
- **单因素方差分析：** F=5.9440, p=0.0029 < 0.05
- **结论：** 三个年级在句子数量上存在统计学意义上的显著性差异

**差异特征描述：**
1. **均值差异：**
   - 高一：14.80个句子
   - 高二：13.39个句子
   - 高三：13.79个句子
   - 高一 > 高三 > 高二

2. **两两比较结果（Tukey HSD事后检验）：**
   - 高一 vs 高二：p=0.0027 < 0.05（显著差异）
   - 高一 vs 高三：p=0.0451 < 0.05（显著差异）
   - 高二 vs 高三：p=0.6097 > 0.05（无显著差异）
   - **结论：** 高一与其他两个年级存在显著差异，但高二与高三之间无显著差异

3. **发展特征：**
   - 从高一到高二，句子数量下降（14.80→13.39）
   - 从高二到高三，句子数量略有回升（13.39→13.79）
   - 高一学生的句子数量明显高于其他年级

## ★ 年级内比较分析

### 单词数量内部差异

**变异系数（相对变异程度）：**
- 高一：17.33%
- 高二：21.81%
- 高三：14.51%

**差异特征：**
1. **高二内部差异最大：** 变异系数21.81%，说明高二学生之间单词数量的个体差异最为明显
2. **高三内部差异最小：** 变异系数14.51%，说明高三学生之间单词数量相对较为一致
3. **高一处于中等水平：** 变异系数17.33%

**极差和四分位距：**
- 高二的极差最大（195），四分位距最大（53.00），进一步证实高二学生内部差异最大
- 高三的极差最小（139），说明高三学生单词数量的分布范围相对集中

### 句子数量内部差异

**变异系数（相对变异程度）：**
- 高一：21.41%
- 高二：22.39%
- 高三：20.02%

**差异特征：**
1. **高二内部差异最大：** 变异系数22.39%，高二学生之间句子数量的个体差异最为明显
2. **高三内部差异最小：** 变异系数20.02%，高三学生之间句子数量相对较为一致
3. **高一处于中等水平：** 变异系数21.41%

**极差和四分位距：**
- 三个年级的极差相近（13-15），说明句子数量的绝对差异范围相似
- 四分位距也相近（3.25-3.50），表明句子数量的离散程度相对稳定

## 总结

1. **年级间差异：** 三个年级在单词数量和句子数量上都存在显著性差异，其中单词数量的差异更为明显
2. **发展模式：** 单词数量呈现"V"型发展（高一→高二下降，高二→高三上升），句子数量呈现下降趋势
3. **内部差异：** 高二学生内部的个体差异最大，高三学生内部的个体差异最小，这可能反映了不同年级阶段学生能力发展的不同特点

---

**User**

在  @grade_analyse.ipynb 创建可视化图表中create_visualization中，首先我希望里面不同类型的图我希望单独实现函数画出来。其次同时我希望每个图要专业美观高大尚。请帮我修改实现

---

**Cursor**

我来帮您重构 `create_visualizations` 函数，将不同类型的图表分别实现为独立的函数，并让每个图表都更加专业、美观、高大上。


现在我来重构可视化部分，创建专业、美观、高大上的图表：

```python:analyse_v2/grade_analyse.ipynb
# ... existing code ...

def create_boxplot_comparison(data, grade_groups, save_path='../res/boxplot_comparison.png'):
    """创建专业美观的箱线图比较"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 准备数据
    word_data = [grade_groups[grade]['单词数量-DESWC-03'] for grade in grade_names]
    sent_data = [grade_groups[grade]['句子数量-DESSC-02'] for grade in grade_names]
    
    # 单词数量箱线图
    bp1 = ax1.boxplot(word_data, patch_artist=True, 
                      boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                      medianprops=dict(color='#E74C3C', linewidth=3),
                      whiskerprops=dict(color='#2C3E50', linewidth=2),
                      capprops=dict(color='#2C3E50', linewidth=2),
                      flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=6, alpha=0.7))
    
    # 添加均值点
    means = [np.mean(data) for data in word_data]
    ax1.plot(range(1, len(means) + 1), means, 'D', color='#E74C3C', markersize=10, 
             markeredgecolor='white', markeredgewidth=2, label='均值')
    
    ax1.set_title('单词数量年级间比较', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax1.set_xlabel('年级', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_ylabel('单词数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12)
    
    # 句子数量箱线图
    bp2 = ax2.boxplot(sent_data, patch_artist=True,
                      boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                      medianprops=dict(color='#E74C3C', linewidth=3),
                      whiskerprops=dict(color='#2C3E50', linewidth=2),
                      capprops=dict(color='#2C3E50', linewidth=2),
                      flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=6, alpha=0.7))
    
    # 添加均值点
    means = [np.mean(data) for data in sent_data]
    ax2.plot(range(1, len(means) + 1), means, 'D', color='#E74C3C', markersize=10,
             markeredgecolor='white', markeredgewidth=2, label='均值')
    
    ax2.set_title('句子数量年级间比较', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax2.set_xlabel('年级', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_ylabel('句子数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=12)
    
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

def create_scatter_relationship(data, grade_groups, save_path='../res/scatter_relationship.png'):
    """创建专业美观的散点关系图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    grade_names = ['高一', '高二', '高三']
    
    for i, grade in enumerate(grade_names):
        grade_data = grade_groups[grade]
        ax.scatter(grade_data['单词数量-DESWC-03'], 
                  grade_data['句子数量-DESSC-02'],
                  alpha=0.7, s=80, c=colors[i], marker=markers[i], 
                  edgecolors='white', linewidth=1.5, label=grade)
    
    # 添加趋势线
    all_words = data['单词数量-DESWC-03']
    all_sents = data['句子数量-DESSC-02']
    z = np.polyfit(all_words, all_sents, 1)
    p = np.poly1d(z)
    ax.plot(all_words, p(all_words), "r--", alpha=0.8, linewidth=3, label='趋势线')
    
    # 计算相关系数
    corr = np.corrcoef(all_words, all_sents)[0, 1]
    ax.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax.transAxes, 
            fontsize=14, fontweight='bold', color='#2C3E50',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_title('单词数量与句子数量关系散点图', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('单词数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('句子数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"散点关系图已保存为: {save_path}")

def create_mean_comparison_bar(data, grade_groups, save_path='../res/mean_comparison_bar.png'):
    """创建专业美观的均值比较条形图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    grade_names = ['高一', '高二', '高三']
    word_means = [grade_groups[grade]['单词数量-DESWC-03'].mean() for grade in grade_names]
    sent_means = [grade_groups[grade]['句子数量-DESSC-02'].mean() for grade in grade_names]
    
    x = np.arange(len(grade_names))
    width = 0.35
    
    # 创建条形图
    bars1 = ax.bar(x - width/2, word_means, width, label='单词数量', 
                   color='#3498DB', alpha=0.8, edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, sent_means, width, label='句子数量',
                   color='#E74C3C', alpha=0.8, edgecolor='white', linewidth=2)
    
    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    ax.set_title('各年级均值比较', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('年级', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"均值比较条形图已保存为: {save_path}")

def create_violin_plot(data, grade_groups, save_path='../res/violin_plot.png'):
    """创建专业美观的小提琴图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 准备数据
    word_data = [grade_groups[grade]['单词数量-DESWC-03'] for grade in grade_names]
    sent_data = [grade_groups[grade]['句子数量-DESSC-02'] for grade in grade_names]
    
    # 单词数量小提琴图
    parts1 = ax1.violinplot(word_data, positions=range(1, len(grade_names) + 1), 
                           showmeans=True, showmedians=True)
    
    # 自定义小提琴图样式
    for pc in parts1['bodies']:
        pc.set_facecolor('#3498DB')
        pc.set_alpha(0.6)
        pc.set_edgecolor('#2C3E50')
        pc.set_linewidth(2)
    
    parts1['cmeans'].set_color('#E74C3C')
    parts1['cmeans'].set_linewidth(3)
    parts1['cmedians'].set_color('#F39C12')
    parts1['cmedians'].set_linewidth=3
    
    ax1.set_title('单词数量分布小提琴图', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax1.set_xlabel('年级', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_ylabel('单词数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_xticks(range(1, len(grade_names) + 1))
    ax1.set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 句子数量小提琴图
    parts2 = ax2.violinplot(sent_data, positions=range(1, len(grade_names) + 1),
                           showmeans=True, showmedians=True)
    
    for pc in parts2['bodies']:
        pc.set_facecolor('#E74C3C')
        pc.set_alpha(0.6)
        pc.set_edgecolor('#2C3E50')
        pc.set_linewidth(2)
    
    parts2['cmeans'].set_color('#E74C3C')
    parts2['cmeans'].set_linewidth(3)
    parts2['cmedians'].set_color('#F39C12')
    parts2['cmedians'].set_linewidth=3
    
    ax2.set_title('句子数量分布小提琴图', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax2.set_xlabel('年级', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_ylabel('句子数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_xticks(range(1, len(grade_names) + 1))
    ax2.set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"小提琴图已保存为: {save_path}")

def create_heatmap_correlation(data, grade_groups, save_path='../res/correlation_heatmap.png'):
    """创建专业美观的相关性热力图"""
    # 计算各年级的相关性矩阵
    correlations = {}
    for grade in ['高一', '高二', '高三']:
        grade_data = grade_groups[grade]
        corr_matrix = grade_data[['单词数量-DESWC-03', '句子数量-DESSC-02']].corr()
        correlations[grade] = corr_matrix
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (grade, corr_matrix) in enumerate(correlations.items()):
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=2, cbar_kws={"shrink": .8},
                   ax=axes[i], fmt='.3f', annot_kws={'size': 14, 'weight': 'bold'})
        
        axes[i].set_title(f'{grade}相关性矩阵', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        axes[i].tick_params(axis='both', which='major', labelsize=12, labelcolor='#2C3E50')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"相关性热力图已保存为: {save_path}")

def create_visualizations(data, grade_groups):
    """创建所有可视化图表"""
    print("\n7. 生成专业可视化图表...")
    
    # 创建输出目录
    import os
    os.makedirs('../res', exist_ok=True)
    
    # 生成各种图表
    create_boxplot_comparison(data, grade_groups)
    create_density_distribution(data, grade_groups)
    create_scatter_relationship(data, grade_groups)
    create_mean_comparison_bar(data, grade_groups)
    create_violin_plot(data, grade_groups)
    create_heatmap_correlation(data, grade_groups)
    
    print("\n所有图表生成完成！")

# ... existing code ...
```

这个重构后的可视化系统具有以下特点：

## �� **专业美观设计特点**

### 1. **色彩搭配**
- 使用专业的配色方案（#2E86AB, #A23B72, #F18F01）
- 统一的视觉风格和品牌色彩
- 适当的透明度和层次感

### 2. **图表类型丰富**
- **箱线图**：显示分布特征和异常值
- **密度分布图**：包含直方图和核密度估计曲线
- **散点关系图**：显示变量间关系，包含趋势线和相关系数
- **均值比较条形图**：直观比较各年级均值
- **小提琴图**：展示完整的数据分布形状
- **相关性热力图**：展示各年级内部变量相关性

### 3. **专业细节**
- 统一的字体样式和大小
- 清晰的标题和标签
- 网格线增强可读性
- 图例样式统一
- 高分辨率输出（300 DPI）

### 4. **功能增强**
- 每个图表独立保存
- 自动创建输出目录
- 详细的保存路径提示
- 模块化设计，便于维护和扩展

### 5. **统计信息展示**
- 均值点标注
- 相关系数显示
- 趋势线分析
- 数值标签

这样的设计让每个图表都更加专业、美观、高大上，同时保持了良好的可读性和学术价值。

```python
# ... existing code ...

def create_boxplot_comparison(data, grade_groups, save_path='../res/boxplot_comparison.png'):
    """创建专业美观的箱线图比较"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 准备数据
    word_data = [grade_groups[grade]['单词数量-DESWC-03'] for grade in grade_names]
    sent_data = [grade_groups[grade]['句子数量-DESSC-02'] for grade in grade_names]
    
    # 单词数量箱线图
    bp1 = ax1.boxplot(word_data, patch_artist=True, 
                      boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                      medianprops=dict(color='#E74C3C', linewidth=3),
                      whiskerprops=dict(color='#2C3E50', linewidth=2),
                      capprops=dict(color='#2C3E50', linewidth=2),
                      flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=6, alpha=0.7))
    
    # 添加均值点
    means = [np.mean(data) for data in word_data]
    ax1.plot(range(1, len(means) + 1), means, 'D', color='#E74C3C', markersize=10, 
             markeredgecolor='white', markeredgewidth=2, label='均值')
    
    ax1.set_title('单词数量年级间比较', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax1.set_xlabel('年级', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_ylabel('单词数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12)
    
    # 句子数量箱线图
    bp2 = ax2.boxplot(sent_data, patch_artist=True,
                      boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                      medianprops=dict(color='#E74C3C', linewidth=3),
                      whiskerprops=dict(color='#2C3E50', linewidth=2),
                      capprops=dict(color='#2C3E50', linewidth=2),
                      flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=6, alpha=0.7))
    
    # 添加均值点
    means = [np.mean(data) for data in sent_data]
    ax2.plot(range(1, len(means) + 1), means, 'D', color='#E74C3C', markersize=10,
             markeredgecolor='white', markeredgewidth=2, label='均值')
    
    ax2.set_title('句子数量年级间比较', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax2.set_xlabel('年级', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_ylabel('句子数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=12)
    
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

def create_scatter_relationship(data, grade_groups, save_path='../res/scatter_relationship.png'):
    """创建专业美观的散点关系图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    markers = ['o', 's', '^']
    grade_names = ['高一', '高二', '高三']
    
    for i, grade in enumerate(grade_names):
        grade_data = grade_groups[grade]
        ax.scatter(grade_data['单词数量-DESWC-03'], 
                  grade_data['句子数量-DESSC-02'],
                  alpha=0.7, s=80, c=colors[i], marker=markers[i], 
                  edgecolors='white', linewidth=1.5, label=grade)
    
    # 添加趋势线
    all_words = data['单词数量-DESWC-03']
    all_sents = data['句子数量-DESSC-02']
    z = np.polyfit(all_words, all_sents, 1)
    p = np.poly1d(z)
    ax.plot(all_words, p(all_words), "r--", alpha=0.8, linewidth=3, label='趋势线')
    
    # 计算相关系数
    corr = np.corrcoef(all_words, all_sents)[0, 1]
    ax.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax.transAxes, 
            fontsize=14, fontweight='bold', color='#2C3E50',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_title('单词数量与句子数量关系散点图', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('单词数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('句子数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"散点关系图已保存为: {save_path}")

def create_mean_comparison_bar(data, grade_groups, save_path='../res/mean_comparison_bar.png'):
    """创建专业美观的均值比较条形图"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    grade_names = ['高一', '高二', '高三']
    word_means = [grade_groups[grade]['单词数量-DESWC-03'].mean() for grade in grade_names]
    sent_means = [grade_groups[grade]['句子数量-DESSC-02'].mean() for grade in grade_names]
    
    x = np.arange(len(grade_names))
    width = 0.35
    
    # 创建条形图
    bars1 = ax.bar(x - width/2, word_means, width, label='单词数量', 
                   color='#3498DB', alpha=0.8, edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, sent_means, width, label='句子数量',
                   color='#E74C3C', alpha=0.8, edgecolor='white', linewidth=2)
    
    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    ax.set_title('各年级均值比较', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('年级', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_xticks(x)
    ax.set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"均值比较条形图已保存为: {save_path}")

def create_violin_plot(data, grade_groups, save_path='../res/violin_plot.png'):
    """创建专业美观的小提琴图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 准备数据
    word_data = [grade_groups[grade]['单词数量-DESWC-03'] for grade in grade_names]
    sent_data = [grade_groups[grade]['句子数量-DESSC-02'] for grade in grade_names]
    
    # 单词数量小提琴图
    parts1 = ax1.violinplot(word_data, positions=range(1, len(grade_names) + 1), 
                           showmeans=True, showmedians=True)
    
    # 自定义小提琴图样式
    for pc in parts1['bodies']:
        pc.set_facecolor('#3498DB')
        pc.set_alpha(0.6)
        pc.set_edgecolor('#2C3E50')
        pc.set_linewidth(2)
    
    parts1['cmeans'].set_color('#E74C3C')
    parts1['cmeans'].set_linewidth(3)
    parts1['cmedians'].set_color('#F39C12')
    parts1['cmedians'].set_linewidth=3
    
    ax1.set_title('单词数量分布小提琴图', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax1.set_xlabel('年级', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_ylabel('单词数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_xticks(range(1, len(grade_names) + 1))
    ax1.set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 句子数量小提琴图
    parts2 = ax2.violinplot(sent_data, positions=range(1, len(grade_names) + 1),
                           showmeans=True, showmedians=True)
    
    for pc in parts2['bodies']:
        pc.set_facecolor('#E74C3C')
        pc.set_alpha(0.6)
        pc.set_edgecolor('#2C3E50')
        pc.set_linewidth(2)
    
    parts2['cmeans'].set_color('#E74C3C')
    parts2['cmeans'].set_linewidth(3)
    parts2['cmedians'].set_color('#F39C12')
    parts2['cmedians'].set_linewidth=3
    
    ax2.set_title('句子数量分布小提琴图', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax2.set_xlabel('年级', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_ylabel('句子数量', fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_xticks(range(1, len(grade_names) + 1))
    ax2.set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"小提琴图已保存为: {save_path}")

def create_heatmap_correlation(data, grade_groups, save_path='../res/correlation_heatmap.png'):
    """创建专业美观的相关性热力图"""
    # 计算各年级的相关性矩阵
    correlations = {}
    for grade in ['高一', '高二', '高三']:
        grade_data = grade_groups[grade]
        corr_matrix = grade_data[['单词数量-DESWC-03', '句子数量-DESSC-02']].corr()
        correlations[grade] = corr_matrix
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (grade, corr_matrix) in enumerate(correlations.items()):
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=2, cbar_kws={"shrink": .8},
                   ax=axes[i], fmt='.3f', annot_kws={'size': 14, 'weight': 'bold'})
        
        axes[i].set_title(f'{grade}相关性矩阵', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        axes[i].tick_params(axis='both', which='major', labelsize=12, labelcolor='#2C3E50')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"相关性热力图已保存为: {save_path}")

def create_visualizations(data, grade_groups):
    """创建所有可视化图表"""
    print("\n7. 生成专业可视化图表...")
    
    # 创建输出目录
    import os
    os.makedirs('../res', exist_ok=True)
    
    # 生成各种图表
    create_boxplot_comparison(data, grade_groups)
    create_density_distribution(data, grade_groups)
    create_scatter_relationship(data, grade_groups)
    create_mean_comparison_bar(data, grade_groups)
    create_violin_plot(data, grade_groups)
    create_heatmap_correlation(data, grade_groups)
    
    print("\n所有图表生成完成！")

# ... existing code ...
```

---

**User**

@grade_analyse_2.ipynb 好的现在我们进行统计要求的第二部分，●年级间比较：每组数据（如高一）同其他两组数据（如高二、高三）相比，三个年级之间是否有显著差异，描述差异特征；
★年级内比较：同一个年级内，“多样性”1个指标、“密度”1个指标、“复杂性”6个指标中，哪组数据的内部差异程度大，描述差异特征，差异程度大的是影响该阶段能力发展的关键因素。请实现

---

**Cursor**

我来帮您实现统计要求的第二部分，分析微观叙事词法维度的年级间比较和年级内比较。


现在我来实现微观叙事词法维度的分析。根据数据，我们需要分析以下指标：

**多样性指标：** 多样性-LDVOCDa（51）
**密度指标：** 密度-WRDFRQC（94）  
**复杂性指标：** 复杂度-K1, 复杂度-K2, 复杂度-K3, 复杂度-K4, 复杂度-K5, 复杂度-K6+

```python:analyse_v2/grade_analyse_2.ipynb
# ... existing code ...

def descriptive_statistics_lexical(data, grade_groups):
    """微观叙事词法描述性统计分析"""
    print("=" * 80)
    print("微观叙事词法描述性统计分析报告")
    print("=" * 80)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
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
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(group_data)}")
        
        for name, col in indicators.items():
            mean_val = group_data[col].mean()
            std_val = group_data[col].std()
            print(f"  {name} - 均值：{mean_val:.3f}, 标准差：{std_val:.3f}")

def normality_test_lexical(data, grade_groups):
    """正态性检验"""
    print("\n3. 正态性检验：")
    
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
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}：")
        
        for name, col in indicators.items():
            stat, p_val = shapiro(group_data[col])
            print(f"  {name} - Shapiro-Wilk检验：W={stat:.4f}, p={p_val:.4f} {'(正态分布)' if p_val > 0.05 else '(非正态分布)'}")

def variance_homogeneity_test_lexical(data, grade_groups):
    """方差齐性检验"""
    print("\n4. 方差齐性检验：")
    
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
    
    for name, col in indicators.items():
        grade1_data = grade_groups['高一'][col]
        grade2_data = grade_groups['高二'][col]
        grade3_data = grade_groups['高三'][col]
        
        stat, p_val = levene(grade1_data, grade2_data, grade3_data)
        print(f"{name} - Levene检验：F={stat:.4f}, p={p_val:.4f} {'(方差齐性)' if p_val > 0.05 else '(方差非齐性)'}")

def grade_comparison_analysis_lexical(data, grade_groups):
    """年级间比较分析"""
    print("\n5. 年级间比较分析：")
    
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
    
    for name, col in indicators.items():
        print(f"\n{name}年级间比较：")
        
        grade1_data = grade_groups['高一'][col]
        grade2_data = grade_groups['高二'][col]
        grade3_data = grade_groups['高三'][col]
        
        # 单因素方差分析
        f_stat, p_val = f_oneway(grade1_data, grade2_data, grade3_data)
        print(f"  单因素方差分析：F={f_stat:.4f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            print("  结果：存在显著性差异")
            # Tukey HSD事后检验
            tukey_result = tukey_hsd(grade1_data, grade2_data, grade3_data)
            print("  Tukey HSD事后检验结果：")
            print(f"    高一 vs 高二：p={float(tukey_result.pvalue[0]):.4f}")
            print(f"    高一 vs 高三：p={float(tukey_result.pvalue[1]):.4f}")
            print(f"    高二 vs 高三：p={float(tukey_result.pvalue[2]):.4f}")
        else:
            print("  结果：无显著性差异")

def within_grade_analysis_lexical(data, grade_groups):
    """年级内比较分析"""
    print("\n6. 年级内比较分析：")
    
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
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}内部差异分析：")
        
        # 计算各指标的变异系数
        cv_results = {}
        for name, col in indicators.items():
            cv = group_data[col].std() / group_data[col].mean() * 100
            cv_results[name] = cv
        
        # 按变异系数排序
        sorted_cv = sorted(cv_results.items(), key=lambda x: x[1], reverse=True)
        
        print("  各指标变异系数（从大到小排序）：")
        for name, cv in sorted_cv:
            print(f"    {name}：{cv:.2f}%")
        
        # 找出差异最大的指标
        max_cv_indicator = sorted_cv[0]
        print(f"\n  差异程度最大的指标：{max_cv_indicator[0]}（变异系数：{max_cv_indicator[1]:.2f}%）")
        print(f"  该指标是影响{grade_name}阶段能力发展的关键因素")

def create_lexical_visualizations(data, grade_groups):
    """创建微观叙事词法可视化图表"""
    print("\n7. 生成微观叙事词法可视化图表...")
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 创建输出目录
    import os
    os.makedirs('../res', exist_ok=True)
    
    # 1. 箱线图比较
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('微观叙事词法各指标年级间比较', fontsize=16, fontweight='bold')
    
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
    
    for i, (name, col) in enumerate(indicators.items()):
        row = i // 4
        col_idx = i % 4
        
        # 准备数据
        data_list = [grade_groups[grade][col] for grade in grade_names]
        
        # 创建箱线图
        bp = axes[row, col_idx].boxplot(data_list, patch_artist=True,
                                       boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                                       medianprops=dict(color='#E74C3C', linewidth=3),
                                       whiskerprops=dict(color='#2C3E50', linewidth=2),
                                       capprops=dict(color='#2C3E50', linewidth=2),
                                       flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=4, alpha=0.7))
        
        # 添加均值点
        means = [np.mean(data) for data in data_list]
        axes[row, col_idx].plot(range(1, len(means) + 1), means, 'D', color='#E74C3C', markersize=8,
                               markeredgecolor='white', markeredgewidth=2, label='均值')
        
        axes[row, col_idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[row, col_idx].set_xticklabels(grade_names, fontsize=10)
        axes[row, col_idx].grid(True, alpha=0.3, linestyle='--')
        axes[row, col_idx].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('../res/微观叙事词法箱线图比较.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("箱线图已保存为: '../res/微观叙事词法箱线图比较.png'")
    
    # 2. 变异系数热力图
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
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
    
    # 创建热力图
    im = ax.imshow(cv_matrix, cmap='RdYlBu_r', aspect='auto')
    
    # 设置标签
    ax.set_xticks(range(len(indicators)))
    ax.set_yticks(range(len(grade_names)))
    ax.set_xticklabels(list(indicators.keys()), rotation=45, ha='right')
    ax.set_yticklabels(grade_names)
    
    # 添加数值标签
    for i in range(len(grade_names)):
        for j in range(len(indicators)):
            text = ax.text(j, i, f'{cv_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('各年级各指标变异系数热力图', fontsize=16, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='变异系数 (%)')
    
    plt.tight_layout()
    plt.savefig('../res/微观叙事词法变异系数热力图.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("变异系数热力图已保存为: '../res/微观叙事词法变异系数热力图.png'")

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 执行各项分析
    descriptive_statistics_lexical(data, grade_groups)
    normality_test_lexical(data, grade_groups)
    variance_homogeneity_test_lexical(data, grade_groups)
    grade_comparison_analysis_lexical(data, grade_groups)
    within_grade_analysis_lexical(data, grade_groups)
    
    # 创建可视化
    create_lexical_visualizations(data, grade_groups)
    
    print("\n" + "=" * 80)
    print("微观叙事词法分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

这个分析脚本实现了以下功能：

## 主要分析内容：

### 1. **年级间比较分析**
- **单因素方差分析（ANOVA）**：检验三个年级在各指标上是否存在显著性差异
- **Tukey HSD事后检验**：如果存在显著差异，进行两两比较
- **差异特征描述**：详细描述各年级间的差异模式

### 2. **年级内比较分析**
- **变异系数计算**：计算各指标在年级内的变异系数
- **差异程度排序**：按变异系数大小排序，找出差异最大的指标
- **关键因素识别**：识别影响该阶段能力发展的关键因素

### 3. **分析指标**
- **多样性指标**：多样性-LDVOCDa（51）
- **密度指标**：密度-WRDFRQC（94）
- **复杂性指标**：复杂度-K1到复杂度-K6+（6个指标）

### 4. **可视化图表**
- **箱线图比较**：显示各年级在各指标上的分布差异
- **变异系数热力图**：直观显示各年级各指标的内部差异程度

### 5. **统计报告输出**
- 详细的描述性统计信息
- 假设检验结果
- 显著性差异分析
- 内部差异特征描述
- 关键因素识别

这个分析将全面回答统计要求中的问题：
- 三个年级在微观叙事词法各指标上是否存在显著性差异
- 各年级内部哪些指标的差异程度最大
- 识别影响各阶段能力发展的关键因素

您可以直接运行这个脚本来获得完整的分析结果。

```python
# ... existing code ...

def descriptive_statistics_lexical(data, grade_groups):
    """微观叙事词法描述性统计分析"""
    print("=" * 80)
    print("微观叙事词法描述性统计分析报告")
    print("=" * 80)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
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
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(group_data)}")
        
        for name, col in indicators.items():
            mean_val = group_data[col].mean()
            std_val = group_data[col].std()
            print(f"  {name} - 均值：{mean_val:.3f}, 标准差：{std_val:.3f}")

def normality_test_lexical(data, grade_groups):
    """正态性检验"""
    print("\n3. 正态性检验：")
    
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
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}：")
        
        for name, col in indicators.items():
            stat, p_val = shapiro(group_data[col])
            print(f"  {name} - Shapiro-Wilk检验：W={stat:.4f}, p={p_val:.4f} {'(正态分布)' if p_val > 0.05 else '(非正态分布)'}")

def variance_homogeneity_test_lexical(data, grade_groups):
    """方差齐性检验"""
    print("\n4. 方差齐性检验：")
    
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
    
    for name, col in indicators.items():
        grade1_data = grade_groups['高一'][col]
        grade2_data = grade_groups['高二'][col]
        grade3_data = grade_groups['高三'][col]
        
        stat, p_val = levene(grade1_data, grade2_data, grade3_data)
        print(f"{name} - Levene检验：F={stat:.4f}, p={p_val:.4f} {'(方差齐性)' if p_val > 0.05 else '(方差非齐性)'}")

def grade_comparison_analysis_lexical(data, grade_groups):
    """年级间比较分析"""
    print("\n5. 年级间比较分析：")
    
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
    
    for name, col in indicators.items():
        print(f"\n{name}年级间比较：")
        
        grade1_data = grade_groups['高一'][col]
        grade2_data = grade_groups['高二'][col]
        grade3_data = grade_groups['高三'][col]
        
        # 单因素方差分析
        f_stat, p_val = f_oneway(grade1_data, grade2_data, grade3_data)
        print(f"  单因素方差分析：F={f_stat:.4f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            print("  结果：存在显著性差异")
            # Tukey HSD事后检验
            tukey_result = tukey_hsd(grade1_data, grade2_data, grade3_data)
            print("  Tukey HSD事后检验结果：")
            print(f"    高一 vs 高二：p={float(tukey_result.pvalue[0]):.4f}")
            print(f"    高一 vs 高三：p={float(tukey_result.pvalue[1]):.4f}")
            print(f"    高二 vs 高三：p={float(tukey_result.pvalue[2]):.4f}")
        else:
            print("  结果：无显著性差异")

def within_grade_analysis_lexical(data, grade_groups):
    """年级内比较分析"""
    print("\n6. 年级内比较分析：")
    
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
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}内部差异分析：")
        
        # 计算各指标的变异系数
        cv_results = {}
        for name, col in indicators.items():
            cv = group_data[col].std() / group_data[col].mean() * 100
            cv_results[name] = cv
        
        # 按变异系数排序
        sorted_cv = sorted(cv_results.items(), key=lambda x: x[1], reverse=True)
        
        print("  各指标变异系数（从大到小排序）：")
        for name, cv in sorted_cv:
            print(f"    {name}：{cv:.2f}%")
        
        # 找出差异最大的指标
        max_cv_indicator = sorted_cv[0]
        print(f"\n  差异程度最大的指标：{max_cv_indicator[0]}（变异系数：{max_cv_indicator[1]:.2f}%）")
        print(f"  该指标是影响{grade_name}阶段能力发展的关键因素")

def create_lexical_visualizations(data, grade_groups):
    """创建微观叙事词法可视化图表"""
    print("\n7. 生成微观叙事词法可视化图表...")
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 创建输出目录
    import os
    os.makedirs('../res', exist_ok=True)
    
    # 1. 箱线图比较
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('微观叙事词法各指标年级间比较', fontsize=16, fontweight='bold')
    
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
    
    for i, (name, col) in enumerate(indicators.items()):
        row = i // 4
        col_idx = i % 4
        
        # 准备数据
        data_list = [grade_groups[grade][col] for grade in grade_names]
        
        # 创建箱线图
        bp = axes[row, col_idx].boxplot(data_list, patch_artist=True,
                                       boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                                       medianprops=dict(color='#E74C3C', linewidth=3),
                                       whiskerprops=dict(color='#2C3E50', linewidth=2),
                                       capprops=dict(color='#2C3E50', linewidth=2),
                                       flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=4, alpha=0.7))
        
        # 添加均值点
        means = [np.mean(data) for data in data_list]
        axes[row, col_idx].plot(range(1, len(means) + 1), means, 'D', color='#E74C3C', markersize=8,
                               markeredgecolor='white', markeredgewidth=2, label='均值')
        
        axes[row, col_idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[row, col_idx].set_xticklabels(grade_names, fontsize=10)
        axes[row, col_idx].grid(True, alpha=0.3, linestyle='--')
        axes[row, col_idx].legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('../res/微观叙事词法箱线图比较.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("箱线图已保存为: '../res/微观叙事词法箱线图比较.png'")
    
    # 2. 变异系数热力图
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
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
    
    # 创建热力图
    im = ax.imshow(cv_matrix, cmap='RdYlBu_r', aspect='auto')
    
    # 设置标签
    ax.set_xticks(range(len(indicators)))
    ax.set_yticks(range(len(grade_names)))
    ax.set_xticklabels(list(indicators.keys()), rotation=45, ha='right')
    ax.set_yticklabels(grade_names)
    
    # 添加数值标签
    for i in range(len(grade_names)):
        for j in range(len(indicators)):
            text = ax.text(j, i, f'{cv_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('各年级各指标变异系数热力图', fontsize=16, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='变异系数 (%)')
    
    plt.tight_layout()
    plt.savefig('../res/微观叙事词法变异系数热力图.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("变异系数热力图已保存为: '../res/微观叙事词法变异系数热力图.png'")

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 执行各项分析
    descriptive_statistics_lexical(data, grade_groups)
    normality_test_lexical(data, grade_groups)
    variance_homogeneity_test_lexical(data, grade_groups)
    grade_comparison_analysis_lexical(data, grade_groups)
    within_grade_analysis_lexical(data, grade_groups)
    
    # 创建可视化
    create_lexical_visualizations(data, grade_groups)
    
    print("\n" + "=" * 80)
    print("微观叙事词法分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

---

**User**

输出的结果为 @分析报告.txt ，现在该如何回答要求的问题：●年级间比较：每组数据（如高一）同其他两组数据（如高二、高三）相比，三个年级之间是否有显著差异，描述差异特征；
★年级内比较：同一个年级内，“多样性”1个指标、“密度”1个指标、“复杂性”6个指标中，哪组数据的内部差异程度大，描述差异特征，差异程度大的是影响该阶段能力发展的关键因素。

---

**Cursor**

根据分析报告的结果，我来帮您回答这两个问题：

## ● 年级间比较分析

### 1. **多样性指标**
- **显著性差异检验：** F=1.1455, p=0.3195 > 0.05
- **结论：** 三个年级在词汇多样性上无显著性差异
- **差异特征：** 各年级的词汇多样性水平相对稳定，均值在86-90之间

### 2. **密度指标**
- **显著性差异检验：** F=30.3682, p=0.0000 < 0.05
- **结论：** 三个年级在词汇密度上存在显著性差异
- **差异特征：**
  - 高一（2.322）> 高二（2.222）> 高三（2.162）
  - 所有年级间两两比较均存在显著差异
  - 呈现随年级升高而下降的趋势

### 3. **复杂性指标**

#### **复杂度-K1（基础词汇复杂度）**
- **显著性差异检验：** F=32.0677, p=0.0000 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高一（87.714）> 高二（84.823）> 高三（83.166）
  - 所有年级间两两比较均存在显著差异
  - 呈现随年级升高而下降的趋势

#### **复杂度-K2（中等复杂度词汇）**
- **显著性差异检验：** F=7.2884, p=0.0008 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（8.033）> 高二（7.502）> 高一（6.926）
  - 仅高一与高三存在显著差异
  - 呈现随年级升高而上升的趋势

#### **复杂度-K3（较高复杂度词汇）**
- **显著性差异检验：** F=38.3531, p=0.0000 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（4.733）> 高二（3.989）> 高一（2.602）
  - 所有年级间两两比较均存在显著差异
  - 呈现随年级升高而显著上升的趋势

#### **复杂度-K4（高复杂度词汇）**
- **显著性差异检验：** F=12.8534, p=0.0000 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（1.939）> 高二（1.608）> 高一（1.211）
  - 高一与其他两个年级存在显著差异
  - 呈现随年级升高而上升的趋势

#### **复杂度-K5（极高复杂度词汇）**
- **显著性差异检验：** F=4.7203, p=0.0096 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（1.263）> 高二（1.185）> 高一（0.949）
  - 仅高一与高三存在显著差异
  - 呈现随年级升高而上升的趋势

#### **复杂度-K6+（最高复杂度词汇）**
- **显著性差异检验：** F=1.7094, p=0.1828 > 0.05
- **结论：** 无显著性差异
- **差异特征：** 各年级在最高复杂度词汇使用上差异不显著

## ★ 年级内比较分析

### **高一内部差异分析**
**各指标变异系数排序（从大到小）：**
1. 复杂度-K6+：112.82%
2. 复杂度-K5：68.60%
3. 复杂度-K4：56.60%
4. 复杂度-K3：46.00%
5. 复杂度-K2：31.39%
6. 多样性：18.81%
7. 密度：4.68%
8. 复杂度-K1：3.25%

**关键因素：** 复杂度-K6+（变异系数112.82%）是影响高一阶段能力发展的关键因素

### **高二内部差异分析**
**各指标变异系数排序（从大到小）：**
1. 复杂度-K6+：140.21%
2. 复杂度-K5：64.89%
3. 复杂度-K4：64.67%
4. 复杂度-K3：48.10%
5. 复杂度-K2：26.97%
6. 多样性：19.35%
7. 密度：7.57%
8. 复杂度-K1：5.28%

**关键因素：** 复杂度-K6+（变异系数140.21%）是影响高二阶段能力发展的关键因素

### **高三内部差异分析**
**各指标变异系数排序（从大到小）：**
1. 复杂度-K6+：162.47%
2. 复杂度-K5：65.17%
3. 复杂度-K4：64.29%
4. 复杂度-K3：42.41%
5. 复杂度-K2：24.24%
6. 多样性：17.23%
7. 密度：7.28%
8. 复杂度-K1：5.56%

**关键因素：** 复杂度-K6+（变异系数162.47%）是影响高三阶段能力发展的关键因素

## 总结

### **年级间差异特征：**
1. **词汇多样性**在各年级间保持稳定
2. **词汇密度**和**基础词汇复杂度**随年级升高而下降
3. **中等及以上复杂度词汇**随年级升高而增加
4. **最高复杂度词汇**在各年级间差异不显著

### **年级内差异特征：**
1. **所有年级**都以**复杂度-K6+**为内部差异最大的指标
2. **差异程度随年级升高而增大**（高一112.82% → 高二140.21% → 高三162.47%）
3. **复杂度-K5和K4**在所有年级中都表现出较大的内部差异
4. **基础指标**（多样性、密度、复杂度-K1）的内部差异相对较小

### **关键发现：**
**复杂度-K6+（最高复杂度词汇）**是影响各阶段能力发展的关键因素，其内部差异程度随年级升高而显著增大，说明高年级学生在高级词汇使用能力上存在更大的个体差异。

---

**User**

在上面给出的可视化create_lexical_visualizations函数这里我希望箱线图和变异系数热力图分成两个函数实现，请实现

---

**Cursor**

我来帮您将 `create_lexical_visualizations` 函数中的箱线图和变异系数热力图分成两个独立的函数实现：

```python:analyse_v2/grade_analyse_2.ipynb
# ... existing code ...

def create_lexical_boxplots(data, grade_groups, save_path='../res/微观叙事词法箱线图比较.png'):
    """创建微观叙事词法箱线图比较"""
    print("生成微观叙事词法箱线图...")
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
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
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('微观叙事词法各指标年级间比较', fontsize=16, fontweight='bold', color='#2C3E50')
    
    for i, (name, col) in enumerate(indicators.items()):
        row = i // 4
        col_idx = i % 4
        
        # 准备数据
        data_list = [grade_groups[grade][col] for grade in grade_names]
        
        # 创建箱线图
        bp = axes[row, col_idx].boxplot(data_list, patch_artist=True,
                                       boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                                       medianprops=dict(color='#E74C3C', linewidth=3),
                                       whiskerprops=dict(color='#2C3E50', linewidth=2),
                                       capprops=dict(color='#2C3E50', linewidth=2),
                                       flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=4, alpha=0.7))
        
        # 添加均值点
        means = [np.mean(data) for data in data_list]
        axes[row, col_idx].plot(range(1, len(means) + 1), means, 'D', color='#E74C3C', markersize=8,
                               markeredgecolor='white', markeredgewidth=2, label='均值')
        
        # 设置标题和标签
        axes[row, col_idx].set_title(f'{name}', fontsize=12, fontweight='bold', color='#2C3E50')
        axes[row, col_idx].set_xticklabels(grade_names, fontsize=10, fontweight='bold')
        axes[row, col_idx].grid(True, alpha=0.3, linestyle='--')
        axes[row, col_idx].legend(fontsize=8, frameon=True, fancybox=True, shadow=True)
        
        # 设置坐标轴标签颜色
        axes[row, col_idx].tick_params(axis='both', which='major', colors='#2C3E50')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"箱线图已保存为: {save_path}")

def create_lexical_cv_heatmap(data, grade_groups, save_path='../res/微观叙事词法变异系数热力图.png'):
    """创建微观叙事词法变异系数热力图"""
    print("生成微观叙事词法变异系数热力图...")
    
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
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # 创建热力图
    im = ax.imshow(cv_matrix, cmap='RdYlBu_r', aspect='auto', vmin=cv_matrix.min(), vmax=cv_matrix.max())
    
    # 设置标签
    ax.set_xticks(range(len(indicators)))
    ax.set_yticks(range(len(grade_names)))
    ax.set_xticklabels(list(indicators.keys()), rotation=45, ha='right', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.set_yticklabels(grade_names, fontsize=12, fontweight='bold', color='#2C3E50')
    
    # 添加数值标签
    for i in range(len(grade_names)):
        for j in range(len(indicators)):
            text = ax.text(j, i, f'{cv_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=11)
    
    # 设置标题
    ax.set_title('各年级各指标变异系数热力图', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='变异系数 (%)', shrink=0.8)
    cbar.ax.tick_params(labelsize=10, colors='#2C3E50')
    cbar.ax.set_ylabel('变异系数 (%)', fontsize=12, fontweight='bold', color='#2C3E50')
    
    # 添加网格线
    ax.set_xticks(np.arange(-.5, len(indicators), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(grade_names), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"变异系数热力图已保存为: {save_path}")

def create_lexical_visualizations(data, grade_groups):
    """创建所有微观叙事词法可视化图表"""
    print("\n7. 生成微观叙事词法可视化图表...")
    
    # 创建输出目录
    import os
    os.makedirs('../res', exist_ok=True)
    
    # 生成各种图表
    create_lexical_boxplots(data, grade_groups)
    create_lexical_cv_heatmap(data, grade_groups)
    
    print("\n所有微观叙事词法图表生成完成！")

# ... existing code ...
```

这样重构后的代码具有以下特点：

## **功能分离优势**

### 1. **模块化设计**
- **`create_lexical_boxplots`**：专门负责生成箱线图
- **`create_lexical_cv_heatmap`**：专门负责生成变异系数热力图
- **`create_lexical_visualizations`**：作为主函数调用两个子函数

### 2. **独立参数控制**
- 每个函数都有独立的 `save_path` 参数
- 可以单独调用某个图表类型
- 便于调试和维护

### 3. **专业美观设计**

#### **箱线图特点：**
- 2×4 布局，展示8个指标
- 统一的颜色主题和样式
- 均值点标注
- 网格线和图例

#### **热力图特点：**
- 直观显示变异系数差异
- 数值标签清晰显示
- 颜色渐变表示差异程度
- 专业的颜色条设计

### 4. **使用方式**

```python
# 单独生成箱线图
create_lexical_boxplots(data, grade_groups, '../res/自定义箱线图路径.png')

# 单独生成热力图
create_lexical_cv_heatmap(data, grade_groups, '../res/自定义热力图路径.png')

# 生成所有图表
create_lexical_visualizations(data, grade_groups)
```

### 5. **代码维护性**
- 每个函数职责单一
- 便于后续修改和扩展
- 代码结构清晰
- 易于理解和维护

这样的设计让代码更加模块化，每个函数都有明确的职责，便于后续的维护和扩展。

```python
# ... existing code ...

def create_lexical_boxplots(data, grade_groups, save_path='../res/微观叙事词法箱线图比较.png'):
    """创建微观叙事词法箱线图比较"""
    print("生成微观叙事词法箱线图...")
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
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
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('微观叙事词法各指标年级间比较', fontsize=16, fontweight='bold', color='#2C3E50')
    
    for i, (name, col) in enumerate(indicators.items()):
        row = i // 4
        col_idx = i % 4
        
        # 准备数据
        data_list = [grade_groups[grade][col] for grade in grade_names]
        
        # 创建箱线图
        bp = axes[row, col_idx].boxplot(data_list, patch_artist=True,
                                       boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                                       medianprops=dict(color='#E74C3C', linewidth=3),
                                       whiskerprops=dict(color='#2C3E50', linewidth=2),
                                       capprops=dict(color='#2C3E50', linewidth=2),
                                       flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=4, alpha=0.7))
        
        # 添加均值点
        means = [np.mean(data) for data in data_list]
        axes[row, col_idx].plot(range(1, len(means) + 1), means, 'D', color='#E74C3C', markersize=8,
                               markeredgecolor='white', markeredgewidth=2, label='均值')
        
        # 设置标题和标签
        axes[row, col_idx].set_title(f'{name}', fontsize=12, fontweight='bold', color='#2C3E50')
        axes[row, col_idx].set_xticklabels(grade_names, fontsize=10, fontweight='bold')
        axes[row, col_idx].grid(True, alpha=0.3, linestyle='--')
        axes[row, col_idx].legend(fontsize=8, frameon=True, fancybox=True, shadow=True)
        
        # 设置坐标轴标签颜色
        axes[row, col_idx].tick_params(axis='both', which='major', colors='#2C3E50')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"箱线图已保存为: {save_path}")

def create_lexical_cv_heatmap(data, grade_groups, save_path='../res/微观叙事词法变异系数热力图.png'):
    """创建微观叙事词法变异系数热力图"""
    print("生成微观叙事词法变异系数热力图...")
    
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
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # 创建热力图
    im = ax.imshow(cv_matrix, cmap='RdYlBu_r', aspect='auto', vmin=cv_matrix.min(), vmax=cv_matrix.max())
    
    # 设置标签
    ax.set_xticks(range(len(indicators)))
    ax.set_yticks(range(len(grade_names)))
    ax.set_xticklabels(list(indicators.keys()), rotation=45, ha='right', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.set_yticklabels(grade_names, fontsize=12, fontweight='bold', color='#2C3E50')
    
    # 添加数值标签
    for i in range(len(grade_names)):
        for j in range(len(indicators)):
            text = ax.text(j, i, f'{cv_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontweight='bold', fontsize=11)
    
    # 设置标题
    ax.set_title('各年级各指标变异系数热力图', fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='变异系数 (%)', shrink=0.8)
    cbar.ax.tick_params(labelsize=10, colors='#2C3E50')
    cbar.ax.set_ylabel('变异系数 (%)', fontsize=12, fontweight='bold', color='#2C3E50')
    
    # 添加网格线
    ax.set_xticks(np.arange(-.5, len(indicators), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(grade_names), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"变异系数热力图已保存为: {save_path}")

def create_lexical_visualizations(data, grade_groups):
    """创建所有微观叙事词法可视化图表"""
    print("\n7. 生成微观叙事词法可视化图表...")
    
    # 创建输出目录
    import os
    os.makedirs('../res', exist_ok=True)
    
    # 生成各种图表
    create_lexical_boxplots(data, grade_groups)
    create_lexical_cv_heatmap(data, grade_groups)
    
    print("\n所有微观叙事词法图表生成完成！")

# ... existing code ...
```

```python
# 单独生成箱线图
create_lexical_boxplots(data, grade_groups, '../res/自定义箱线图路径.png')

# 单独生成热力图
create_lexical_cv_heatmap(data, grade_groups, '../res/自定义热力图路径.png')

# 生成所有图表
create_lexical_visualizations(data, grade_groups)
```

---

**User**

在拆分成两个函数之后，我在 @grade_analyse_2.ipynb 中使用了create_lexical_boxplots，现在我想按照 @grede_analyse_1_untils.py 中箱线图函数的逻辑和图表样式进行修改，请实现

---

**Cursor**

我来帮您按照 `grede_analyse_1_untils.py` 中箱线图函数的逻辑和图表样式来修改 `create_lexical_boxplots` 函数。


现在我来按照 `grede_analyse_1_untils.py` 中的箱线图函数逻辑和样式来修改 `create_lexical_boxplots` 函数：

```python:analyse_v2/grade_analyse_2.ipynb
# ... existing code ...

def create_lexical_boxplots(data, grade_groups, save_path='../res/微观叙事词法箱线图比较.png'):
    """创建微观叙事词法箱线图比较 - 采用高级美观的seaborn风格"""
    print("生成微观叙事词法箱线图...")
    
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

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
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
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), dpi=120, constrained_layout=True)
    fig.suptitle('微观叙事词法各指标年级间分布对比', fontsize=24, fontweight='bold', color='#222', y=0.98)
    
    # 导入pandas用于数据处理
    import pandas as pd
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    for i, (name, col) in enumerate(indicators.items()):
        row = i // 4
        col_idx = i % 4
        
        # 准备数据
        data_list = [grade_groups[grade][col] for grade in grade_names]
        
        # 构造DataFrame用于seaborn
        df = pd.DataFrame({
            '年级': np.repeat(grade_names, [len(x) for x in data_list]),
            '数值': np.concatenate(data_list)
        })
        
        # 创建小提琴图+箱线图组合
        sns.violinplot(
            x='年级', y='数值', data=df, ax=axes[row, col_idx],
            inner=None, palette=colors, linewidth=0, alpha=0.18
        )
        sns.boxplot(
            x='年级', y='数值', data=df, ax=axes[row, col_idx],
            width=0.25, palette=colors, boxprops=dict(alpha=0.7)
        )
        
        # 添加均值点
        means = df.groupby('年级')['数值'].mean()
        axes[row, col_idx].scatter(
            range(len(grade_names)), means, 
            color='#E74C3C', s=80, marker='D', 
            edgecolor='white', zorder=10, label='均值'
        )
        
        # 美化设置
        axes[row, col_idx].set_title(f'{name}', fontsize=16, fontweight='bold', color='#222')
        axes[row, col_idx].set_xlabel('年级', fontsize=14, fontweight='bold')
        axes[row, col_idx].set_ylabel('数值', fontsize=14, fontweight='bold')
        axes[row, col_idx].grid(axis='y', linestyle='--', alpha=0.25)
        axes[row, col_idx].set_axisbelow(True)
        axes[row, col_idx].spines['top'].set_visible(False)
        axes[row, col_idx].spines['right'].set_visible(False)
        
        # 设置x轴标签
        axes[row, col_idx].set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    
    # 自定义图例（只在第一个子图中显示）
    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='高一', alpha=0.7),
        Patch(facecolor='#A23B72', edgecolor='#A23B72', label='高二', alpha=0.7),
        Patch(facecolor='#F18F01', edgecolor='#F18F01', label='高三', alpha=0.7),
        Line2D([0], [0], marker='D', color='w', label='均值', 
               markerfacecolor='#E74C3C', markeredgecolor='white', markersize=10)
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper right', 
                     frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 为总标题留出空间
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"箱线图已保存为: {save_path}")

def create_lexical_cv_heatmap(data, grade_groups, save_path='../res/微观叙事词法变异系数热力图.png'):
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

def create_lexical_visualizations(data, grade_groups):
    """创建所有微观叙事词法可视化图表"""
    print("\n7. 生成微观叙事词法可视化图表...")
    
    # 创建输出目录
    import os
    os.makedirs('../res', exist_ok=True)
    
    # 生成各种图表
    create_lexical_boxplots(data, grade_groups)
    create_lexical_cv_heatmap(data, grade_groups)
    
    print("\n所有微观叙事词法图表生成完成！")

# ... existing code ...
```

这个修改后的函数具有以下特点：

## **主要改进**

### 1. **采用seaborn高级风格**
- 使用 `sns.set_theme()` 设置全局风格
- 统一的字体、颜色和样式配置
- 专业的网格线和边框设置

### 2. **小提琴图+箱线图组合**
- 小提琴图显示完整的数据分布形状
- 箱线图叠加显示关键统计信息
- 均值点用菱形标记突出显示

### 3. **专业的美化设计**
- 统一的颜色主题（#2E86AB, #A23B72, #F18F01）
- 去除顶部和右侧边框
- 网格线增强可读性
- 自定义图例样式

### 4. **布局优化**
- 2×4 布局展示8个指标
- 总标题和子标题层次分明
- 适当的间距和边距设置

### 5. **热力图改进**
- 使用seaborn的heatmap函数
- 更好的颜色映射和标注
- 旋转x轴标签提高可读性

### 6. **代码结构**
- 模块化设计，便于维护
- 清晰的注释和文档
- 统一的保存和输出格式

这样的设计让图表更加专业、美观、高大上，同时保持了良好的可读性和学术价值。

```python
# ... existing code ...

def create_lexical_boxplots(data, grade_groups, save_path='../res/微观叙事词法箱线图比较.png'):
    """创建微观叙事词法箱线图比较 - 采用高级美观的seaborn风格"""
    print("生成微观叙事词法箱线图...")
    
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

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
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
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), dpi=120, constrained_layout=True)
    fig.suptitle('微观叙事词法各指标年级间分布对比', fontsize=24, fontweight='bold', color='#222', y=0.98)
    
    # 导入pandas用于数据处理
    import pandas as pd
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    for i, (name, col) in enumerate(indicators.items()):
        row = i // 4
        col_idx = i % 4
        
        # 准备数据
        data_list = [grade_groups[grade][col] for grade in grade_names]
        
        # 构造DataFrame用于seaborn
        df = pd.DataFrame({
            '年级': np.repeat(grade_names, [len(x) for x in data_list]),
            '数值': np.concatenate(data_list)
        })
        
        # 创建小提琴图+箱线图组合
        sns.violinplot(
            x='年级', y='数值', data=df, ax=axes[row, col_idx],
            inner=None, palette=colors, linewidth=0, alpha=0.18
        )
        sns.boxplot(
            x='年级', y='数值', data=df, ax=axes[row, col_idx],
            width=0.25, palette=colors, boxprops=dict(alpha=0.7)
        )
        
        # 添加均值点
        means = df.groupby('年级')['数值'].mean()
        axes[row, col_idx].scatter(
            range(len(grade_names)), means, 
            color='#E74C3C', s=80, marker='D', 
            edgecolor='white', zorder=10, label='均值'
        )
        
        # 美化设置
        axes[row, col_idx].set_title(f'{name}', fontsize=16, fontweight='bold', color='#222')
        axes[row, col_idx].set_xlabel('年级', fontsize=14, fontweight='bold')
        axes[row, col_idx].set_ylabel('数值', fontsize=14, fontweight='bold')
        axes[row, col_idx].grid(axis='y', linestyle='--', alpha=0.25)
        axes[row, col_idx].set_axisbelow(True)
        axes[row, col_idx].spines['top'].set_visible(False)
        axes[row, col_idx].spines['right'].set_visible(False)
        
        # 设置x轴标签
        axes[row, col_idx].set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    
    # 自定义图例（只在第一个子图中显示）
    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='高一', alpha=0.7),
        Patch(facecolor='#A23B72', edgecolor='#A23B72', label='高二', alpha=0.7),
        Patch(facecolor='#F18F01', edgecolor='#F18F01', label='高三', alpha=0.7),
        Line2D([0], [0], marker='D', color='w', label='均值', 
               markerfacecolor='#E74C3C', markeredgecolor='white', markersize=10)
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper right', 
                     frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 为总标题留出空间
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"箱线图已保存为: {save_path}")

def create_lexical_cv_heatmap(data, grade_groups, save_path='../res/微观叙事词法变异系数热力图.png'):
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

def create_lexical_visualizations(data, grade_groups):
    """创建所有微观叙事词法可视化图表"""
    print("\n7. 生成微观叙事词法可视化图表...")
    
    # 创建输出目录
    import os
    os.makedirs('../res', exist_ok=True)
    
    # 生成各种图表
    create_lexical_boxplots(data, grade_groups)
    create_lexical_cv_heatmap(data, grade_groups)
    
    print("\n所有微观叙事词法图表生成完成！")

# ... existing code ...
```

---

**User**

@grade_analyse_3.ipynb  现在我们进行统计要求中第三部分 @tmp.txt  请实现

---

**Cursor**

我来帮您实现统计要求的第三部分，分析微观叙事句法维度的年级间比较和年级内比较。


现在我来实现微观叙事句法维度的分析。根据数据，我们需要分析以下指标：

**短语长度指标：** 短语长度-SYNLE（69）、短语长度-SYNNP（70）
**短语密度指标：** 短语密度-DRNP（76）、短语密度-DRVP（77）、短语密度-DRPP（79）
**句子长度指标：** 句子长度-DESSL（06）
**句子结构指标：** 句子结构-DRPVAL（80）、句子结构-DRNEG（81）、句子结构-DRINF（83）
**句法相似指标：** 句法相似-SYNSTRUTa（74）、句法相似-SYNSTRUTt（75）

```python:analyse_v2/grade_analyse_3.ipynb
# ... existing code ...

def descriptive_statistics_syntactic(data, grade_groups):
    """微观叙事句法描述性统计分析"""
    print("=" * 80)
    print("微观叙事句法描述性统计分析报告")
    print("=" * 80)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
    indicators = {
        '短语长度-SYNLE': '短语长度-SYNLE（69）',
        '短语长度-SYNNP': '短语长度-SYNNP（70）',
        '短语密度-DRNP': '短语密度-DRNP（76）',
        '短语密度-DRVP': '短语密度-DRVP（77）',
        '短语密度-DRPP': '短语密度-DRPP（79）',
        '句子长度-DESSL': '句子长度-DESSL（06）',
        '句子结构-DRPVAL': '句子结构-DRPVAL（80）',
        '句子结构-DRNEG': '句子结构-DRNEG（81）',
        '句子结构-DRINF': '句子结构-DRINF（83）',
        '句法相似-SYNSTRUTa': '句法相似-SYNSTRUTa（74）',
        '句法相似-SYNSTRUTt': '句法相似-SYNSTRUTt（75）'
    }
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(group_data)}")
        
        for name, col in indicators.items():
            mean_val = group_data[col].mean()
            std_val = group_data[col].std()
            print(f"  {name} - 均值：{mean_val:.3f}, 标准差：{std_val:.3f}")

def normality_test_syntactic(data, grade_groups):
    """正态性检验"""
    print("\n3. 正态性检验：")
    
    indicators = {
        '短语长度-SYNLE': '短语长度-SYNLE（69）',
        '短语长度-SYNNP': '短语长度-SYNNP（70）',
        '短语密度-DRNP': '短语密度-DRNP（76）',
        '短语密度-DRVP': '短语密度-DRVP（77）',
        '短语密度-DRPP': '短语密度-DRPP（79）',
        '句子长度-DESSL': '句子长度-DESSL（06）',
        '句子结构-DRPVAL': '句子结构-DRPVAL（80）',
        '句子结构-DRNEG': '句子结构-DRNEG（81）',
        '句子结构-DRINF': '句子结构-DRINF（83）',
        '句法相似-SYNSTRUTa': '句法相似-SYNSTRUTa（74）',
        '句法相似-SYNSTRUTt': '句法相似-SYNSTRUTt（75）'
    }
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}：")
        
        for name, col in indicators.items():
            stat, p_val = shapiro(group_data[col])
            print(f"  {name} - Shapiro-Wilk检验：W={stat:.4f}, p={p_val:.4f} {'(正态分布)' if p_val > 0.05 else '(非正态分布)'}")

def variance_homogeneity_test_syntactic(data, grade_groups):
    """方差齐性检验"""
    print("\n4. 方差齐性检验：")
    
    indicators = {
        '短语长度-SYNLE': '短语长度-SYNLE（69）',
        '短语长度-SYNNP': '短语长度-SYNNP（70）',
        '短语密度-DRNP': '短语密度-DRNP（76）',
        '短语密度-DRVP': '短语密度-DRVP（77）',
        '短语密度-DRPP': '短语密度-DRPP（79）',
        '句子长度-DESSL': '句子长度-DESSL（06）',
        '句子结构-DRPVAL': '句子结构-DRPVAL（80）',
        '句子结构-DRNEG': '句子结构-DRNEG（81）',
        '句子结构-DRINF': '句子结构-DRINF（83）',
        '句法相似-SYNSTRUTa': '句法相似-SYNSTRUTa（74）',
        '句法相似-SYNSTRUTt': '句法相似-SYNSTRUTt（75）'
    }
    
    for name, col in indicators.items():
        grade1_data = grade_groups['高一'][col]
        grade2_data = grade_groups['高二'][col]
        grade3_data = grade_groups['高三'][col]
        
        stat, p_val = levene(grade1_data, grade2_data, grade3_data)
        print(f"{name} - Levene检验：F={stat:.4f}, p={p_val:.4f} {'(方差齐性)' if p_val > 0.05 else '(方差非齐性)'}")

def grade_comparison_analysis_syntactic(data, grade_groups):
    """年级间比较分析"""
    print("\n5. 年级间比较分析：")
    
    indicators = {
        '短语长度-SYNLE': '短语长度-SYNLE（69）',
        '短语长度-SYNNP': '短语长度-SYNNP（70）',
        '短语密度-DRNP': '短语密度-DRNP（76）',
        '短语密度-DRVP': '短语密度-DRVP（77）',
        '短语密度-DRPP': '短语密度-DRPP（79）',
        '句子长度-DESSL': '句子长度-DESSL（06）',
        '句子结构-DRPVAL': '句子结构-DRPVAL（80）',
        '句子结构-DRNEG': '句子结构-DRNEG（81）',
        '句子结构-DRINF': '句子结构-DRINF（83）',
        '句法相似-SYNSTRUTa': '句法相似-SYNSTRUTa（74）',
        '句法相似-SYNSTRUTt': '句法相似-SYNSTRUTt（75）'
    }
    
    for name, col in indicators.items():
        print(f"\n{name}年级间比较：")
        
        grade1_data = grade_groups['高一'][col]
        grade2_data = grade_groups['高二'][col]
        grade3_data = grade_groups['高三'][col]
        
        # 单因素方差分析
        f_stat, p_val = f_oneway(grade1_data, grade2_data, grade3_data)
        print(f"  单因素方差分析：F={f_stat:.4f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            print("  结果：存在显著性差异")
            # Tukey HSD事后检验
            tukey_result = tukey_hsd(grade1_data, grade2_data, grade3_data)
            print("  Tukey HSD事后检验结果：")
            print(f"    高一 vs 高二：p={float(tukey_result.pvalue[0]):.4f}")
            print(f"    高一 vs 高三：p={float(tukey_result.pvalue[1]):.4f}")
            print(f"    高二 vs 高三：p={float(tukey_result.pvalue[2]):.4f}")
        else:
            print("  结果：无显著性差异")

def within_grade_analysis_syntactic(data, grade_groups):
    """年级内比较分析"""
    print("\n6. 年级内比较分析：")
    
    # 定义指标分组
    indicator_groups = {
        '短语长度指标': ['短语长度-SYNLE（69）', '短语长度-SYNNP（70）'],
        '短语密度指标': ['短语密度-DRNP（76）', '短语密度-DRVP（77）', '短语密度-DRPP（79）'],
        '句子长度指标': ['句子长度-DESSL（06）'],
        '句子结构指标': ['句子结构-DRPVAL（80）', '句子结构-DRNEG（81）', '句子结构-DRINF（83）'],
        '句法相似指标': ['句法相似-SYNSTRUTa（74）', '句法相似-SYNSTRUTt（75）']
    }
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}内部差异分析：")
        
        # 计算各组指标的变异系数
        group_cv_results = {}
        for group_name, cols in indicator_groups.items():
            group_cvs = []
            for col in cols:
                cv = group_data[col].std() / group_data[col].mean() * 100
                group_cvs.append(cv)
            # 取该组指标的平均变异系数
            group_cv_results[group_name] = np.mean(group_cvs)
        
        # 按变异系数排序
        sorted_group_cv = sorted(group_cv_results.items(), key=lambda x: x[1], reverse=True)
        
        print("  各组指标平均变异系数（从大到小排序）：")
        for group_name, cv in sorted_group_cv:
            print(f"    {group_name}：{cv:.2f}%")
        
        # 找出差异最大的指标组
        max_cv_group = sorted_group_cv[0]
        print(f"\n  差异程度最大的指标组：{max_cv_group[0]}（平均变异系数：{max_cv_group[1]:.2f}%）")
        print(f"  该指标组是影响{grade_name}阶段能力发展的关键因素")
        
        # 详细分析该组内各指标的变异系数
        print(f"\n  {max_cv_group[0]}内部各指标变异系数：")
        for col in indicator_groups[max_cv_group[0]]:
            cv = group_data[col].std() / group_data[col].mean() * 100
            print(f"    {col}：{cv:.2f}%")

def create_syntactic_visualizations(data, grade_groups):
    """创建微观叙事句法可视化图表"""
    print("\n7. 生成微观叙事句法可视化图表...")
    
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
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '短语长度-SYNLE': '短语长度-SYNLE（69）',
        '短语长度-SYNNP': '短语长度-SYNNP（70）',
        '短语密度-DRNP': '短语密度-DRNP（76）',
        '短语密度-DRVP': '短语密度-DRVP（77）',
        '短语密度-DRPP': '短语密度-DRPP（79）',
        '句子长度-DESSL': '句子长度-DESSL（06）',
        '句子结构-DRPVAL': '句子结构-DRPVAL（80）',
        '句子结构-DRNEG': '句子结构-DRNEG（81）',
        '句子结构-DRINF': '句子结构-DRINF（83）',
        '句法相似-SYNSTRUTa': '句法相似-SYNSTRUTa（74）',
        '句法相似-SYNSTRUTt': '句法相似-SYNSTRUTt（75）'
    }
    
    # 创建输出目录
    import os
    os.makedirs('../res', exist_ok=True)
    
    # 1. 箱线图比较
    fig, axes = plt.subplots(3, 4, figsize=(24, 18), dpi=120, constrained_layout=True)
    fig.suptitle('微观叙事句法各指标年级间分布对比', fontsize=24, fontweight='bold', color='#222', y=0.98)
    
    import pandas as pd
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    for i, (name, col) in enumerate(indicators.items()):
        row = i // 4
        col_idx = i % 4
        
        # 准备数据
        data_list = [grade_groups[grade][col] for grade in grade_names]
        
        # 构造DataFrame用于seaborn
        df = pd.DataFrame({
            '年级': np.repeat(grade_names, [len(x) for x in data_list]),
            '数值': np.concatenate(data_list)
        })
        
        # 创建小提琴图+箱线图组合
        sns.violinplot(
            x='年级', y='数值', data=df, ax=axes[row, col_idx],
            inner=None, palette=colors, linewidth=0, alpha=0.18
        )
        sns.boxplot(
            x='年级', y='数值', data=df, ax=axes[row, col_idx],
            width=0.25, palette=colors, boxprops=dict(alpha=0.7)
        )
        
        # 添加均值点
        means = df.groupby('年级')['数值'].mean()
        axes[row, col_idx].scatter(
            range(len(grade_names)), means, 
            color='#E74C3C', s=80, marker='D', 
            edgecolor='white', zorder=10, label='均值'
        )
        
        # 美化设置
        axes[row, col_idx].set_title(f'{name}', fontsize=16, fontweight='bold', color='#222')
        axes[row, col_idx].set_xlabel('年级', fontsize=14, fontweight='bold')
        axes[row, col_idx].set_ylabel('数值', fontsize=14, fontweight='bold')
        axes[row, col_idx].grid(axis='y', linestyle='--', alpha=0.25)
        axes[row, col_idx].set_axisbelow(True)
        axes[row, col_idx].spines['top'].set_visible(False)
        axes[row, col_idx].spines['right'].set_visible(False)
        
        # 设置x轴标签
        axes[row, col_idx].set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    
    # 自定义图例（只在第一个子图中显示）
    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='高一', alpha=0.7),
        Patch(facecolor='#A23B72', edgecolor='#A23B72', label='高二', alpha=0.7),
        Patch(facecolor='#F18F01', edgecolor='#F18F01', label='高三', alpha=0.7),
        Line2D([0], [0], marker='D', color='w', label='均值', 
               markerfacecolor='#E74C3C', markeredgecolor='white', markersize=10)
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper right', 
                     frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 为总标题留出空间
    
    # 保存图片
    plt.savefig('../res/微观叙事句法箱线图比较.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("箱线图已保存为: '../res/微观叙事句法箱线图比较.png'")
    
    # 2. 变异系数热力图
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=120, constrained_layout=True)
    
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
    plt.savefig('../res/微观叙事句法变异系数热力图.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("变异系数热力图已保存为: '../res/微观叙事句法变异系数热力图.png'")

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 执行各项分析
    descriptive_statistics_syntactic(data, grade_groups)
    normality_test_syntactic(data, grade_groups)
    variance_homogeneity_test_syntactic(data, grade_groups)
    grade_comparison_analysis_syntactic(data, grade_groups)
    within_grade_analysis_syntactic(data, grade_groups)
    
    # 创建可视化
    create_syntactic_visualizations(data, grade_groups)
    
    print("\n" + "=" * 80)
    print("微观叙事句法分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

这个分析脚本实现了以下功能：

## 主要分析内容：

### 1. **年级间比较分析**
- **单因素方差分析（ANOVA）**：检验三个年级在各句法指标上是否存在显著性差异
- **Tukey HSD事后检验**：如果存在显著差异，进行两两比较
- **差异特征描述**：详细描述各年级间的差异模式

### 2. **年级内比较分析**
- **指标分组分析**：将11个指标分为5个组别进行分析
- **组内变异系数计算**：计算各组指标的平均变异系数
- **关键因素识别**：识别影响该阶段能力发展的关键指标组

### 3. **分析指标分组**
- **短语长度指标**：短语长度-SYNLE（69）、短语长度-SYNNP（70）
- **短语密度指标**：短语密度-DRNP（76）、短语密度-DRVP（77）、短语密度-DRPP（79）
- **句子长度指标**：句子长度-DESSL（06）
- **句子结构指标**：句子结构-DRPVAL（80）、句子结构-DRNEG（81）、句子结构-DRINF（83）
- **句法相似指标**：句法相似-SYNSTRUTa（74）、句法相似-SYNSTRUTt（75）

### 4. **可视化图表**
- **箱线图比较**：显示各年级在各句法指标上的分布差异
- **变异系数热力图**：直观显示各年级各指标的内部差异程度

### 5. **统计报告输出**
- 详细的描述性统计信息
- 假设检验结果
- 显著性差异分析
- 内部差异特征描述
- 关键因素识别

这个分析将全面回答统计要求中的问题：
- 三个年级在微观叙事句法各指标上是否存在显著性差异
- 各年级内部哪些指标组的差异程度最大
- 识别影响各阶段能力发展的关键因素

您可以直接运行这个脚本来获得完整的分析结果。

```python
# ... existing code ...

def descriptive_statistics_syntactic(data, grade_groups):
    """微观叙事句法描述性统计分析"""
    print("=" * 80)
    print("微观叙事句法描述性统计分析报告")
    print("=" * 80)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
    indicators = {
        '短语长度-SYNLE': '短语长度-SYNLE（69）',
        '短语长度-SYNNP': '短语长度-SYNNP（70）',
        '短语密度-DRNP': '短语密度-DRNP（76）',
        '短语密度-DRVP': '短语密度-DRVP（77）',
        '短语密度-DRPP': '短语密度-DRPP（79）',
        '句子长度-DESSL': '句子长度-DESSL（06）',
        '句子结构-DRPVAL': '句子结构-DRPVAL（80）',
        '句子结构-DRNEG': '句子结构-DRNEG（81）',
        '句子结构-DRINF': '句子结构-DRINF（83）',
        '句法相似-SYNSTRUTa': '句法相似-SYNSTRUTa（74）',
        '句法相似-SYNSTRUTt': '句法相似-SYNSTRUTt（75）'
    }
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(group_data)}")
        
        for name, col in indicators.items():
            mean_val = group_data[col].mean()
            std_val = group_data[col].std()
            print(f"  {name} - 均值：{mean_val:.3f}, 标准差：{std_val:.3f}")

def normality_test_syntactic(data, grade_groups):
    """正态性检验"""
    print("\n3. 正态性检验：")
    
    indicators = {
        '短语长度-SYNLE': '短语长度-SYNLE（69）',
        '短语长度-SYNNP': '短语长度-SYNNP（70）',
        '短语密度-DRNP': '短语密度-DRNP（76）',
        '短语密度-DRVP': '短语密度-DRVP（77）',
        '短语密度-DRPP': '短语密度-DRPP（79）',
        '句子长度-DESSL': '句子长度-DESSL（06）',
        '句子结构-DRPVAL': '句子结构-DRPVAL（80）',
        '句子结构-DRNEG': '句子结构-DRNEG（81）',
        '句子结构-DRINF': '句子结构-DRINF（83）',
        '句法相似-SYNSTRUTa': '句法相似-SYNSTRUTa（74）',
        '句法相似-SYNSTRUTt': '句法相似-SYNSTRUTt（75）'
    }
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}：")
        
        for name, col in indicators.items():
            stat, p_val = shapiro(group_data[col])
            print(f"  {name} - Shapiro-Wilk检验：W={stat:.4f}, p={p_val:.4f} {'(正态分布)' if p_val > 0.05 else '(非正态分布)'}")

def variance_homogeneity_test_syntactic(data, grade_groups):
    """方差齐性检验"""
    print("\n4. 方差齐性检验：")
    
    indicators = {
        '短语长度-SYNLE': '短语长度-SYNLE（69）',
        '短语长度-SYNNP': '短语长度-SYNNP（70）',
        '短语密度-DRNP': '短语密度-DRNP（76）',
        '短语密度-DRVP': '短语密度-DRVP（77）',
        '短语密度-DRPP': '短语密度-DRPP（79）',
        '句子长度-DESSL': '句子长度-DESSL（06）',
        '句子结构-DRPVAL': '句子结构-DRPVAL（80）',
        '句子结构-DRNEG': '句子结构-DRNEG（81）',
        '句子结构-DRINF': '句子结构-DRINF（83）',
        '句法相似-SYNSTRUTa': '句法相似-SYNSTRUTa（74）',
        '句法相似-SYNSTRUTt': '句法相似-SYNSTRUTt（75）'
    }
    
    for name, col in indicators.items():
        grade1_data = grade_groups['高一'][col]
        grade2_data = grade_groups['高二'][col]
        grade3_data = grade_groups['高三'][col]
        
        stat, p_val = levene(grade1_data, grade2_data, grade3_data)
        print(f"{name} - Levene检验：F={stat:.4f}, p={p_val:.4f} {'(方差齐性)' if p_val > 0.05 else '(方差非齐性)'}")

def grade_comparison_analysis_syntactic(data, grade_groups):
    """年级间比较分析"""
    print("\n5. 年级间比较分析：")
    
    indicators = {
        '短语长度-SYNLE': '短语长度-SYNLE（69）',
        '短语长度-SYNNP': '短语长度-SYNNP（70）',
        '短语密度-DRNP': '短语密度-DRNP（76）',
        '短语密度-DRVP': '短语密度-DRVP（77）',
        '短语密度-DRPP': '短语密度-DRPP（79）',
        '句子长度-DESSL': '句子长度-DESSL（06）',
        '句子结构-DRPVAL': '句子结构-DRPVAL（80）',
        '句子结构-DRNEG': '句子结构-DRNEG（81）',
        '句子结构-DRINF': '句子结构-DRINF（83）',
        '句法相似-SYNSTRUTa': '句法相似-SYNSTRUTa（74）',
        '句法相似-SYNSTRUTt': '句法相似-SYNSTRUTt（75）'
    }
    
    for name, col in indicators.items():
        print(f"\n{name}年级间比较：")
        
        grade1_data = grade_groups['高一'][col]
        grade2_data = grade_groups['高二'][col]
        grade3_data = grade_groups['高三'][col]
        
        # 单因素方差分析
        f_stat, p_val = f_oneway(grade1_data, grade2_data, grade3_data)
        print(f"  单因素方差分析：F={f_stat:.4f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            print("  结果：存在显著性差异")
            # Tukey HSD事后检验
            tukey_result = tukey_hsd(grade1_data, grade2_data, grade3_data)
            print("  Tukey HSD事后检验结果：")
            print(f"    高一 vs 高二：p={float(tukey_result.pvalue[0]):.4f}")
            print(f"    高一 vs 高三：p={float(tukey_result.pvalue[1]):.4f}")
            print(f"    高二 vs 高三：p={float(tukey_result.pvalue[2]):.4f}")
        else:
            print("  结果：无显著性差异")

def within_grade_analysis_syntactic(data, grade_groups):
    """年级内比较分析"""
    print("\n6. 年级内比较分析：")
    
    # 定义指标分组
    indicator_groups = {
        '短语长度指标': ['短语长度-SYNLE（69）', '短语长度-SYNNP（70）'],
        '短语密度指标': ['短语密度-DRNP（76）', '短语密度-DRVP（77）', '短语密度-DRPP（79）'],
        '句子长度指标': ['句子长度-DESSL（06）'],
        '句子结构指标': ['句子结构-DRPVAL（80）', '句子结构-DRNEG（81）', '句子结构-DRINF（83）'],
        '句法相似指标': ['句法相似-SYNSTRUTa（74）', '句法相似-SYNSTRUTt（75）']
    }
    
    for grade_name in ['高一', '高二', '高三']:
        group_data = grade_groups[grade_name]
        print(f"\n{grade_name}内部差异分析：")
        
        # 计算各组指标的变异系数
        group_cv_results = {}
        for group_name, cols in indicator_groups.items():
            group_cvs = []
            for col in cols:
                cv = group_data[col].std() / group_data[col].mean() * 100
                group_cvs.append(cv)
            # 取该组指标的平均变异系数
            group_cv_results[group_name] = np.mean(group_cvs)
        
        # 按变异系数排序
        sorted_group_cv = sorted(group_cv_results.items(), key=lambda x: x[1], reverse=True)
        
        print("  各组指标平均变异系数（从大到小排序）：")
        for group_name, cv in sorted_group_cv:
            print(f"    {group_name}：{cv:.2f}%")
        
        # 找出差异最大的指标组
        max_cv_group = sorted_group_cv[0]
        print(f"\n  差异程度最大的指标组：{max_cv_group[0]}（平均变异系数：{max_cv_group[1]:.2f}%）")
        print(f"  该指标组是影响{grade_name}阶段能力发展的关键因素")
        
        # 详细分析该组内各指标的变异系数
        print(f"\n  {max_cv_group[0]}内部各指标变异系数：")
        for col in indicator_groups[max_cv_group[0]]:
            cv = group_data[col].std() / group_data[col].mean() * 100
            print(f"    {col}：{cv:.2f}%")

def create_syntactic_visualizations(data, grade_groups):
    """创建微观叙事句法可视化图表"""
    print("\n7. 生成微观叙事句法可视化图表...")
    
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
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '短语长度-SYNLE': '短语长度-SYNLE（69）',
        '短语长度-SYNNP': '短语长度-SYNNP（70）',
        '短语密度-DRNP': '短语密度-DRNP（76）',
        '短语密度-DRVP': '短语密度-DRVP（77）',
        '短语密度-DRPP': '短语密度-DRPP（79）',
        '句子长度-DESSL': '句子长度-DESSL（06）',
        '句子结构-DRPVAL': '句子结构-DRPVAL（80）',
        '句子结构-DRNEG': '句子结构-DRNEG（81）',
        '句子结构-DRINF': '句子结构-DRINF（83）',
        '句法相似-SYNSTRUTa': '句法相似-SYNSTRUTa（74）',
        '句法相似-SYNSTRUTt': '句法相似-SYNSTRUTt（75）'
    }
    
    # 创建输出目录
    import os
    os.makedirs('../res', exist_ok=True)
    
    # 1. 箱线图比较
    fig, axes = plt.subplots(3, 4, figsize=(24, 18), dpi=120, constrained_layout=True)
    fig.suptitle('微观叙事句法各指标年级间分布对比', fontsize=24, fontweight='bold', color='#222', y=0.98)
    
    import pandas as pd
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    for i, (name, col) in enumerate(indicators.items()):
        row = i // 4
        col_idx = i % 4
        
        # 准备数据
        data_list = [grade_groups[grade][col] for grade in grade_names]
        
        # 构造DataFrame用于seaborn
        df = pd.DataFrame({
            '年级': np.repeat(grade_names, [len(x) for x in data_list]),
            '数值': np.concatenate(data_list)
        })
        
        # 创建小提琴图+箱线图组合
        sns.violinplot(
            x='年级', y='数值', data=df, ax=axes[row, col_idx],
            inner=None, palette=colors, linewidth=0, alpha=0.18
        )
        sns.boxplot(
            x='年级', y='数值', data=df, ax=axes[row, col_idx],
            width=0.25, palette=colors, boxprops=dict(alpha=0.7)
        )
        
        # 添加均值点
        means = df.groupby('年级')['数值'].mean()
        axes[row, col_idx].scatter(
            range(len(grade_names)), means, 
            color='#E74C3C', s=80, marker='D', 
            edgecolor='white', zorder=10, label='均值'
        )
        
        # 美化设置
        axes[row, col_idx].set_title(f'{name}', fontsize=16, fontweight='bold', color='#222')
        axes[row, col_idx].set_xlabel('年级', fontsize=14, fontweight='bold')
        axes[row, col_idx].set_ylabel('数值', fontsize=14, fontweight='bold')
        axes[row, col_idx].grid(axis='y', linestyle='--', alpha=0.25)
        axes[row, col_idx].set_axisbelow(True)
        axes[row, col_idx].spines['top'].set_visible(False)
        axes[row, col_idx].spines['right'].set_visible(False)
        
        # 设置x轴标签
        axes[row, col_idx].set_xticklabels(grade_names, fontsize=12, fontweight='bold')
    
    # 自定义图例（只在第一个子图中显示）
    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='高一', alpha=0.7),
        Patch(facecolor='#A23B72', edgecolor='#A23B72', label='高二', alpha=0.7),
        Patch(facecolor='#F18F01', edgecolor='#F18F01', label='高三', alpha=0.7),
        Line2D([0], [0], marker='D', color='w', label='均值', 
               markerfacecolor='#E74C3C', markeredgecolor='white', markersize=10)
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper right', 
                     frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # 为总标题留出空间
    
    # 保存图片
    plt.savefig('../res/微观叙事句法箱线图比较.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("箱线图已保存为: '../res/微观叙事句法箱线图比较.png'")
    
    # 2. 变异系数热力图
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=120, constrained_layout=True)
    
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
    plt.savefig('../res/微观叙事句法变异系数热力图.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print("变异系数热力图已保存为: '../res/微观叙事句法变异系数热力图.png'")

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 执行各项分析
    descriptive_statistics_syntactic(data, grade_groups)
    normality_test_syntactic(data, grade_groups)
    variance_homogeneity_test_syntactic(data, grade_groups)
    grade_comparison_analysis_syntactic(data, grade_groups)
    within_grade_analysis_syntactic(data, grade_groups)
    
    # 创建可视化
    create_syntactic_visualizations(data, grade_groups)
    
    print("\n" + "=" * 80)
    print("微观叙事句法分析完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

---

**User**

在执行代码之后输出结果 @分析结果.txt ，现在根据结果该如何回答 @tmp.txt 中对应的问题呢

---

**Cursor**

根据分析结果，我来帮您回答统计要求中的问题：

## ● 年级间比较分析

### 1. **短语长度指标**

#### **短语长度-SYNLE（69）**
- **显著性差异检验：** F=14.7893, p=0.0000 < 0.05
- **结论：** 三个年级在短语长度上存在显著性差异
- **差异特征：**
  - 高三（4.488）> 高二（3.706）> 高一（3.519）
  - 高一与高二无显著差异（p=0.5842）
  - 高一与高三、高二与高三均存在显著差异
  - 呈现随年级升高而增长的趋势

#### **短语长度-SYNNP（70）**
- **显著性差异检验：** F=31.5812, p=0.0000 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（0.775）> 高二（0.719）> 高一（0.628）
  - 所有年级间两两比较均存在显著差异
  - 呈现随年级升高而增长的趋势

### 2. **短语密度指标**

#### **短语密度-DRNP（76）**
- **显著性差异检验：** F=4.5822, p=0.0110 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高二（344.285）> 高三（335.952）> 高一（333.229）
  - 仅高一与高二存在显著差异（p=0.0110）
  - 高二达到最高水平

#### **短语密度-DRVP（77）**
- **显著性差异检验：** F=16.2537, p=0.0000 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高一（248.163）> 高二（232.045）> 高三（228.639）
  - 高一与高二、高一与高三均存在显著差异
  - 呈现随年级升高而下降的趋势

#### **短语密度-DRPP（79）**
- **显著性差异检验：** F=1.1257, p=0.3258 > 0.05
- **结论：** 无显著性差异
- **差异特征：** 各年级水平相对稳定

### 3. **句子长度指标**

#### **句子长度-DESSL（06）**
- **显著性差异检验：** F=14.8851, p=0.0000 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（16.230）> 高二（14.015）> 高一（13.961）
  - 高一与高二无显著差异（p=0.9930）
  - 高一与高三、高二与高三均存在显著差异
  - 高三学生句子长度显著增加

### 4. **句子结构指标**

#### **句子结构-DRPVAL（80）**
- **显著性差异检验：** F=1.0352, p=0.3564 > 0.05
- **结论：** 无显著性差异

#### **句子结构-DRNEG（81）**
- **显著性差异检验：** F=3.1498, p=0.0443 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（8.859）> 高一（7.260）> 高二（6.629）
  - 仅高二与高三存在显著差异（p=0.0409）

#### **句子结构-DRINF（83）**
- **显著性差异检验：** F=0.0643, p=0.9378 > 0.05
- **结论：** 无显著性差异

### 5. **句法相似指标**

#### **句法相似-SYNSTRUTa（74）**
- **显著性差异检验：** F=7.5321, p=0.0006 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高一（0.134）> 高二（0.127）> 高三（0.115）
  - 高一与高三存在显著差异（p=0.0005）
  - 呈现随年级升高而下降的趋势

#### **句法相似-SYNSTRUTt（75）**
- **显著性差异检验：** F=9.8797, p=0.0001 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高一（0.132）> 高二（0.124）> 高三（0.113）
  - 高一与高三存在显著差异（p=0.0000）
  - 呈现随年级升高而下降的趋势

## ★ 年级内比较分析

### **高一内部差异分析**
- **差异程度排序（从大到小）：**
  1. 句子结构指标：98.42%
  2. 句法相似指标：24.10%
  3. 短语长度指标：23.66%
  4. 句子长度指标：22.61%
  5. 短语密度指标：12.69%

**关键因素：** 句子结构指标（平均变异系数：98.42%）
- 该指标组内部差异最大，是影响高一阶段能力发展的关键因素
- 其中句子结构-DRPVAL（80）变异系数最高（152.20%）

### **高二内部差异分析**
- **差异程度排序（从大到小）：**
  1. 句子结构指标：90.30%
  2. 短语长度指标：28.67%
  3. 句法相似指标：27.14%
  4. 句子长度指标：24.93%
  5. 短语密度指标：14.16%

**关键因素：** 句子结构指标（平均变异系数：90.30%）
- 该指标组内部差异最大，是影响高二阶段能力发展的关键因素
- 其中句子结构-DRPVAL（80）变异系数最高（126.41%）

### **高三内部差异分析**
- **差异程度排序（从大到小）：**
  1. 句子结构指标：81.63%
  2. 句法相似指标：28.22%
  3. 短语长度指标：26.12%
  4. 句子长度指标：21.01%
  5. 短语密度指标：12.70%

**关键因素：** 句子结构指标（平均变异系数：81.63%）
- 该指标组内部差异最大，是影响高三阶段能力发展的关键因素
- 其中句子结构-DRPVAL（80）变异系数最高（119.67%）

## 总结

**年级间发展特征：**
1. 短语长度和句子长度呈现随年级升高而增长的趋势
2. 句法相似性呈现随年级升高而下降的趋势（说明句法结构更加多样化）
3. 部分指标（如短语密度-DRVP）呈现随年级升高而下降的趋势

**年级内关键因素：**
所有年级中，**句子结构指标**都是内部差异最大的指标组，是影响各阶段能力发展的关键因素，特别是句子结构-DRPVAL（80）指标在各年级都表现出最大的个体差异。

---

**User**

@grade_analyse_4.ipynb 现在我们来完成统计要求的第四部分 @tmp.txt ，请实现

---

**Cursor**

我来帮您实现统计要求的第四部分，分析宏观叙事语篇连贯性维度的年级间比较和年级内比较。


现在我来实现宏观叙事语篇连贯性维度的分析。根据数据，我们需要分析以下指标：

**局部连贯性指标：** 局部连贯-CRFAO1（29）、局部连贯-CRFSO1（30）、局部连贯-LSASS1(40)
**整体连贯性指标：** 整体连贯-CRFAOa（32）、整体连贯-CRFSOa（33）、整体连贯-LSASSp（42）、整体连贯-LSAPP1（44）、整体连贯-LSAGN（46）

```python:analyse_v2/grade_analyse_4.ipynb
# ... existing code ...

def descriptive_statistics_coherence(data, grade_groups):
    """宏观叙事语篇连贯性描述性统计分析"""
    print("=" * 80)
    print("宏观叙事语篇连贯性描述性统计分析报告")
    print("=" * 80)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(grade_groups[grade_name])}")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            mean_val = values.mean()
            std_val = values.std()
            print(f"  {indicator_name} - 均值：{mean_val:.3f}, 标准差：{std_val:.3f}")
    
    # 3. 正态性检验
    print("\n3. 正态性检验：")
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            stat, p_value = shapiro(values)
            normality = "正态分布" if p_value > 0.05 else "非正态分布"
            print(f"  {indicator_name} - Shapiro-Wilk检验：W={stat:.4f}, p={p_value:.4f} ({normality})")
    
    # 4. 方差齐性检验
    print("\n4. 方差齐性检验：")
    for indicator_name, indicator_col in indicators.items():
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        stat, p_value = levene(grade1_data, grade2_data, grade3_data)
        homogeneity = "方差齐性" if p_value > 0.05 else "方差非齐性"
        print(f"{indicator_name} - Levene检验：F={stat:.4f}, p={p_value:.4f} ({homogeneity})")
    
    # 5. 年级间比较分析
    print("\n5. 年级间比较分析：")
    for indicator_name, indicator_col in indicators.items():
        print(f"\n{indicator_name}年级间比较：")
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        # 单因素方差分析
        f_stat, p_value = f_oneway(grade1_data, grade2_data, grade3_data)
        print(f"  单因素方差分析：F={f_stat:.4f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("  结果：存在显著性差异")
            # Tukey HSD事后检验
            tukey_result = tukey_hsd(grade1_data, grade2_data, grade3_data)
            print("  Tukey HSD事后检验结果：")
            print(f"    高一 vs 高二：p={tukey_result.pvalue[0]:.4f}")
            print(f"    高一 vs 高三：p={tukey_result.pvalue[1]:.4f}")
            print(f"    高二 vs 高三：p={tukey_result.pvalue[2]:.4f}")
        else:
            print("  结果：无显著性差异")
    
    # 6. 年级内比较分析
    print("\n6. 年级内比较分析：")
    
    # 定义指标组
    local_coherence_indicators = ['局部连贯-CRFAO1（29）', '局部连贯-CRFSO1（30）', '局部连贯-LSASS1(40)']
    global_coherence_indicators = ['整体连贯-CRFAOa（32）', '整体连贯-CRFSOa（33）', '整体连贯-LSASSp（42）', 
                                  '整体连贯-LSAPP1（44）', '整体连贯-LSAGN（46）']
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}内部差异分析：")
        
        # 计算局部连贯性指标的平均变异系数
        local_cvs = []
        for indicator in local_coherence_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            local_cvs.append(cv)
        avg_local_cv = np.mean(local_cvs)
        
        # 计算整体连贯性指标的平均变异系数
        global_cvs = []
        for indicator in global_coherence_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            global_cvs.append(cv)
        avg_global_cv = np.mean(global_cvs)
        
        # 比较差异程度
        if avg_local_cv > avg_global_cv:
            max_diff_group = "局部连贯性指标"
            max_cv = avg_local_cv
            min_diff_group = "整体连贯性指标"
            min_cv = avg_global_cv
        else:
            max_diff_group = "整体连贯性指标"
            max_cv = avg_global_cv
            min_diff_group = "局部连贯性指标"
            min_cv = avg_local_cv
        
        print(f"  各组指标平均变异系数（从大到小排序）：")
        print(f"    {max_diff_group}：{max_cv:.2f}%")
        print(f"    {min_diff_group}：{min_cv:.2f}%")
        print(f"  差异程度最大的指标组：{max_diff_group}（平均变异系数：{max_cv:.2f}%）")
        print(f"  该指标组是影响{grade_name}阶段能力发展的关键因素")
        
        # 显示各指标的具体变异系数
        print(f"  {max_diff_group}内部各指标变异系数：")
        if max_diff_group == "局部连贯性指标":
            for i, indicator in enumerate(local_coherence_indicators):
                values = grade_groups[grade_name][indicator]
                cv = (values.std() / values.mean()) * 100
                print(f"    {indicator}：{cv:.2f}%")
        else:
            for i, indicator in enumerate(global_coherence_indicators):
                values = grade_groups[grade_name][indicator]
                cv = (values.std() / values.mean()) * 100
                print(f"    {indicator}：{cv:.2f}%")
    
    print("\n" + "=" * 80)
    print("分析结论")
    print("=" * 80)

def create_coherence_visualizations(data, grade_groups):
    """创建宏观叙事语篇连贯性可视化图表"""
    print("生成宏观叙事语篇连贯性可视化图表...")
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('宏观叙事语篇连贯性各指标年级间比较', fontsize=20, fontweight='bold')
    
    # 绘制箱线图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 准备数据
        data_list = [grade_groups[grade][indicator_col] for grade in grade_names]
        
        # 绘制箱线图
        bp = ax.boxplot(data_list, patch_artist=True, 
                       boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                       medianprops=dict(color='#E74C3C', linewidth=2),
                       whiskerprops=dict(color='#2C3E50', linewidth=1.5),
                       capprops=dict(color='#2C3E50', linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=4))
        
        # 设置颜色
        for patch in bp['boxes']:
            patch.set_facecolor(colors[i % len(colors)])
        
        ax.set_title(indicator_name, fontsize=14, fontweight='bold')
        ax.set_xticklabels(grade_names, fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../res/宏观叙事语篇连贯性箱线图比较.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建变异系数热力图
    print("生成变异系数热力图...")
    
    # 计算各年级各指标的变异系数
    cv_data = []
    for grade_name in grade_names:
        grade_cvs = []
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            cv = (values.std() / values.mean()) * 100
            grade_cvs.append(cv)
        cv_data.append(grade_cvs)
    
    cv_df = pd.DataFrame(cv_data, index=grade_names, columns=list(indicators.keys()))
    
    # 绘制热力图
    plt.figure(figsize=(16, 8))
    sns.heatmap(cv_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': '变异系数 (%)'}, 
                linewidths=0.5, linecolor='white')
    plt.title('各年级语篇连贯性指标变异系数热力图', fontsize=16, fontweight='bold')
    plt.xlabel('指标', fontsize=14)
    plt.ylabel('年级', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../res/宏观叙事语篇连贯性变异系数热力图.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 描述性统计分析
    descriptive_statistics_coherence(data, grade_groups)
    
    # 创建可视化图表
    create_coherence_visualizations(data, grade_groups)
    
    # 保存分析结果到文件
    import sys
    from io import StringIO
    
    # 重定向输出到文件
    original_stdout = sys.stdout
    result_file = StringIO()
    sys.stdout = result_file
    
    # 重新运行分析
    descriptive_statistics_coherence(data, grade_groups)
    
    # 恢复输出
    sys.stdout = original_stdout
    
    # 保存结果到文件
    with open('../res/res_v2/宏观叙事结构-语篇连贯性/分析结果.txt', 'w', encoding='utf-8') as f:
        f.write(result_file.getvalue())
    
    print("分析完成！结果已保存到 '../res/res_v2/宏观叙事结构-语篇连贯性/分析结果.txt'")

if __name__ == "__main__":
    main()
```

这个实现包含了：

1. **描述性统计分析**：计算各年级在各指标上的均值、标准差
2. **正态性检验**：使用Shapiro-Wilk检验判断数据分布
3. **方差齐性检验**：使用Levene检验判断方差是否相等
4. **年级间比较**：使用单因素方差分析和Tukey HSD事后检验
5. **年级内比较**：计算局部连贯性和整体连贯性指标组的变异系数，找出差异最大的指标组
6. **可视化图表**：创建箱线图和变异系数热力图
7. **结果保存**：将分析结果保存到文件

运行这个代码后，您将得到完整的宏观叙事语篇连贯性分析报告，包括年级间比较和年级内比较的结果。

```python
# ... existing code ...

def descriptive_statistics_coherence(data, grade_groups):
    """宏观叙事语篇连贯性描述性统计分析"""
    print("=" * 80)
    print("宏观叙事语篇连贯性描述性统计分析报告")
    print("=" * 80)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(grade_groups[grade_name])}")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            mean_val = values.mean()
            std_val = values.std()
            print(f"  {indicator_name} - 均值：{mean_val:.3f}, 标准差：{std_val:.3f}")
    
    # 3. 正态性检验
    print("\n3. 正态性检验：")
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            stat, p_value = shapiro(values)
            normality = "正态分布" if p_value > 0.05 else "非正态分布"
            print(f"  {indicator_name} - Shapiro-Wilk检验：W={stat:.4f}, p={p_value:.4f} ({normality})")
    
    # 4. 方差齐性检验
    print("\n4. 方差齐性检验：")
    for indicator_name, indicator_col in indicators.items():
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        stat, p_value = levene(grade1_data, grade2_data, grade3_data)
        homogeneity = "方差齐性" if p_value > 0.05 else "方差非齐性"
        print(f"{indicator_name} - Levene检验：F={stat:.4f}, p={p_value:.4f} ({homogeneity})")
    
    # 5. 年级间比较分析
    print("\n5. 年级间比较分析：")
    for indicator_name, indicator_col in indicators.items():
        print(f"\n{indicator_name}年级间比较：")
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        # 单因素方差分析
        f_stat, p_value = f_oneway(grade1_data, grade2_data, grade3_data)
        print(f"  单因素方差分析：F={f_stat:.4f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("  结果：存在显著性差异")
            # Tukey HSD事后检验
            tukey_result = tukey_hsd(grade1_data, grade2_data, grade3_data)
            print("  Tukey HSD事后检验结果：")
            print(f"    高一 vs 高二：p={tukey_result.pvalue[0]:.4f}")
            print(f"    高一 vs 高三：p={tukey_result.pvalue[1]:.4f}")
            print(f"    高二 vs 高三：p={tukey_result.pvalue[2]:.4f}")
        else:
            print("  结果：无显著性差异")
    
    # 6. 年级内比较分析
    print("\n6. 年级内比较分析：")
    
    # 定义指标组
    local_coherence_indicators = ['局部连贯-CRFAO1（29）', '局部连贯-CRFSO1（30）', '局部连贯-LSASS1(40)']
    global_coherence_indicators = ['整体连贯-CRFAOa（32）', '整体连贯-CRFSOa（33）', '整体连贯-LSASSp（42）', 
                                  '整体连贯-LSAPP1（44）', '整体连贯-LSAGN（46）']
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}内部差异分析：")
        
        # 计算局部连贯性指标的平均变异系数
        local_cvs = []
        for indicator in local_coherence_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            local_cvs.append(cv)
        avg_local_cv = np.mean(local_cvs)
        
        # 计算整体连贯性指标的平均变异系数
        global_cvs = []
        for indicator in global_coherence_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            global_cvs.append(cv)
        avg_global_cv = np.mean(global_cvs)
        
        # 比较差异程度
        if avg_local_cv > avg_global_cv:
            max_diff_group = "局部连贯性指标"
            max_cv = avg_local_cv
            min_diff_group = "整体连贯性指标"
            min_cv = avg_global_cv
        else:
            max_diff_group = "整体连贯性指标"
            max_cv = avg_global_cv
            min_diff_group = "局部连贯性指标"
            min_cv = avg_local_cv
        
        print(f"  各组指标平均变异系数（从大到小排序）：")
        print(f"    {max_diff_group}：{max_cv:.2f}%")
        print(f"    {min_diff_group}：{min_cv:.2f}%")
        print(f"  差异程度最大的指标组：{max_diff_group}（平均变异系数：{max_cv:.2f}%）")
        print(f"  该指标组是影响{grade_name}阶段能力发展的关键因素")
        
        # 显示各指标的具体变异系数
        print(f"  {max_diff_group}内部各指标变异系数：")
        if max_diff_group == "局部连贯性指标":
            for i, indicator in enumerate(local_coherence_indicators):
                values = grade_groups[grade_name][indicator]
                cv = (values.std() / values.mean()) * 100
                print(f"    {indicator}：{cv:.2f}%")
        else:
            for i, indicator in enumerate(global_coherence_indicators):
                values = grade_groups[grade_name][indicator]
                cv = (values.std() / values.mean()) * 100
                print(f"    {indicator}：{cv:.2f}%")
    
    print("\n" + "=" * 80)
    print("分析结论")
    print("=" * 80)

def create_coherence_visualizations(data, grade_groups):
    """创建宏观叙事语篇连贯性可视化图表"""
    print("生成宏观叙事语篇连贯性可视化图表...")
    
    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('宏观叙事语篇连贯性各指标年级间比较', fontsize=20, fontweight='bold')
    
    # 绘制箱线图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 准备数据
        data_list = [grade_groups[grade][indicator_col] for grade in grade_names]
        
        # 绘制箱线图
        bp = ax.boxplot(data_list, patch_artist=True, 
                       boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                       medianprops=dict(color='#E74C3C', linewidth=2),
                       whiskerprops=dict(color='#2C3E50', linewidth=1.5),
                       capprops=dict(color='#2C3E50', linewidth=1.5),
                       flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=4))
        
        # 设置颜色
        for patch in bp['boxes']:
            patch.set_facecolor(colors[i % len(colors)])
        
        ax.set_title(indicator_name, fontsize=14, fontweight='bold')
        ax.set_xticklabels(grade_names, fontsize=12)
        ax.set_ylabel('数值', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../res/宏观叙事语篇连贯性箱线图比较.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建变异系数热力图
    print("生成变异系数热力图...")
    
    # 计算各年级各指标的变异系数
    cv_data = []
    for grade_name in grade_names:
        grade_cvs = []
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            cv = (values.std() / values.mean()) * 100
            grade_cvs.append(cv)
        cv_data.append(grade_cvs)
    
    cv_df = pd.DataFrame(cv_data, index=grade_names, columns=list(indicators.keys()))
    
    # 绘制热力图
    plt.figure(figsize=(16, 8))
    sns.heatmap(cv_df, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': '变异系数 (%)'}, 
                linewidths=0.5, linecolor='white')
    plt.title('各年级语篇连贯性指标变异系数热力图', fontsize=16, fontweight='bold')
    plt.xlabel('指标', fontsize=14)
    plt.ylabel('年级', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../res/宏观叙事语篇连贯性变异系数热力图.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 描述性统计分析
    descriptive_statistics_coherence(data, grade_groups)
    
    # 创建可视化图表
    create_coherence_visualizations(data, grade_groups)
    
    # 保存分析结果到文件
    import sys
    from io import StringIO
    
    # 重定向输出到文件
    original_stdout = sys.stdout
    result_file = StringIO()
    sys.stdout = result_file
    
    # 重新运行分析
    descriptive_statistics_coherence(data, grade_groups)
    
    # 恢复输出
    sys.stdout = original_stdout
    
    # 保存结果到文件
    with open('../res/res_v2/宏观叙事结构-语篇连贯性/分析结果.txt', 'w', encoding='utf-8') as f:
        f.write(result_file.getvalue())
    
    print("分析完成！结果已保存到 '../res/res_v2/宏观叙事结构-语篇连贯性/分析结果.txt'")

if __name__ == "__main__":
    main()
```

---

**User**

好的 现在的代码生成的箱线图和变异系数热力图看上去不够美观高级，请你修改可视化图表的函数使其生成的图高级美观，同时每中类型的图使用单独的函数实现

---

**Cursor**

我来帮您修改可视化图表函数，使其生成更加高级美观的图表，并将不同类型的图表分别实现为独立的函数。


现在我来修改可视化图表函数，创建更加高级美观的图表，并将不同类型的图表分别实现为独立的函数：

```python:analyse_v2/grade_analyse_4.ipynb
# ... existing code ...

def create_coherence_boxplots(data, grade_groups, save_path='../res/宏观叙事语篇连贯性箱线图比较.png'):
    """创建宏观叙事语篇连贯性箱线图比较 - 高级美观版本"""
    print("生成宏观叙事语篇连贯性箱线图...")
    
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

    # 设置颜色主题 - 使用渐变色彩
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(24, 14))
    fig.suptitle('宏观叙事语篇连贯性各指标年级间比较分析', fontsize=24, fontweight='bold', y=0.98)
    
    # 添加背景色
    fig.patch.set_facecolor('#f8f9fa')
    
    # 绘制箱线图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 设置子图背景
        ax.set_facecolor('white')
        
        # 准备数据
        data_list = [grade_groups[grade][indicator_col] for grade in grade_names]
        
        # 绘制箱线图
        bp = ax.boxplot(data_list, patch_artist=True, 
                       boxprops=dict(facecolor='white', alpha=0.9, edgecolor='#2C3E50', linewidth=2),
                       medianprops=dict(color='#E74C3C', linewidth=3),
                       whiskerprops=dict(color='#2C3E50', linewidth=2),
                       capprops=dict(color='#2C3E50', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=6, alpha=0.7))
        
        # 设置颜色
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[j])
            patch.set_alpha(0.8)
        
        # 添加均值点
        for j, data_group in enumerate(data_list):
            mean_val = np.mean(data_group)
            ax.scatter(j+1, mean_val, color='white', s=100, zorder=5, edgecolors='#2C3E50', linewidth=2)
            ax.scatter(j+1, mean_val, color='#E74C3C', s=60, zorder=6)
        
        # 设置标题和标签
        ax.set_title(indicator_name, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels(grade_names, fontsize=14, fontweight='bold')
        ax.set_ylabel('数值', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置y轴范围，留出一些空间
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#2C3E50')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"箱线图已保存到: {save_path}")

def create_coherence_heatmap(data, grade_groups, save_path='../res/宏观叙事语篇连贯性变异系数热力图.png'):
    """创建宏观叙事语篇连贯性变异系数热力图 - 高级美观版本"""
    print("生成宏观叙事语篇连贯性变异系数热力图...")
    
    # 设置全局风格
    sns.set_theme(style="white", font="SimHei", rc={
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
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    grade_names = ['高一', '高二', '高三']
    
    # 计算各年级各指标的变异系数
    cv_data = []
    for grade_name in grade_names:
        grade_cvs = []
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            cv = (values.std() / values.mean()) * 100
            grade_cvs.append(cv)
        cv_data.append(grade_cvs)
    
    cv_df = pd.DataFrame(cv_data, index=grade_names, columns=list(indicators.keys()))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 设置背景色
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('white')
    
    # 创建自定义颜色映射
    colors = ['#E8F4FD', '#B3D9F2', '#66B3FF', '#0066CC', '#004080']
    n_bins = len(colors)
    
    # 绘制热力图
    sns.heatmap(cv_df, annot=True, fmt='.1f', 
                cmap=sns.color_palette(colors, n_colors=n_bins),
                cbar_kws={'label': '变异系数 (%)', 'shrink': 0.8}, 
                linewidths=1, linecolor='white',
                square=False, ax=ax,
                annot_kws={'size': 12, 'weight': 'bold'})
    
    # 设置标题和标签
    ax.set_title('宏观叙事语篇连贯性指标变异系数热力图', fontsize=22, fontweight='bold', pad=30)
    ax.set_xlabel('指标', fontsize=16, fontweight='bold', labelpad=20)
    ax.set_ylabel('年级', fontsize=16, fontweight='bold', labelpad=20)
    
    # 调整x轴标签
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"热力图已保存到: {save_path}")

def create_coherence_density_plots(data, grade_groups, save_path='../res/宏观叙事语篇连贯性密度图.png'):
    """创建宏观叙事语篇连贯性密度图 - 高级美观版本"""
    print("生成宏观叙事语篇连贯性密度图...")
    
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(24, 14))
    fig.suptitle('宏观叙事语篇连贯性各指标分布密度图', fontsize=22, fontweight='bold', y=0.98)
    
    # 添加背景色
    fig.patch.set_facecolor('#f8f9fa')
    
    # 绘制密度图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 设置子图背景
        ax.set_facecolor('white')
        
        # 绘制每个年级的密度图
        for j, grade_name in enumerate(grade_names):
            values = grade_groups[grade_name][indicator_col]
            sns.kdeplot(data=values, ax=ax, color=colors[j], linewidth=3, 
                       label=grade_name, alpha=0.8)
            
            # 添加均值线
            mean_val = values.mean()
            ax.axvline(mean_val, color=colors[j], linestyle='--', linewidth=2, alpha=0.7)
        
        # 设置标题和标签
        ax.set_title(indicator_name, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('数值', fontsize=12, fontweight='bold')
        ax.set_ylabel('密度', fontsize=12, fontweight='bold')
        
        # 添加图例
        ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#2C3E50')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"密度图已保存到: {save_path}")

def create_coherence_summary_chart(data, grade_groups, save_path='../res/宏观叙事语篇连贯性汇总图.png'):
    """创建宏观叙事语篇连贯性汇总图 - 高级美观版本"""
    print("生成宏观叙事语篇连贯性汇总图...")
    
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标组
    local_indicators = ['局部连贯-CRFAO1（29）', '局部连贯-CRFSO1（30）', '局部连贯-LSASS1(40)']
    global_indicators = ['整体连贯-CRFAOa（32）', '整体连贯-CRFSOa（33）', '整体连贯-LSASSp（42）', 
                        '整体连贯-LSAPP1（44）', '整体连贯-LSAGN（46）']
    
    # 计算各年级各指标组的平均变异系数
    local_cvs = []
    global_cvs = []
    
    for grade_name in grade_names:
        # 局部连贯性平均变异系数
        local_cv_list = []
        for indicator in local_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            local_cv_list.append(cv)
        local_cvs.append(np.mean(local_cv_list))
        
        # 整体连贯性平均变异系数
        global_cv_list = []
        for indicator in global_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            global_cv_list.append(cv)
        global_cvs.append(np.mean(global_cv_list))
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('宏观叙事语篇连贯性指标组变异系数对比', fontsize=22, fontweight='bold', y=0.95)
    
    # 添加背景色
    fig.patch.set_facecolor('#f8f9fa')
    
    # 左图：柱状图对比
    ax1.set_facecolor('white')
    x = np.arange(len(grade_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, local_cvs, width, label='局部连贯性指标', 
                    color=colors[0], alpha=0.8, edgecolor='#2C3E50', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, global_cvs, width, label='整体连贯性指标', 
                    color=colors[1], alpha=0.8, edgecolor='#2C3E50', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('年级', fontsize=14, fontweight='bold')
    ax1.set_ylabel('平均变异系数 (%)', fontsize=14, fontweight='bold')
    ax1.set_title('各年级指标组变异系数对比', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(grade_names, fontweight='bold')
    ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 右图：雷达图
    ax2.set_facecolor('white')
    
    # 准备雷达图数据
    categories = grade_names
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形
    
    # 准备数据
    local_cvs_radar = local_cvs + [local_cvs[0]]
    global_cvs_radar = global_cvs + [global_cvs[0]]
    
    # 绘制雷达图
    ax2.plot(angles, local_cvs_radar, 'o-', linewidth=3, label='局部连贯性指标', 
             color=colors[0], alpha=0.8)
    ax2.fill(angles, local_cvs_radar, alpha=0.25, color=colors[0])
    
    ax2.plot(angles, global_cvs_radar, 'o-', linewidth=3, label='整体连贯性指标', 
             color=colors[1], alpha=0.8)
    ax2.fill(angles, global_cvs_radar, alpha=0.25, color=colors[1])
    
    # 设置雷达图属性
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontweight='bold')
    ax2.set_ylim(0, max(max(local_cvs), max(global_cvs)) * 1.2)
    ax2.set_title('变异系数雷达图对比', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 添加边框
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#2C3E50')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"汇总图已保存到: {save_path}")

def create_coherence_visualizations(data, grade_groups):
    """创建宏观叙事语篇连贯性可视化图表 - 主函数"""
    print("开始生成宏观叙事语篇连贯性可视化图表...")
    
    # 创建各种类型的图表
    create_coherence_boxplots(data, grade_groups)
    create_coherence_heatmap(data, grade_groups)
    create_coherence_density_plots(data, grade_groups)
    create_coherence_summary_chart(data, grade_groups)
    
    print("所有可视化图表生成完成！")

# ... existing code ...

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 描述性统计分析
    descriptive_statistics_coherence(data, grade_groups)
    
    # 创建可视化图表
    create_coherence_visualizations(data, grade_groups)
    
    # 保存分析结果到文件
    import sys
    from io import StringIO
    
    # 重定向输出到文件
    original_stdout = sys.stdout
    result_file = StringIO()
    sys.stdout = result_file
    
    # 重新运行分析
    descriptive_statistics_coherence(data, grade_groups)
    
    # 恢复输出
    sys.stdout = original_stdout
    
    # 保存结果到文件
    with open('../res/res_v2/宏观叙事结构-语篇连贯性/分析结果.txt', 'w', encoding='utf-8') as f:
        f.write(result_file.getvalue())
    
    print("分析完成！结果已保存到 '../res/res_v2/宏观叙事结构-语篇连贯性/分析结果.txt'")

if __name__ == "__main__":
    main()
```

这个修改后的代码包含了以下改进：

## 主要改进点：

### 1. **分离的函数设计**
- `create_coherence_boxplots()` - 专门负责箱线图
- `create_coherence_heatmap()` - 专门负责热力图
- `create_coherence_density_plots()` - 专门负责密度图
- `create_coherence_summary_chart()` - 专门负责汇总图

### 2. **高级美观的设计元素**
- **渐变色彩方案**：使用专业的配色方案
- **高质量背景**：设置白色背景和网格
- **增强的边框**：加粗的边框线条
- **均值标记**：在箱线图中添加均值点
- **阴影效果**：图例添加阴影和圆角

### 3. **专业的图表类型**
- **箱线图**：显示分布特征和异常值
- **热力图**：直观显示变异系数差异
- **密度图**：展示数据分布形状
- **汇总图**：包含柱状图和雷达图对比

### 4. **增强的可读性**
- **清晰的标题**：大字体、粗体标题
- **优化的标签**：旋转的x轴标签
- **数值标注**：在图表上显示具体数值
- **图例优化**：美观的图例设计

### 5. **高质量输出**
- **高分辨率**：300 DPI输出
- **合适的尺寸**：优化的图表尺寸
- **自动保存**：自动保存到指定路径

这样的设计使得生成的图表更加专业、美观，适合用于学术报告或论文发表。

```python
# ... existing code ...

def create_coherence_boxplots(data, grade_groups, save_path='../res/宏观叙事语篇连贯性箱线图比较.png'):
    """创建宏观叙事语篇连贯性箱线图比较 - 高级美观版本"""
    print("生成宏观叙事语篇连贯性箱线图...")
    
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

    # 设置颜色主题 - 使用渐变色彩
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(24, 14))
    fig.suptitle('宏观叙事语篇连贯性各指标年级间比较分析', fontsize=24, fontweight='bold', y=0.98)
    
    # 添加背景色
    fig.patch.set_facecolor('#f8f9fa')
    
    # 绘制箱线图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 设置子图背景
        ax.set_facecolor('white')
        
        # 准备数据
        data_list = [grade_groups[grade][indicator_col] for grade in grade_names]
        
        # 绘制箱线图
        bp = ax.boxplot(data_list, patch_artist=True, 
                       boxprops=dict(facecolor='white', alpha=0.9, edgecolor='#2C3E50', linewidth=2),
                       medianprops=dict(color='#E74C3C', linewidth=3),
                       whiskerprops=dict(color='#2C3E50', linewidth=2),
                       capprops=dict(color='#2C3E50', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=6, alpha=0.7))
        
        # 设置颜色
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[j])
            patch.set_alpha(0.8)
        
        # 添加均值点
        for j, data_group in enumerate(data_list):
            mean_val = np.mean(data_group)
            ax.scatter(j+1, mean_val, color='white', s=100, zorder=5, edgecolors='#2C3E50', linewidth=2)
            ax.scatter(j+1, mean_val, color='#E74C3C', s=60, zorder=6)
        
        # 设置标题和标签
        ax.set_title(indicator_name, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticklabels(grade_names, fontsize=14, fontweight='bold')
        ax.set_ylabel('数值', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置y轴范围，留出一些空间
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#2C3E50')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"箱线图已保存到: {save_path}")

def create_coherence_heatmap(data, grade_groups, save_path='../res/宏观叙事语篇连贯性变异系数热力图.png'):
    """创建宏观叙事语篇连贯性变异系数热力图 - 高级美观版本"""
    print("生成宏观叙事语篇连贯性变异系数热力图...")
    
    # 设置全局风格
    sns.set_theme(style="white", font="SimHei", rc={
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
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    grade_names = ['高一', '高二', '高三']
    
    # 计算各年级各指标的变异系数
    cv_data = []
    for grade_name in grade_names:
        grade_cvs = []
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            cv = (values.std() / values.mean()) * 100
            grade_cvs.append(cv)
        cv_data.append(grade_cvs)
    
    cv_df = pd.DataFrame(cv_data, index=grade_names, columns=list(indicators.keys()))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # 设置背景色
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('white')
    
    # 创建自定义颜色映射
    colors = ['#E8F4FD', '#B3D9F2', '#66B3FF', '#0066CC', '#004080']
    n_bins = len(colors)
    
    # 绘制热力图
    sns.heatmap(cv_df, annot=True, fmt='.1f', 
                cmap=sns.color_palette(colors, n_colors=n_bins),
                cbar_kws={'label': '变异系数 (%)', 'shrink': 0.8}, 
                linewidths=1, linecolor='white',
                square=False, ax=ax,
                annot_kws={'size': 12, 'weight': 'bold'})
    
    # 设置标题和标签
    ax.set_title('宏观叙事语篇连贯性指标变异系数热力图', fontsize=22, fontweight='bold', pad=30)
    ax.set_xlabel('指标', fontsize=16, fontweight='bold', labelpad=20)
    ax.set_ylabel('年级', fontsize=16, fontweight='bold', labelpad=20)
    
    # 调整x轴标签
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"热力图已保存到: {save_path}")

def create_coherence_density_plots(data, grade_groups, save_path='../res/宏观叙事语篇连贯性密度图.png'):
    """创建宏观叙事语篇连贯性密度图 - 高级美观版本"""
    print("生成宏观叙事语篇连贯性密度图...")
    
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(24, 14))
    fig.suptitle('宏观叙事语篇连贯性各指标分布密度图', fontsize=22, fontweight='bold', y=0.98)
    
    # 添加背景色
    fig.patch.set_facecolor('#f8f9fa')
    
    # 绘制密度图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 设置子图背景
        ax.set_facecolor('white')
        
        # 绘制每个年级的密度图
        for j, grade_name in enumerate(grade_names):
            values = grade_groups[grade_name][indicator_col]
            sns.kdeplot(data=values, ax=ax, color=colors[j], linewidth=3, 
                       label=grade_name, alpha=0.8)
            
            # 添加均值线
            mean_val = values.mean()
            ax.axvline(mean_val, color=colors[j], linestyle='--', linewidth=2, alpha=0.7)
        
        # 设置标题和标签
        ax.set_title(indicator_name, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('数值', fontsize=12, fontweight='bold')
        ax.set_ylabel('密度', fontsize=12, fontweight='bold')
        
        # 添加图例
        ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#2C3E50')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"密度图已保存到: {save_path}")

def create_coherence_summary_chart(data, grade_groups, save_path='../res/宏观叙事语篇连贯性汇总图.png'):
    """创建宏观叙事语篇连贯性汇总图 - 高级美观版本"""
    print("生成宏观叙事语篇连贯性汇总图...")
    
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标组
    local_indicators = ['局部连贯-CRFAO1（29）', '局部连贯-CRFSO1（30）', '局部连贯-LSASS1(40)']
    global_indicators = ['整体连贯-CRFAOa（32）', '整体连贯-CRFSOa（33）', '整体连贯-LSASSp（42）', 
                        '整体连贯-LSAPP1（44）', '整体连贯-LSAGN（46）']
    
    # 计算各年级各指标组的平均变异系数
    local_cvs = []
    global_cvs = []
    
    for grade_name in grade_names:
        # 局部连贯性平均变异系数
        local_cv_list = []
        for indicator in local_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            local_cv_list.append(cv)
        local_cvs.append(np.mean(local_cv_list))
        
        # 整体连贯性平均变异系数
        global_cv_list = []
        for indicator in global_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            global_cv_list.append(cv)
        global_cvs.append(np.mean(global_cv_list))
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('宏观叙事语篇连贯性指标组变异系数对比', fontsize=22, fontweight='bold', y=0.95)
    
    # 添加背景色
    fig.patch.set_facecolor('#f8f9fa')
    
    # 左图：柱状图对比
    ax1.set_facecolor('white')
    x = np.arange(len(grade_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, local_cvs, width, label='局部连贯性指标', 
                    color=colors[0], alpha=0.8, edgecolor='#2C3E50', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, global_cvs, width, label='整体连贯性指标', 
                    color=colors[1], alpha=0.8, edgecolor='#2C3E50', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('年级', fontsize=14, fontweight='bold')
    ax1.set_ylabel('平均变异系数 (%)', fontsize=14, fontweight='bold')
    ax1.set_title('各年级指标组变异系数对比', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(grade_names, fontweight='bold')
    ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 右图：雷达图
    ax2.set_facecolor('white')
    
    # 准备雷达图数据
    categories = grade_names
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形
    
    # 准备数据
    local_cvs_radar = local_cvs + [local_cvs[0]]
    global_cvs_radar = global_cvs + [global_cvs[0]]
    
    # 绘制雷达图
    ax2.plot(angles, local_cvs_radar, 'o-', linewidth=3, label='局部连贯性指标', 
             color=colors[0], alpha=0.8)
    ax2.fill(angles, local_cvs_radar, alpha=0.25, color=colors[0])
    
    ax2.plot(angles, global_cvs_radar, 'o-', linewidth=3, label='整体连贯性指标', 
             color=colors[1], alpha=0.8)
    ax2.fill(angles, global_cvs_radar, alpha=0.25, color=colors[1])
    
    # 设置雷达图属性
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontweight='bold')
    ax2.set_ylim(0, max(max(local_cvs), max(global_cvs)) * 1.2)
    ax2.set_title('变异系数雷达图对比', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 添加边框
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('#2C3E50')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"汇总图已保存到: {save_path}")

def create_coherence_visualizations(data, grade_groups):
    """创建宏观叙事语篇连贯性可视化图表 - 主函数"""
    print("开始生成宏观叙事语篇连贯性可视化图表...")
    
    # 创建各种类型的图表
    create_coherence_boxplots(data, grade_groups)
    create_coherence_heatmap(data, grade_groups)
    create_coherence_density_plots(data, grade_groups)
    create_coherence_summary_chart(data, grade_groups)
    
    print("所有可视化图表生成完成！")

# ... existing code ...

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 描述性统计分析
    descriptive_statistics_coherence(data, grade_groups)
    
    # 创建可视化图表
    create_coherence_visualizations(data, grade_groups)
    
    # 保存分析结果到文件
    import sys
    from io import StringIO
    
    # 重定向输出到文件
    original_stdout = sys.stdout
    result_file = StringIO()
    sys.stdout = result_file
    
    # 重新运行分析
    descriptive_statistics_coherence(data, grade_groups)
    
    # 恢复输出
    sys.stdout = original_stdout
    
    # 保存结果到文件
    with open('../res/res_v2/宏观叙事结构-语篇连贯性/分析结果.txt', 'w', encoding='utf-8') as f:
        f.write(result_file.getvalue())
    
    print("分析完成！结果已保存到 '../res/res_v2/宏观叙事结构-语篇连贯性/分析结果.txt'")

if __name__ == "__main__":
    main()
```

---

**User**

@grade_analyse_4.ipynb 在箱线图的绘制中我希望采用@grede_analyse_1_untils.py 中create_boxplot_comparison逻辑和样式，密度图和热力度也采用   @@grede_analyse_1_untils.py 中的create_density_distributioncreate_density_distribution逻辑和样式

---

**Cursor**

我来帮您修改 `grade_analyse_4.ipynb` 中的可视化函数，采用 `grede_analyse_1_untils.py` 中的逻辑和样式。


现在我来修改 `grade_analyse_4.ipynb` 中的可视化函数，采用 `grede_analyse_1_untils.py` 中的逻辑和样式：

```python:analyse_v2/grade_analyse_4.ipynb
# ... existing code ...

def create_coherence_boxplots(data, grade_groups, save_path='../res/宏观叙事语篇连贯性箱线图比较.png'):
    """创建宏观叙事语篇连贯性箱线图比较 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事语篇连贯性箱线图...")
    
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

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(24, 14), dpi=120, constrained_layout=True)
    fig.suptitle('宏观叙事语篇连贯性各指标年级间分布对比', fontsize=24, fontweight='bold', y=0.98)
    
    # 绘制箱线图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 准备数据
        data_list = [grade_groups[grade][indicator_col] for grade in grade_names]
        
        # 构造DataFrame用于seaborn
        df_data = pd.DataFrame({
            '年级': np.repeat(grade_names, [len(x) for x in data_list]),
            '数值': np.concatenate(data_list)
        })
        
        # 绘制小提琴图+箱线图
        sns.violinplot(
            x='年级', y='数值', data=df_data, ax=ax,
            inner=None, palette=colors, linewidth=0, alpha=0.18
        )
        sns.boxplot(
            x='年级', y='数值', data=df_data, ax=ax,
            width=0.25, palette=colors, boxprops=dict(alpha=0.7)
        )
        
        # 均值点
        means = df_data.groupby('年级')['数值'].mean()
        ax.scatter(range(len(grade_names)), means, color='#E74C3C', s=120, marker='D', 
                  edgecolor='white', zorder=10, label='均值')
        
        # 美化
        ax.set_title(indicator_name, fontsize=16, fontweight='bold', color='#222')
        ax.set_xlabel('年级', fontsize=14, fontweight='bold')
        ax.set_ylabel('数值', fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 自定义图例
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='高一', alpha=0.7),
            Patch(facecolor='#A23B72', edgecolor='#A23B72', label='高二', alpha=0.7),
            Patch(facecolor='#F18F01', edgecolor='#F18F01', label='高三', alpha=0.7),
            Line2D([0], [0], marker='D', color='w', label='均值', markerfacecolor='#E74C3C', 
                   markeredgecolor='white', markersize=12)
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"箱线图已保存到: {save_path}")

def create_coherence_heatmap(data, grade_groups, save_path='../res/宏观叙事语篇连贯性变异系数热力图.png'):
    """创建宏观叙事语篇连贯性变异系数热力图 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事语篇连贯性变异系数热力图...")
    
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
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
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
    ax.set_title('宏观叙事语篇连贯性各年级各指标变异系数热力图', fontsize=20, fontweight='bold', color='#222', pad=20)
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
    
    print(f"热力图已保存到: {save_path}")

def create_coherence_density_plots(data, grade_groups, save_path='../res/宏观叙事语篇连贯性密度图.png'):
    """创建宏观叙事语篇连贯性密度图 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事语篇连贯性密度图...")
    
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(24, 14), dpi=120, constrained_layout=True)
    fig.suptitle('宏观叙事语篇连贯性各指标分布密度图', fontsize=22, fontweight='bold', y=0.98)
    
    # 绘制密度图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 绘制每个年级的密度图
        for j, grade_name in enumerate(grade_names):
            grade_data = grade_groups[grade_name][indicator_col]
            
            # 绘制直方图
            ax.hist(grade_data, bins=20, alpha=0.6, density=True, 
                   color=colors[j], edgecolor='white', linewidth=1.5, label=grade_name)
            
            # 添加核密度估计曲线
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(grade_data)
            x_range = np.linspace(grade_data.min(), grade_data.max(), 100)
            ax.plot(x_range, kde(x_range), color=colors[j], linewidth=3, alpha=0.8)
        
        # 设置标题和标签
        ax.set_title(indicator_name, fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        ax.set_xlabel('数值', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_ylabel('密度', fontsize=12, fontweight='bold', color='#2C3E50')
        
        # 添加图例
        ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"密度图已保存到: {save_path}")

def create_coherence_summary_chart(data, grade_groups, save_path='../res/宏观叙事语篇连贯性汇总图.png'):
    """创建宏观叙事语篇连贯性汇总图 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事语篇连贯性汇总图...")
    
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标组
    local_indicators = ['局部连贯-CRFAO1（29）', '局部连贯-CRFSO1（30）', '局部连贯-LSASS1(40)']
    global_indicators = ['整体连贯-CRFAOa（32）', '整体连贯-CRFSOa（33）', '整体连贯-LSASSp（42）', 
                        '整体连贯-LSAPP1（44）', '整体连贯-LSAGN（46）']
    
    # 计算各年级各指标组的平均变异系数
    local_cvs = []
    global_cvs = []
    
    for grade_name in grade_names:
        # 局部连贯性平均变异系数
        local_cv_list = []
        for indicator in local_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            local_cv_list.append(cv)
        local_cvs.append(np.mean(local_cv_list))
        
        # 整体连贯性平均变异系数
        global_cv_list = []
        for indicator in global_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            global_cv_list.append(cv)
        global_cvs.append(np.mean(global_cv_list))
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=120, constrained_layout=True)
    fig.suptitle('宏观叙事语篇连贯性指标组变异系数对比', fontsize=22, fontweight='bold', y=0.95)
    
    # 左图：柱状图对比
    x = np.arange(len(grade_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, local_cvs, width, label='局部连贯性指标', 
                    color=colors[0], alpha=0.8, edgecolor='#2C3E50', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, global_cvs, width, label='整体连贯性指标', 
                    color=colors[1], alpha=0.8, edgecolor='#2C3E50', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('年级', fontsize=14, fontweight='bold')
    ax1.set_ylabel('平均变异系数 (%)', fontsize=14, fontweight='bold')
    ax1.set_title('各年级指标组变异系数对比', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(grade_names, fontweight='bold')
    ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 右图：雷达图
    # 准备雷达图数据
    categories = grade_names
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形
    
    # 准备数据
    local_cvs_radar = local_cvs + [local_cvs[0]]
    global_cvs_radar = global_cvs + [global_cvs[0]]
    
    # 绘制雷达图
    ax2.plot(angles, local_cvs_radar, 'o-', linewidth=3, label='局部连贯性指标', 
             color=colors[0], alpha=0.8)
    ax2.fill(angles, local_cvs_radar, alpha=0.25, color=colors[0])
    
    ax2.plot(angles, global_cvs_radar, 'o-', linewidth=3, label='整体连贯性指标', 
             color=colors[1], alpha=0.8)
    ax2.fill(angles, global_cvs_radar, alpha=0.25, color=colors[1])
    
    # 设置雷达图属性
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontweight='bold')
    ax2.set_ylim(0, max(max(local_cvs), max(global_cvs)) * 1.2)
    ax2.set_title('变异系数雷达图对比', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"汇总图已保存到: {save_path}")

def create_coherence_visualizations(data, grade_groups):
    """创建宏观叙事语篇连贯性可视化图表 - 主函数"""
    print("开始生成宏观叙事语篇连贯性可视化图表...")
    
    # 创建各种类型的图表
    create_coherence_boxplots(data, grade_groups)
    create_coherence_heatmap(data, grade_groups)
    create_coherence_density_plots(data, grade_groups)
    create_coherence_summary_chart(data, grade_groups)
    
    print("所有可视化图表生成完成！")

# ... existing code ...
```

主要修改内容：

## 1. **箱线图函数 (`create_coherence_boxplots`)**
- 采用 `grede_analyse_1_untils.py` 中的小提琴图+箱线图组合
- 添加均值点标记（红色菱形）
- 使用相同的颜色主题和样式设置
- 添加自定义图例和美化效果

## 2. **热力图函数 (`create_coherence_heatmap`)**
- 采用 `grede_analyse_1_untils.py` 中的热力图样式
- 使用 `RdYlBu_r` 颜色映射
- 添加数值标注和美化效果
- 旋转x轴标签提高可读性

## 3. **密度图函数 (`create_coherence_density_plots`)**
- 采用 `grede_analyse_1_untils.py` 中的密度图样式
- 结合直方图和核密度估计曲线
- 使用相同的颜色主题和网格样式
- 添加图例和美化效果

## 4. **汇总图函数 (`create_coherence_summary_chart`)**
- 保持原有的柱状图和雷达图组合
- 采用相同的样式设置和美化效果

这样修改后，所有的可视化图表都将具有与 `grede_analyse_1_untils.py` 中相同的高级美观风格，包括：
- 统一的颜色主题
- 专业的样式设置
- 高质量的输出效果
- 美观的图例和标签

```python
# ... existing code ...

def create_coherence_boxplots(data, grade_groups, save_path='../res/宏观叙事语篇连贯性箱线图比较.png'):
    """创建宏观叙事语篇连贯性箱线图比较 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事语篇连贯性箱线图...")
    
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

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(24, 14), dpi=120, constrained_layout=True)
    fig.suptitle('宏观叙事语篇连贯性各指标年级间分布对比', fontsize=24, fontweight='bold', y=0.98)
    
    # 绘制箱线图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 准备数据
        data_list = [grade_groups[grade][indicator_col] for grade in grade_names]
        
        # 构造DataFrame用于seaborn
        df_data = pd.DataFrame({
            '年级': np.repeat(grade_names, [len(x) for x in data_list]),
            '数值': np.concatenate(data_list)
        })
        
        # 绘制小提琴图+箱线图
        sns.violinplot(
            x='年级', y='数值', data=df_data, ax=ax,
            inner=None, palette=colors, linewidth=0, alpha=0.18
        )
        sns.boxplot(
            x='年级', y='数值', data=df_data, ax=ax,
            width=0.25, palette=colors, boxprops=dict(alpha=0.7)
        )
        
        # 均值点
        means = df_data.groupby('年级')['数值'].mean()
        ax.scatter(range(len(grade_names)), means, color='#E74C3C', s=120, marker='D', 
                  edgecolor='white', zorder=10, label='均值')
        
        # 美化
        ax.set_title(indicator_name, fontsize=16, fontweight='bold', color='#222')
        ax.set_xlabel('年级', fontsize=14, fontweight='bold')
        ax.set_ylabel('数值', fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 自定义图例
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='高一', alpha=0.7),
            Patch(facecolor='#A23B72', edgecolor='#A23B72', label='高二', alpha=0.7),
            Patch(facecolor='#F18F01', edgecolor='#F18F01', label='高三', alpha=0.7),
            Line2D([0], [0], marker='D', color='w', label='均值', markerfacecolor='#E74C3C', 
                   markeredgecolor='white', markersize=12)
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"箱线图已保存到: {save_path}")

def create_coherence_heatmap(data, grade_groups, save_path='../res/宏观叙事语篇连贯性变异系数热力图.png'):
    """创建宏观叙事语篇连贯性变异系数热力图 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事语篇连贯性变异系数热力图...")
    
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
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
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
    ax.set_title('宏观叙事语篇连贯性各年级各指标变异系数热力图', fontsize=20, fontweight='bold', color='#222', pad=20)
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
    
    print(f"热力图已保存到: {save_path}")

def create_coherence_density_plots(data, grade_groups, save_path='../res/宏观叙事语篇连贯性密度图.png'):
    """创建宏观叙事语篇连贯性密度图 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事语篇连贯性密度图...")
    
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '局部连贯-CRFAO1': '局部连贯-CRFAO1（29）',
        '局部连贯-CRFSO1': '局部连贯-CRFSO1（30）',
        '局部连贯-LSASS1': '局部连贯-LSASS1(40)',
        '整体连贯-CRFAOa': '整体连贯-CRFAOa（32）',
        '整体连贯-CRFSOa': '整体连贯-CRFSOa（33）',
        '整体连贯-LSASSp': '整体连贯-LSASSp（42）',
        '整体连贯-LSAPP1': '整体连贯-LSAPP1（44）',
        '整体连贯-LSAGN': '整体连贯-LSAGN（46）'
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(24, 14), dpi=120, constrained_layout=True)
    fig.suptitle('宏观叙事语篇连贯性各指标分布密度图', fontsize=22, fontweight='bold', y=0.98)
    
    # 绘制密度图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # 绘制每个年级的密度图
        for j, grade_name in enumerate(grade_names):
            grade_data = grade_groups[grade_name][indicator_col]
            
            # 绘制直方图
            ax.hist(grade_data, bins=20, alpha=0.6, density=True, 
                   color=colors[j], edgecolor='white', linewidth=1.5, label=grade_name)
            
            # 添加核密度估计曲线
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(grade_data)
            x_range = np.linspace(grade_data.min(), grade_data.max(), 100)
            ax.plot(x_range, kde(x_range), color=colors[j], linewidth=3, alpha=0.8)
        
        # 设置标题和标签
        ax.set_title(indicator_name, fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        ax.set_xlabel('数值', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_ylabel('密度', fontsize=12, fontweight='bold', color='#2C3E50')
        
        # 添加图例
        ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"密度图已保存到: {save_path}")

def create_coherence_summary_chart(data, grade_groups, save_path='../res/宏观叙事语篇连贯性汇总图.png'):
    """创建宏观叙事语篇连贯性汇总图 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事语篇连贯性汇总图...")
    
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标组
    local_indicators = ['局部连贯-CRFAO1（29）', '局部连贯-CRFSO1（30）', '局部连贯-LSASS1(40)']
    global_indicators = ['整体连贯-CRFAOa（32）', '整体连贯-CRFSOa（33）', '整体连贯-LSASSp（42）', 
                        '整体连贯-LSAPP1（44）', '整体连贯-LSAGN（46）']
    
    # 计算各年级各指标组的平均变异系数
    local_cvs = []
    global_cvs = []
    
    for grade_name in grade_names:
        # 局部连贯性平均变异系数
        local_cv_list = []
        for indicator in local_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            local_cv_list.append(cv)
        local_cvs.append(np.mean(local_cv_list))
        
        # 整体连贯性平均变异系数
        global_cv_list = []
        for indicator in global_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            global_cv_list.append(cv)
        global_cvs.append(np.mean(global_cv_list))
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=120, constrained_layout=True)
    fig.suptitle('宏观叙事语篇连贯性指标组变异系数对比', fontsize=22, fontweight='bold', y=0.95)
    
    # 左图：柱状图对比
    x = np.arange(len(grade_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, local_cvs, width, label='局部连贯性指标', 
                    color=colors[0], alpha=0.8, edgecolor='#2C3E50', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, global_cvs, width, label='整体连贯性指标', 
                    color=colors[1], alpha=0.8, edgecolor='#2C3E50', linewidth=1.5)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('年级', fontsize=14, fontweight='bold')
    ax1.set_ylabel('平均变异系数 (%)', fontsize=14, fontweight='bold')
    ax1.set_title('各年级指标组变异系数对比', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(grade_names, fontweight='bold')
    ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 右图：雷达图
    # 准备雷达图数据
    categories = grade_names
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形
    
    # 准备数据
    local_cvs_radar = local_cvs + [local_cvs[0]]
    global_cvs_radar = global_cvs + [global_cvs[0]]
    
    # 绘制雷达图
    ax2.plot(angles, local_cvs_radar, 'o-', linewidth=3, label='局部连贯性指标', 
             color=colors[0], alpha=0.8)
    ax2.fill(angles, local_cvs_radar, alpha=0.25, color=colors[0])
    
    ax2.plot(angles, global_cvs_radar, 'o-', linewidth=3, label='整体连贯性指标', 
             color=colors[1], alpha=0.8)
    ax2.fill(angles, global_cvs_radar, alpha=0.25, color=colors[1])
    
    # 设置雷达图属性
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontweight='bold')
    ax2.set_ylim(0, max(max(local_cvs), max(global_cvs)) * 1.2)
    ax2.set_title('变异系数雷达图对比', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"汇总图已保存到: {save_path}")

def create_coherence_visualizations(data, grade_groups):
    """创建宏观叙事语篇连贯性可视化图表 - 主函数"""
    print("开始生成宏观叙事语篇连贯性可视化图表...")
    
    # 创建各种类型的图表
    create_coherence_boxplots(data, grade_groups)
    create_coherence_heatmap(data, grade_groups)
    create_coherence_density_plots(data, grade_groups)
    create_coherence_summary_chart(data, grade_groups)
    
    print("所有可视化图表生成完成！")

# ... existing code ...
```

---

**User**

输出结果 @分析结果.txt ，根据分析结果该如何回答●年级间比较：每组数据（如高一）同其他两组数据（如高二、高三）相比，三个年级之间是否有显著差异，描述差异特征；
★年级内比较：同一个年级内，“局部连贯性”3个指标、“整体连贯性”5个指标，哪组数据的内部差异程度大，描述差异特征，差异程度大的是影响该阶段能力发展的关键因素。

---

**Cursor**

根据分析结果，我来帮您回答统计要求中的问题：

## ● 年级间比较分析

### 1. **局部连贯性指标**

#### **局部连贯-CRFAO1（29）**
- **显著性差异检验：** F=1.6447, p=0.1948 > 0.05
- **结论：** 无显著性差异
- **差异特征：** 三个年级在该指标上水平相对稳定

#### **局部连贯-CRFSO1（30）**
- **显著性差异检验：** F=4.7034, p=0.0097 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（0.177）> 高二（0.138）> 高一（0.128）
  - 仅高一与高三存在显著差异（p=0.0109）
  - 呈现随年级升高而增长的趋势

#### **局部连贯-LSASS1（40）**
- **显著性差异检验：** F=1.0576, p=0.3486 > 0.05
- **结论：** 无显著性差异
- **差异特征：** 各年级水平相对稳定

### 2. **整体连贯性指标**

#### **整体连贯-CRFAOa（32）**
- **显著性差异检验：** F=4.5569, p=0.0112 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（0.614）> 高一（0.578）> 高二（0.546）
  - 仅高二与高三存在显著差异（p=0.0078）
  - 高三学生在该指标上表现最佳

#### **整体连贯-CRFSOa（33）**
- **显著性差异检验：** F=14.5839, p=0.0000 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（0.181）> 高一（0.129）> 高二（0.123）
  - 高一与高三、高二与高三均存在显著差异
  - 呈现随年级升高而显著增长的趋势

#### **整体连贯-LSASSp（42）**
- **显著性差异检验：** F=5.0847, p=0.0067 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（0.127）> 高一（0.114）> 高二（0.112）
  - 高一与高三、高二与高三均存在显著差异
  - 呈现随年级升高而增长的趋势

#### **整体连贯-LSAPP1（44）**
- **显著性差异检验：** F=7.7641, p=0.0005 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（0.462）> 高一（0.428）> 高二（0.408）
  - 高一与高三、高二与高三均存在显著差异
  - 呈现随年级升高而增长的趋势

#### **整体连贯-LSAGN（46）**
- **显著性差异检验：** F=10.7130, p=0.0000 < 0.05
- **结论：** 存在显著性差异
- **差异特征：**
  - 高三（0.264）> 高一（0.263）> 高二（0.249）
  - 高一与高二、高二与高三均存在显著差异
  - 高二在该指标上表现相对较低

## ★ 年级内比较分析

### **高一内部差异分析**
- **差异程度排序（从大到小）：**
  1. 局部连贯性指标：47.88%
  2. 整体连贯性指标：31.01%

**关键因素：** 局部连贯性指标（平均变异系数：47.88%）
- 该指标组内部差异最大，是影响高一阶段能力发展的关键因素
- 其中局部连贯-CRFSO1（30）变异系数最高（84.74%）

### **高二内部差异分析**
- **差异程度排序（从大到小）：**
  1. 局部连贯性指标：48.54%
  2. 整体连贯性指标：33.34%

**关键因素：** 局部连贯性指标（平均变异系数：48.54%）
- 该指标组内部差异最大，是影响高二阶段能力发展的关键因素
- 其中局部连贯-CRFSO1（30）变异系数最高（82.10%）

### **高三内部差异分析**
- **差异程度排序（从大到小）：**
  1. 局部连贯性指标：46.79%
  2. 整体连贯性指标：26.11%

**关键因素：** 局部连贯性指标（平均变异系数：46.79%）
- 该指标组内部差异最大，是影响高三阶段能力发展的关键因素
- 其中局部连贯-CRFSO1（30）变异系数最高（74.28%）

## 总结

**年级间发展特征：**
1. **整体连贯性指标**普遍呈现随年级升高而增长的趋势，特别是整体连贯-CRFSOa（33）和整体连贯-LSAPP1（44）增长最为显著
2. **局部连贯性指标**中，局部连贯-CRFSO1（30）呈现随年级升高而增长的趋势
3. 高三学生在大部分语篇连贯性指标上表现最佳，显示出语篇组织能力的提升

**年级内关键因素：**
所有年级中，**局部连贯性指标**都是内部差异最大的指标组，是影响各阶段能力发展的关键因素，特别是局部连贯-CRFSO1（30）指标在各年级都表现出最大的个体差异，说明学生在局部语篇连贯性方面存在较大的个体差异。

---

**User**

@grade_analyse_5.ipynb   现在我们完成统计要求的第5部分，●年级间比较：每组数据（如高一）同其他两组数据（如高二、高三）相比，三个年级之间是否有显著差异，描述差异特征；
★年级内比较：同一个年级内，“段落平均长度”、“段落平均长度标准差”，哪组数据的内部差异程度大（请将差异大的组别标明，后续需要具体分析）。请实现

---

**Cursor**

我来帮您实现统计要求的第5部分，分析宏观叙事段落长度维度的年级间比较和年级内比较。


现在我来实现宏观叙事段落长度维度的分析。根据数据，我们需要分析以下指标：

**段落平均长度指标：** 段落平均长度-DESPL-04
**段落平均长度标准差指标：** 段落平均长度标准差-DESPLd-05

```python:analyse_v2/grade_analyse_5.ipynb
# ... existing code ...

def descriptive_statistics_paragraph_length(data, grade_groups):
    """宏观叙事段落长度描述性统计分析"""
    print("=" * 80)
    print("宏观叙事段落长度描述性统计分析报告")
    print("=" * 80)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
    indicators = {
        '段落平均长度': '段落平均长度-DESPL-04',
        '段落平均长度标准差': '段落平均长度标准差-DESPLd-05'
    }
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(grade_groups[grade_name])}")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            mean_val = values.mean()
            std_val = values.std()
            print(f"  {indicator_name} - 均值：{mean_val:.3f}, 标准差：{std_val:.3f}")
    
    # 3. 正态性检验
    print("\n3. 正态性检验：")
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            stat, p_value = shapiro(values)
            normality = "正态分布" if p_value > 0.05 else "非正态分布"
            print(f"  {indicator_name} - Shapiro-Wilk检验：W={stat:.4f}, p={p_value:.4f} ({normality})")
    
    # 4. 方差齐性检验
    print("\n4. 方差齐性检验：")
    for indicator_name, indicator_col in indicators.items():
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        stat, p_value = levene(grade1_data, grade2_data, grade3_data)
        homogeneity = "方差齐性" if p_value > 0.05 else "方差非齐性"
        print(f"{indicator_name} - Levene检验：F={stat:.4f}, p={p_value:.4f} ({homogeneity})")
    
    # 5. 年级间比较分析
    print("\n5. 年级间比较分析：")
    for indicator_name, indicator_col in indicators.items():
        print(f"\n{indicator_name}年级间比较：")
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        # 单因素方差分析
        f_stat, p_value = f_oneway(grade1_data, grade2_data, grade3_data)
        print(f"  单因素方差分析：F={f_stat:.4f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("  结果：存在显著性差异")
            # Tukey HSD事后检验
            tukey_result = tukey_hsd(grade1_data, grade2_data, grade3_data)
            print("  Tukey HSD事后检验结果：")
            print(f"    高一 vs 高二：p={tukey_result.pvalue[0]:.4f}")
            print(f"    高一 vs 高三：p={tukey_result.pvalue[1]:.4f}")
            print(f"    高二 vs 高三：p={tukey_result.pvalue[2]:.4f}")
        else:
            print("  结果：无显著性差异")
    
    # 6. 年级内比较分析
    print("\n6. 年级内比较分析：")
    
    # 定义指标组
    paragraph_length_indicators = ['段落平均长度-DESPL-04']
    paragraph_std_indicators = ['段落平均长度标准差-DESPLd-05']
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}内部差异分析：")
        
        # 计算段落平均长度指标的平均变异系数
        length_cvs = []
        for indicator in paragraph_length_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            length_cvs.append(cv)
        avg_length_cv = np.mean(length_cvs)
        
        # 计算段落平均长度标准差指标的平均变异系数
        std_cvs = []
        for indicator in paragraph_std_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            std_cvs.append(cv)
        avg_std_cv = np.mean(std_cvs)
        
        # 比较差异程度
        if avg_length_cv > avg_std_cv:
            max_diff_group = "段落平均长度指标"
            max_cv = avg_length_cv
            min_diff_group = "段落平均长度标准差指标"
            min_cv = avg_std_cv
        else:
            max_diff_group = "段落平均长度标准差指标"
            max_cv = avg_std_cv
            min_diff_group = "段落平均长度指标"
            min_cv = avg_length_cv
        
        print(f"  各组指标平均变异系数（从大到小排序）：")
        print(f"    {max_diff_group}：{max_cv:.2f}%")
        print(f"    {min_diff_group}：{min_cv:.2f}%")
        print(f"  差异程度最大的指标组：{max_diff_group}（平均变异系数：{max_cv:.2f}%）")
        print(f"  该指标组是影响{grade_name}阶段能力发展的关键因素")
        
        # 显示各指标的具体变异系数
        print(f"  {max_diff_group}内部各指标变异系数：")
        if max_diff_group == "段落平均长度指标":
            for i, indicator in enumerate(paragraph_length_indicators):
                values = grade_groups[grade_name][indicator]
                cv = (values.std() / values.mean()) * 100
                print(f"    {indicator}：{cv:.2f}%")
        else:
            for i, indicator in enumerate(paragraph_std_indicators):
                values = grade_groups[grade_name][indicator]
                cv = (values.std() / values.mean()) * 100
                print(f"    {indicator}：{cv:.2f}%")
    
    print("\n" + "=" * 80)
    print("分析结论")
    print("=" * 80)

def create_paragraph_length_boxplots(data, grade_groups, save_path='../res/宏观叙事段落长度箱线图比较.png'):
    """创建宏观叙事段落长度箱线图比较 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事段落长度箱线图...")
    
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

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '段落平均长度': '段落平均长度-DESPL-04',
        '段落平均长度标准差': '段落平均长度标准差-DESPLd-05'
    }
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120, constrained_layout=True)
    fig.suptitle('宏观叙事段落长度各指标年级间分布对比', fontsize=22, fontweight='bold', y=0.95)
    
    # 绘制箱线图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        ax = axes[i]
        
        # 准备数据
        data_list = [grade_groups[grade][indicator_col] for grade in grade_names]
        
        # 构造DataFrame用于seaborn
        df_data = pd.DataFrame({
            '年级': np.repeat(grade_names, [len(x) for x in data_list]),
            '数值': np.concatenate(data_list)
        })
        
        # 绘制小提琴图+箱线图
        sns.violinplot(
            x='年级', y='数值', data=df_data, ax=ax,
            inner=None, palette=colors, linewidth=0, alpha=0.18
        )
        sns.boxplot(
            x='年级', y='数值', data=df_data, ax=ax,
            width=0.25, palette=colors, boxprops=dict(alpha=0.7)
        )
        
        # 均值点
        means = df_data.groupby('年级')['数值'].mean()
        ax.scatter(range(len(grade_names)), means, color='#E74C3C', s=120, marker='D', 
                  edgecolor='white', zorder=10, label='均值')
        
        # 美化
        ax.set_title(indicator_name, fontsize=18, fontweight='bold', color='#222')
        ax.set_xlabel('年级', fontsize=14, fontweight='bold')
        ax.set_ylabel('数值', fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 自定义图例
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='高一', alpha=0.7),
            Patch(facecolor='#A23B72', edgecolor='#A23B72', label='高二', alpha=0.7),
            Patch(facecolor='#F18F01', edgecolor='#F18F01', label='高三', alpha=0.7),
            Line2D([0], [0], marker='D', color='w', label='均值', markerfacecolor='#E74C3C', 
                   markeredgecolor='white', markersize=12)
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"箱线图已保存到: {save_path}")

def create_paragraph_length_heatmap(data, grade_groups, save_path='../res/宏观叙事段落长度变异系数热力图.png'):
    """创建宏观叙事段落长度变异系数热力图 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事段落长度变异系数热力图...")
    
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
        '段落平均长度': '段落平均长度-DESPL-04',
        '段落平均长度标准差': '段落平均长度标准差-DESPLd-05'
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=120, constrained_layout=True)
    
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
    ax.set_title('宏观叙事段落长度各年级各指标变异系数热力图', fontsize=18, fontweight='bold', color='#222', pad=20)
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
    
    print(f"热力图已保存到: {save_path}")

def create_paragraph_length_density_plots(data, grade_groups, save_path='../res/宏观叙事段落长度密度图.png'):
    """创建宏观叙事段落长度密度图 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事段落长度密度图...")
    
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '段落平均长度': '段落平均长度-DESPL-04',
        '段落平均长度标准差': '段落平均长度标准差-DESPLd-05'
    }
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120, constrained_layout=True)
    fig.suptitle('宏观叙事段落长度各指标分布密度图', fontsize=20, fontweight='bold', y=0.95)
    
    # 绘制密度图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        ax = axes[i]
        
        # 绘制每个年级的密度图
        for j, grade_name in enumerate(grade_names):
            values = grade_groups[grade_name][indicator_col]
            
            # 绘制直方图
            ax.hist(values, bins=20, alpha=0.6, density=True, 
                   color=colors[j], edgecolor='white', linewidth=1.5, label=grade_name)
            
            # 添加核密度估计曲线
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 100)
            ax.plot(x_range, kde(x_range), color=colors[j], linewidth=3, alpha=0.8)
        
        # 设置标题和标签
        ax.set_title(indicator_name, fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        ax.set_xlabel('数值', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_ylabel('密度', fontsize=12, fontweight='bold', color='#2C3E50')
        
        # 添加图例
        ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"密度图已保存到: {save_path}")

def create_paragraph_length_visualizations(data, grade_groups):
    """创建宏观叙事段落长度可视化图表 - 主函数"""
    print("开始生成宏观叙事段落长度可视化图表...")
    
    # 创建各种类型的图表
    create_paragraph_length_boxplots(data, grade_groups)
    create_paragraph_length_heatmap(data, grade_groups)
    create_paragraph_length_density_plots(data, grade_groups)
    
    print("所有可视化图表生成完成！")

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 描述性统计分析
    descriptive_statistics_paragraph_length(data, grade_groups)
    
    # 创建可视化图表
    create_paragraph_length_visualizations(data, grade_groups)
    
    # 保存分析结果到文件
    import sys
    from io import StringIO
    
    # 重定向输出到文件
    original_stdout = sys.stdout
    result_file = StringIO()
    sys.stdout = result_file
    
    # 重新运行分析
    descriptive_statistics_paragraph_length(data, grade_groups)
    
    # 恢复输出
    sys.stdout = original_stdout
    
    # 保存结果到文件
    with open('../res/res_v2/宏观叙事结构-段落长度/分析结果.txt', 'w', encoding='utf-8') as f:
        f.write(result_file.getvalue())
    
    print("分析完成！结果已保存到 '../res/res_v2/宏观叙事结构-段落长度/分析结果.txt'")

if __name__ == "__main__":
    main()
```

这个实现包含了：

1. **描述性统计分析**：计算各年级在各指标上的均值、标准差
2. **正态性检验**：使用Shapiro-Wilk检验判断数据分布
3. **方差齐性检验**：使用Levene检验判断方差是否相等
4. **年级间比较**：使用单因素方差分析和Tukey HSD事后检验
5. **年级内比较**：计算段落平均长度和段落平均长度标准差指标组的变异系数，找出差异最大的指标组
6. **可视化图表**：创建箱线图、热力图和密度图
7. **结果保存**：将分析结果保存到文件

运行这个代码后，您将得到完整的宏观叙事段落长度分析报告，包括年级间比较和年级内比较的结果。

```python
# ... existing code ...

def descriptive_statistics_paragraph_length(data, grade_groups):
    """宏观叙事段落长度描述性统计分析"""
    print("=" * 80)
    print("宏观叙事段落长度描述性统计分析报告")
    print("=" * 80)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
    indicators = {
        '段落平均长度': '段落平均长度-DESPL-04',
        '段落平均长度标准差': '段落平均长度标准差-DESPLd-05'
    }
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(grade_groups[grade_name])}")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            mean_val = values.mean()
            std_val = values.std()
            print(f"  {indicator_name} - 均值：{mean_val:.3f}, 标准差：{std_val:.3f}")
    
    # 3. 正态性检验
    print("\n3. 正态性检验：")
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            stat, p_value = shapiro(values)
            normality = "正态分布" if p_value > 0.05 else "非正态分布"
            print(f"  {indicator_name} - Shapiro-Wilk检验：W={stat:.4f}, p={p_value:.4f} ({normality})")
    
    # 4. 方差齐性检验
    print("\n4. 方差齐性检验：")
    for indicator_name, indicator_col in indicators.items():
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        stat, p_value = levene(grade1_data, grade2_data, grade3_data)
        homogeneity = "方差齐性" if p_value > 0.05 else "方差非齐性"
        print(f"{indicator_name} - Levene检验：F={stat:.4f}, p={p_value:.4f} ({homogeneity})")
    
    # 5. 年级间比较分析
    print("\n5. 年级间比较分析：")
    for indicator_name, indicator_col in indicators.items():
        print(f"\n{indicator_name}年级间比较：")
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        # 单因素方差分析
        f_stat, p_value = f_oneway(grade1_data, grade2_data, grade3_data)
        print(f"  单因素方差分析：F={f_stat:.4f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("  结果：存在显著性差异")
            # Tukey HSD事后检验
            tukey_result = tukey_hsd(grade1_data, grade2_data, grade3_data)
            print("  Tukey HSD事后检验结果：")
            print(f"    高一 vs 高二：p={tukey_result.pvalue[0]:.4f}")
            print(f"    高一 vs 高三：p={tukey_result.pvalue[1]:.4f}")
            print(f"    高二 vs 高三：p={tukey_result.pvalue[2]:.4f}")
        else:
            print("  结果：无显著性差异")
    
    # 6. 年级内比较分析
    print("\n6. 年级内比较分析：")
    
    # 定义指标组
    paragraph_length_indicators = ['段落平均长度-DESPL-04']
    paragraph_std_indicators = ['段落平均长度标准差-DESPLd-05']
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}内部差异分析：")
        
        # 计算段落平均长度指标的平均变异系数
        length_cvs = []
        for indicator in paragraph_length_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            length_cvs.append(cv)
        avg_length_cv = np.mean(length_cvs)
        
        # 计算段落平均长度标准差指标的平均变异系数
        std_cvs = []
        for indicator in paragraph_std_indicators:
            values = grade_groups[grade_name][indicator]
            cv = (values.std() / values.mean()) * 100
            std_cvs.append(cv)
        avg_std_cv = np.mean(std_cvs)
        
        # 比较差异程度
        if avg_length_cv > avg_std_cv:
            max_diff_group = "段落平均长度指标"
            max_cv = avg_length_cv
            min_diff_group = "段落平均长度标准差指标"
            min_cv = avg_std_cv
        else:
            max_diff_group = "段落平均长度标准差指标"
            max_cv = avg_std_cv
            min_diff_group = "段落平均长度指标"
            min_cv = avg_length_cv
        
        print(f"  各组指标平均变异系数（从大到小排序）：")
        print(f"    {max_diff_group}：{max_cv:.2f}%")
        print(f"    {min_diff_group}：{min_cv:.2f}%")
        print(f"  差异程度最大的指标组：{max_diff_group}（平均变异系数：{max_cv:.2f}%）")
        print(f"  该指标组是影响{grade_name}阶段能力发展的关键因素")
        
        # 显示各指标的具体变异系数
        print(f"  {max_diff_group}内部各指标变异系数：")
        if max_diff_group == "段落平均长度指标":
            for i, indicator in enumerate(paragraph_length_indicators):
                values = grade_groups[grade_name][indicator]
                cv = (values.std() / values.mean()) * 100
                print(f"    {indicator}：{cv:.2f}%")
        else:
            for i, indicator in enumerate(paragraph_std_indicators):
                values = grade_groups[grade_name][indicator]
                cv = (values.std() / values.mean()) * 100
                print(f"    {indicator}：{cv:.2f}%")
    
    print("\n" + "=" * 80)
    print("分析结论")
    print("=" * 80)

def create_paragraph_length_boxplots(data, grade_groups, save_path='../res/宏观叙事段落长度箱线图比较.png'):
    """创建宏观叙事段落长度箱线图比较 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事段落长度箱线图...")
    
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

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '段落平均长度': '段落平均长度-DESPL-04',
        '段落平均长度标准差': '段落平均长度标准差-DESPLd-05'
    }
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120, constrained_layout=True)
    fig.suptitle('宏观叙事段落长度各指标年级间分布对比', fontsize=22, fontweight='bold', y=0.95)
    
    # 绘制箱线图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        ax = axes[i]
        
        # 准备数据
        data_list = [grade_groups[grade][indicator_col] for grade in grade_names]
        
        # 构造DataFrame用于seaborn
        df_data = pd.DataFrame({
            '年级': np.repeat(grade_names, [len(x) for x in data_list]),
            '数值': np.concatenate(data_list)
        })
        
        # 绘制小提琴图+箱线图
        sns.violinplot(
            x='年级', y='数值', data=df_data, ax=ax,
            inner=None, palette=colors, linewidth=0, alpha=0.18
        )
        sns.boxplot(
            x='年级', y='数值', data=df_data, ax=ax,
            width=0.25, palette=colors, boxprops=dict(alpha=0.7)
        )
        
        # 均值点
        means = df_data.groupby('年级')['数值'].mean()
        ax.scatter(range(len(grade_names)), means, color='#E74C3C', s=120, marker='D', 
                  edgecolor='white', zorder=10, label='均值')
        
        # 美化
        ax.set_title(indicator_name, fontsize=18, fontweight='bold', color='#222')
        ax.set_xlabel('年级', fontsize=14, fontweight='bold')
        ax.set_ylabel('数值', fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 自定义图例
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='高一', alpha=0.7),
            Patch(facecolor='#A23B72', edgecolor='#A23B72', label='高二', alpha=0.7),
            Patch(facecolor='#F18F01', edgecolor='#F18F01', label='高三', alpha=0.7),
            Line2D([0], [0], marker='D', color='w', label='均值', markerfacecolor='#E74C3C', 
                   markeredgecolor='white', markersize=12)
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"箱线图已保存到: {save_path}")

def create_paragraph_length_heatmap(data, grade_groups, save_path='../res/宏观叙事段落长度变异系数热力图.png'):
    """创建宏观叙事段落长度变异系数热力图 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事段落长度变异系数热力图...")
    
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
        '段落平均长度': '段落平均长度-DESPL-04',
        '段落平均长度标准差': '段落平均长度标准差-DESPLd-05'
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=120, constrained_layout=True)
    
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
    ax.set_title('宏观叙事段落长度各年级各指标变异系数热力图', fontsize=18, fontweight='bold', color='#222', pad=20)
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
    
    print(f"热力图已保存到: {save_path}")

def create_paragraph_length_density_plots(data, grade_groups, save_path='../res/宏观叙事段落长度密度图.png'):
    """创建宏观叙事段落长度密度图 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事段落长度密度图...")
    
    # 设置全局风格
    sns.set_theme(style="whitegrid", font="SimHei", rc={
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.titleweight": "bold"
    })
    plt.rcParams['axes.unicode_minus'] = False

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '段落平均长度': '段落平均长度-DESPL-04',
        '段落平均长度标准差': '段落平均长度标准差-DESPLd-05'
    }
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120, constrained_layout=True)
    fig.suptitle('宏观叙事段落长度各指标分布密度图', fontsize=20, fontweight='bold', y=0.95)
    
    # 绘制密度图
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        ax = axes[i]
        
        # 绘制每个年级的密度图
        for j, grade_name in enumerate(grade_names):
            values = grade_groups[grade_name][indicator_col]
            
            # 绘制直方图
            ax.hist(values, bins=20, alpha=0.6, density=True, 
                   color=colors[j], edgecolor='white', linewidth=1.5, label=grade_name)
            
            # 添加核密度估计曲线
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 100)
            ax.plot(x_range, kde(x_range), color=colors[j], linewidth=3, alpha=0.8)
        
        # 设置标题和标签
        ax.set_title(indicator_name, fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
        ax.set_xlabel('数值', fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_ylabel('密度', fontsize=12, fontweight='bold', color='#2C3E50')
        
        # 添加图例
        ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"密度图已保存到: {save_path}")

def create_paragraph_length_visualizations(data, grade_groups):
    """创建宏观叙事段落长度可视化图表 - 主函数"""
    print("开始生成宏观叙事段落长度可视化图表...")
    
    # 创建各种类型的图表
    create_paragraph_length_boxplots(data, grade_groups)
    create_paragraph_length_heatmap(data, grade_groups)
    create_paragraph_length_density_plots(data, grade_groups)
    
    print("所有可视化图表生成完成！")

def main():
    """主函数"""
    # 加载数据
    data, grade_groups = load_and_prepare_data()
    
    # 描述性统计分析
    descriptive_statistics_paragraph_length(data, grade_groups)
    
    # 创建可视化图表
    create_paragraph_length_visualizations(data, grade_groups)
    
    # 保存分析结果到文件
    import sys
    from io import StringIO
    
    # 重定向输出到文件
    original_stdout = sys.stdout
    result_file = StringIO()
    sys.stdout = result_file
    
    # 重新运行分析
    descriptive_statistics_paragraph_length(data, grade_groups)
    
    # 恢复输出
    sys.stdout = original_stdout
    
    # 保存结果到文件
    with open('../res/res_v2/宏观叙事结构-段落长度/分析结果.txt', 'w', encoding='utf-8') as f:
        f.write(result_file.getvalue())
    
    print("分析完成！结果已保存到 '../res/res_v2/宏观叙事结构-段落长度/分析结果.txt'")

if __name__ == "__main__":
    main()
```

---

**User**

分析结果 @分析结果.txt 按照分析结果如何回答●年级间比较：每组数据（如高一）同其他两组数据（如高二、高三）相比，三个年级之间是否有显著差异，描述差异特征；
★年级内比较：同一个年级内，“段落平均长度”、“段落平均长度标准差”，哪组数据的内部差异程度大（请将差异大的组别标明，后续需要具体分析）。

---

**Cursor**

根据分析结果，我来帮您回答统计要求中的问题：

## ● 年级间比较分析

### 1. **段落平均长度指标**
- **显著性差异检验：** F=5.9440, p=0.0029 < 0.05
- **结论：** 三个年级在段落平均长度上存在显著性差异
- **差异特征：**
  - 高一（7.400）> 高三（6.895）> 高二（6.695）
  - 高一与高二存在显著差异（p=0.0027）
  - 高一与高三存在显著差异（p=0.0451）
  - 高二与高三无显著差异（p=0.6097）
  - 呈现从高一到高二下降，高二到高三略有回升的趋势

### 2. **段落平均长度标准差指标**
- **显著性差异检验：** F=17.5908, p=0.0000 < 0.05
- **结论：** 三个年级在段落平均长度标准差上存在显著性差异
- **差异特征：**
  - 高二（2.510）> 高一（1.711）> 高三（1.421）
  - 高一与高二存在显著差异（p=0.0001）
  - 高二与高三存在显著差异（p=0.0000）
  - 高一与高三无显著差异（p=0.2808）
  - 高二学生在段落长度变异性上达到最高水平

## ★ 年级内比较分析

### **高一内部差异分析**
- **段落平均长度指标：** 平均变异系数 21.41%
- **段落平均长度标准差指标：** 平均变异系数 76.63%
- **结论：** **段落平均长度标准差指标** 的内部差异程度更大
- **关键因素：** 段落平均长度标准差指标是影响高一阶段能力发展的关键因素

### **高二内部差异分析**
- **段落平均长度指标：** 平均变异系数 22.39%
- **段落平均长度标准差指标：** 平均变异系数 63.99%
- **结论：** **段落平均长度标准差指标** 的内部差异程度更大
- **关键因素：** 段落平均长度标准差指标是影响高二阶段能力发展的关键因素

### **高三内部差异分析**
- **段落平均长度指标：** 平均变异系数 20.02%
- **段落平均长度标准差指标：** 平均变异系数 74.67%
- **结论：** **段落平均长度标准差指标** 的内部差异程度更大
- **关键因素：** 段落平均长度标准差指标是影响高三阶段能力发展的关键因素

## 总结

**差异大的组别：段落平均长度标准差指标**

在所有三个年级中，**段落平均长度标准差指标** 都表现出更大的内部差异程度，其变异系数远高于段落平均长度指标。这表明学生在段落长度的一致性控制方面存在较大的个体差异，是影响各年级学生叙述能力发展的关键因素。

---

**User**

@grade_analyse_6.ipynb 现在完成统计要求的第6部分，●年级间比较：每组数据（如高一）同其他两组数据（如高二、高三）相比，三个年级之间是否有显著差异，描述差异特征；
★年级内比较：同一个年级内，“因果衔接”、“意图衔接”、“时体衔接”，哪组数据的内部差异程度大，描述差异特征，差异程度大的是影响该阶段能力发展的关键因素。 请实现

---

**Cursor**

我来帮您实现统计要求的第6部分，分析宏观叙事情境模型维度的年级间比较和年级内比较。


现在我来实现宏观叙事情境模型维度的分析。根据数据，我们需要分析以下指标：

**因果衔接指标：** 因果衔接-SMCAUSr-64
**意图衔接指标：** 意图衔接-SMINTEr-65  
**时体衔接指标：** 时体衔接-SMTEMP-68

```python:analyse_v2/grade_analyse_6.ipynb
# ... existing code ...

def descriptive_statistics_situation_model(data, grade_groups):
    """宏观叙事情境模型描述性统计分析"""
    print("=" * 80)
    print("宏观叙事情境模型描述性统计分析报告")
    print("=" * 80)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
    indicators = {
        '因果衔接': '因果衔接-SMCAUSr-64',
        '意图衔接': '意图衔接-SMINTEr-65',
        '时体衔接': '时体衔接-SMTEMP-68'
    }
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(grade_groups[grade_name])}")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            print(f"  {indicator_name} - 均值：{values.mean():.3f}, 标准差：{values.std():.3f}")
    
    # 3. 正态性检验
    print("\n3. 正态性检验：")
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            stat, p_value = shapiro(values)
            normality = "正态分布" if p_value > 0.05 else "非正态分布"
            print(f"  {indicator_name} - Shapiro-Wilk检验：W={stat:.4f}, p={p_value:.4f} ({normality})")
    
    # 4. 方差齐性检验
    print("\n4. 方差齐性检验：")
    
    for indicator_name, indicator_col in indicators.items():
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        stat, p_value = levene(grade1_data, grade2_data, grade3_data)
        homogeneity = "方差齐性" if p_value > 0.05 else "方差非齐性"
        print(f"{indicator_name} - Levene检验：F={stat:.4f}, p={p_value:.4f} ({homogeneity})")
    
    # 5. 年级间比较分析
    print("\n5. 年级间比较分析：")
    
    for indicator_name, indicator_col in indicators.items():
        print(f"\n{indicator_name}年级间比较：")
        
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        # 单因素方差分析
        f_stat, p_value = f_oneway(grade1_data, grade2_data, grade3_data)
        print(f"  单因素方差分析：F={f_stat:.4f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("  结果：存在显著性差异")
            
            # Tukey HSD事后检验
            tukey_result = tukey_hsd(grade1_data, grade2_data, grade3_data)
            print("  Tukey HSD事后检验结果：")
            print(f"    高一 vs 高二：p={tukey_result.pvalue[0]:.4f}")
            print(f"    高一 vs 高三：p={tukey_result.pvalue[1]:.4f}")
            print(f"    高二 vs 高三：p={tukey_result.pvalue[2]:.4f}")
        else:
            print("  结果：无显著性差异")
    
    # 6. 年级内比较分析
    print("\n6. 年级内比较分析：")
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}内部差异分析：")
        
        # 计算各指标组的变异系数
        cv_dict = {}
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            cv = (values.std() / values.mean()) * 100
            cv_dict[indicator_name] = cv
        
        # 按变异系数排序
        sorted_cv = sorted(cv_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("  各组指标平均变异系数（从大到小排序）：")
        for indicator_name, cv in sorted_cv:
            print(f"    {indicator_name}指标：{cv:.2f}%")
        
        # 找出差异程度最大的指标组
        max_cv_indicator, max_cv = sorted_cv[0]
        print(f"  差异程度最大的指标组：{max_cv_indicator}指标（平均变异系数：{max_cv:.2f}%）")
        print(f"  该指标组是影响{grade_name}阶段能力发展的关键因素")
        
        # 显示该指标组内部各指标的变异系数
        print(f"  {max_cv_indicator}指标内部各指标变异系数：")
        for indicator_name, indicator_col in indicators.items():
            if indicator_name == max_cv_indicator:
                values = grade_groups[grade_name][indicator_col]
                cv = (values.std() / values.mean()) * 100
                print(f"    {indicator_col}：{cv:.2f}%")

def create_situation_model_visualizations(data, grade_groups):
    """创建宏观叙事情境模型可视化图表"""
    print("生成宏观叙事情境模型可视化图表...")
    
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

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '因果衔接': '因果衔接-SMCAUSr-64',
        '意图衔接': '意图衔接-SMINTEr-65',
        '时体衔接': '时体衔接-SMTEMP-68'
    }
    
    # 创建箱线图
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('宏观叙事情境模型各指标年级间比较', fontsize=24, fontweight='bold', y=0.98)
    
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        ax = axes[i]
        
        # 准备数据
        data_list = [grade_groups[grade][indicator_col] for grade in grade_names]
        
        # 创建箱线图
        bp = ax.boxplot(data_list, patch_artist=True, 
                       boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                       medianprops=dict(color='#E74C3C', linewidth=3),
                       whiskerprops=dict(color='#2C3E50', linewidth=2),
                       capprops=dict(color='#2C3E50', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=6, alpha=0.7))
        
        # 设置颜色
        for patch in bp['boxes']:
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        
        # 添加均值点
        for j, grade_data in enumerate(data_list):
            mean_val = np.mean(grade_data)
            ax.plot(j+1, mean_val, 'o', color='#E74C3C', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        # 设置标题和标签
        ax.set_title(f'{indicator_name}', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('年级', fontsize=16, fontweight='bold')
        ax.set_ylabel('指标值', fontsize=16, fontweight='bold')
        ax.set_xticklabels(grade_names, fontsize=14)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置背景色
        ax.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    plt.savefig('../res/宏观叙事情境模型箱线图比较.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 创建变异系数热力图
    print("生成变异系数热力图...")
    
    # 计算各年级各指标的变异系数
    cv_data = []
    for grade_name in grade_names:
        row = []
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            cv = (values.std() / values.mean()) * 100
            row.append(cv)
        cv_data.append(row)
    
    cv_df = pd.DataFrame(cv_data, 
                        index=grade_names, 
                        columns=[name for name in indicators.keys()])
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    
    # 使用自定义颜色映射
    colors_list = ['#E8F4FD', '#B3D9F2', '#66B2FF', '#0066CC', '#004080']
    n_bins = len(colors_list)
    
    # 创建热力图
    sns.heatmap(cv_df, annot=True, fmt='.1f', cmap='Blues', 
                cbar_kws={'label': '变异系数 (%)', 'shrink': 0.8},
                square=True, linewidths=0.5, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.title('宏观叙事情境模型各年级各指标变异系数热力图', 
              fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('指标类型', fontsize=16, fontweight='bold')
    plt.ylabel('年级', fontsize=16, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig('../res/宏观叙事情境模型变异系数热力图.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

def main():
    """主函数"""
    # 执行描述性统计分析
    descriptive_statistics_situation_model(data, grade_groups)
    
    # 创建可视化图表
    create_situation_model_visualizations(data, grade_groups)
    
    # 保存分析结果到文件
    import sys
    from io import StringIO
    
    # 重定向输出到字符串
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result
    
    # 重新执行分析
    descriptive_statistics_situation_model(data, grade_groups)
    
    # 恢复标准输出
    sys.stdout = old_stdout
    
    # 保存结果到文件
    output_path = '../res/res_v2/宏观叙事-情境模型/分析结果.txt'
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.getvalue())
    
    print(f"分析结果已保存到：{output_path}")
    print("=" * 80)
    print("分析完成！")
    print("=" * 80)

# 执行主函数
if __name__ == "__main__":
    main()
```

这个代码实现了以下功能：

1. **描述性统计分析**：
   - 基本统计信息（样本数、年级分布）
   - 各年级在各指标上的描述性统计
   - 正态性检验（Shapiro-Wilk检验）
   - 方差齐性检验（Levene检验）
   - 年级间比较分析（单因素方差分析和Tukey HSD事后检验）
   - 年级内比较分析（变异系数计算和排序）

2. **可视化图表**：
   - 专业美观的箱线图比较
   - 变异系数热力图
   - 采用高级的seaborn风格和配色方案

3. **结果保存**：
   - 将分析结果保存到文本文件
   - 将图表保存为高质量PNG文件

运行这个代码后，您将得到完整的宏观叙事情境模型分析报告，包括年级间比较和年级内比较的详细结果。

```python
# ... existing code ...

def descriptive_statistics_situation_model(data, grade_groups):
    """宏观叙事情境模型描述性统计分析"""
    print("=" * 80)
    print("宏观叙事情境模型描述性统计分析报告")
    print("=" * 80)
    
    # 1. 基本统计信息
    print("\n1. 数据基本信息：")
    print(f"总样本数：{len(data)}")
    print(f"年级分布：\n{data['年级名称'].value_counts().sort_index()}")
    
    # 2. 各年级描述性统计
    print("\n2. 各年级描述性统计：")
    
    indicators = {
        '因果衔接': '因果衔接-SMCAUSr-64',
        '意图衔接': '意图衔接-SMINTEr-65',
        '时体衔接': '时体衔接-SMTEMP-68'
    }
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        print(f"  样本数：{len(grade_groups[grade_name])}")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            print(f"  {indicator_name} - 均值：{values.mean():.3f}, 标准差：{values.std():.3f}")
    
    # 3. 正态性检验
    print("\n3. 正态性检验：")
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}：")
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            stat, p_value = shapiro(values)
            normality = "正态分布" if p_value > 0.05 else "非正态分布"
            print(f"  {indicator_name} - Shapiro-Wilk检验：W={stat:.4f}, p={p_value:.4f} ({normality})")
    
    # 4. 方差齐性检验
    print("\n4. 方差齐性检验：")
    
    for indicator_name, indicator_col in indicators.items():
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        stat, p_value = levene(grade1_data, grade2_data, grade3_data)
        homogeneity = "方差齐性" if p_value > 0.05 else "方差非齐性"
        print(f"{indicator_name} - Levene检验：F={stat:.4f}, p={p_value:.4f} ({homogeneity})")
    
    # 5. 年级间比较分析
    print("\n5. 年级间比较分析：")
    
    for indicator_name, indicator_col in indicators.items():
        print(f"\n{indicator_name}年级间比较：")
        
        grade1_data = grade_groups['高一'][indicator_col]
        grade2_data = grade_groups['高二'][indicator_col]
        grade3_data = grade_groups['高三'][indicator_col]
        
        # 单因素方差分析
        f_stat, p_value = f_oneway(grade1_data, grade2_data, grade3_data)
        print(f"  单因素方差分析：F={f_stat:.4f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print("  结果：存在显著性差异")
            
            # Tukey HSD事后检验
            tukey_result = tukey_hsd(grade1_data, grade2_data, grade3_data)
            print("  Tukey HSD事后检验结果：")
            print(f"    高一 vs 高二：p={tukey_result.pvalue[0]:.4f}")
            print(f"    高一 vs 高三：p={tukey_result.pvalue[1]:.4f}")
            print(f"    高二 vs 高三：p={tukey_result.pvalue[2]:.4f}")
        else:
            print("  结果：无显著性差异")
    
    # 6. 年级内比较分析
    print("\n6. 年级内比较分析：")
    
    for grade_name in ['高一', '高二', '高三']:
        print(f"\n{grade_name}内部差异分析：")
        
        # 计算各指标组的变异系数
        cv_dict = {}
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            cv = (values.std() / values.mean()) * 100
            cv_dict[indicator_name] = cv
        
        # 按变异系数排序
        sorted_cv = sorted(cv_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("  各组指标平均变异系数（从大到小排序）：")
        for indicator_name, cv in sorted_cv:
            print(f"    {indicator_name}指标：{cv:.2f}%")
        
        # 找出差异程度最大的指标组
        max_cv_indicator, max_cv = sorted_cv[0]
        print(f"  差异程度最大的指标组：{max_cv_indicator}指标（平均变异系数：{max_cv:.2f}%）")
        print(f"  该指标组是影响{grade_name}阶段能力发展的关键因素")
        
        # 显示该指标组内部各指标的变异系数
        print(f"  {max_cv_indicator}指标内部各指标变异系数：")
        for indicator_name, indicator_col in indicators.items():
            if indicator_name == max_cv_indicator:
                values = grade_groups[grade_name][indicator_col]
                cv = (values.std() / values.mean()) * 100
                print(f"    {indicator_col}：{cv:.2f}%")

def create_situation_model_visualizations(data, grade_groups):
    """创建宏观叙事情境模型可视化图表"""
    print("生成宏观叙事情境模型可视化图表...")
    
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

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '因果衔接': '因果衔接-SMCAUSr-64',
        '意图衔接': '意图衔接-SMINTEr-65',
        '时体衔接': '时体衔接-SMTEMP-68'
    }
    
    # 创建箱线图
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('宏观叙事情境模型各指标年级间比较', fontsize=24, fontweight='bold', y=0.98)
    
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        ax = axes[i]
        
        # 准备数据
        data_list = [grade_groups[grade][indicator_col] for grade in grade_names]
        
        # 创建箱线图
        bp = ax.boxplot(data_list, patch_artist=True, 
                       boxprops=dict(facecolor='white', alpha=0.8, edgecolor='#2C3E50', linewidth=2),
                       medianprops=dict(color='#E74C3C', linewidth=3),
                       whiskerprops=dict(color='#2C3E50', linewidth=2),
                       capprops=dict(color='#2C3E50', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='#3498DB', markersize=6, alpha=0.7))
        
        # 设置颜色
        for patch in bp['boxes']:
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        
        # 添加均值点
        for j, grade_data in enumerate(data_list):
            mean_val = np.mean(grade_data)
            ax.plot(j+1, mean_val, 'o', color='#E74C3C', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        # 设置标题和标签
        ax.set_title(f'{indicator_name}', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('年级', fontsize=16, fontweight='bold')
        ax.set_ylabel('指标值', fontsize=16, fontweight='bold')
        ax.set_xticklabels(grade_names, fontsize=14)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置背景色
        ax.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    plt.savefig('../res/宏观叙事情境模型箱线图比较.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # 创建变异系数热力图
    print("生成变异系数热力图...")
    
    # 计算各年级各指标的变异系数
    cv_data = []
    for grade_name in grade_names:
        row = []
        for indicator_name, indicator_col in indicators.items():
            values = grade_groups[grade_name][indicator_col]
            cv = (values.std() / values.mean()) * 100
            row.append(cv)
        cv_data.append(row)
    
    cv_df = pd.DataFrame(cv_data, 
                        index=grade_names, 
                        columns=[name for name in indicators.keys()])
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    
    # 使用自定义颜色映射
    colors_list = ['#E8F4FD', '#B3D9F2', '#66B2FF', '#0066CC', '#004080']
    n_bins = len(colors_list)
    
    # 创建热力图
    sns.heatmap(cv_df, annot=True, fmt='.1f', cmap='Blues', 
                cbar_kws={'label': '变异系数 (%)', 'shrink': 0.8},
                square=True, linewidths=0.5, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.title('宏观叙事情境模型各年级各指标变异系数热力图', 
              fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('指标类型', fontsize=16, fontweight='bold')
    plt.ylabel('年级', fontsize=16, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig('../res/宏观叙事情境模型变异系数热力图.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()

def main():
    """主函数"""
    # 执行描述性统计分析
    descriptive_statistics_situation_model(data, grade_groups)
    
    # 创建可视化图表
    create_situation_model_visualizations(data, grade_groups)
    
    # 保存分析结果到文件
    import sys
    from io import StringIO
    
    # 重定向输出到字符串
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result
    
    # 重新执行分析
    descriptive_statistics_situation_model(data, grade_groups)
    
    # 恢复标准输出
    sys.stdout = old_stdout
    
    # 保存结果到文件
    output_path = '../res/res_v2/宏观叙事-情境模型/分析结果.txt'
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.getvalue())
    
    print(f"分析结果已保存到：{output_path}")
    print("=" * 80)
    print("分析完成！")
    print("=" * 80)

# 执行主函数
if __name__ == "__main__":
    main()
```

---

**User**

@grade_analyse_6.ipynb 这里的可视化我希望采用@grede_analyse_1_untils.py 中对应图的逻辑和样式

---

**Cursor**

我来帮您修改 `grade_analyse_6.ipynb` 中的可视化函数，采用 `grede_analyse_1_untils.py` 中的逻辑和样式。


现在我来修改 `grade_analyse_6.ipynb` 中的可视化函数，采用 `grede_analyse_1_untils.py` 中的逻辑和样式：

```python:analyse_v2/grade_analyse_6.ipynb
# ... existing code ...

def create_situation_model_boxplots(data, grade_groups, save_path='../res/宏观叙事情境模型箱线图比较.png'):
    """创建宏观叙事情境模型箱线图比较 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事情境模型箱线图...")
    
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

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '因果衔接': '因果衔接-SMCAUSr-64',
        '意图衔接': '意图衔接-SMINTEr-65',
        '时体衔接': '时体衔接-SMTEMP-68'
    }
    
    # 构造DataFrame用于seaborn
    df_list = []
    for indicator_name, indicator_col in indicators.items():
        for grade in grade_names:
            grade_data = grade_groups[grade][indicator_col]
            df_temp = pd.DataFrame({
                '年级': [grade] * len(grade_data),
                '指标值': grade_data,
                '指标类型': [indicator_name] * len(grade_data)
            })
            df_list.append(df_temp)
    
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), dpi=120, constrained_layout=True)
    
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        ax = axes[i]
        df_indicator = df_combined[df_combined['指标类型'] == indicator_name]
        
        # 小提琴图+箱线图
        sns.violinplot(
            x='年级', y='指标值', data=df_indicator, ax=ax,
            inner=None, palette=colors, linewidth=0, alpha=0.18
        )
        sns.boxplot(
            x='年级', y='指标值', data=df_indicator, ax=ax,
            width=0.25, palette=colors, boxprops=dict(alpha=0.7)
        )
        
        # 均值点
        means = df_indicator.groupby('年级')['指标值'].mean()
        ax.scatter(range(len(grade_names)), means, color='#E74C3C', s=120, marker='D', 
                  edgecolor='white', zorder=10, label='均值')
        
        # 美化
        ax.set_title(f'{indicator_name}年级间分布对比', fontsize=20, fontweight='bold', color='#222')
        ax.set_xlabel('年级', fontsize=16, fontweight='bold')
        ax.set_ylabel('指标值', fontsize=16, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 自定义图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='高一', alpha=0.7),
        Patch(facecolor='#A23B72', edgecolor='#A23B72', label='高二', alpha=0.7),
        Patch(facecolor='#F18F01', edgecolor='#F18F01', label='高三', alpha=0.7),
        Line2D([0], [0], marker='D', color='w', label='均值', markerfacecolor='#E74C3C', 
               markeredgecolor='white', markersize=12)
    ]
    
    for ax in axes:
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # 去除多余边距
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"箱线图已保存为: {save_path}")

def create_situation_model_heatmap(data, grade_groups, save_path='../res/宏观叙事情境模型变异系数热力图.png'):
    """创建宏观叙事情境模型变异系数热力图"""
    print("生成宏观叙事情境模型变异系数热力图...")
    
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
        '因果衔接': '因果衔接-SMCAUSr-64',
        '意图衔接': '意图衔接-SMINTEr-65',
        '时体衔接': '时体衔接-SMTEMP-68'
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=120, constrained_layout=True)
    
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

def create_situation_model_visualizations(data, grade_groups):
    """创建宏观叙事情境模型可视化图表"""
    print("生成宏观叙事情境模型可视化图表...")
    
    # 创建箱线图
    create_situation_model_boxplots(data, grade_groups)
    
    # 创建变异系数热力图
    create_situation_model_heatmap(data, grade_groups)

# ... existing code ...
```

主要修改包括：

1. **箱线图函数**：
   - 采用小提琴图+箱线图的组合显示
   - 添加均值点标记
   - 使用统一的颜色主题和样式
   - 添加自定义图例
   - 美化网格和边框

2. **热力图函数**：
   - 使用 `RdYlBu_r` 颜色映射
   - 添加中心值设置
   - 美化标题和标签
   - 旋转x轴标签提高可读性
   - 统一的样式设置

3. **整体风格**：
   - 采用seaborn的whitegrid风格
   - 统一的字体和颜色设置
   - 高质量的输出设置（300 DPI）
   - 专业的布局和间距

这样修改后，可视化图表将具有与 `grede_analyse_1_untils.py` 中相同的专业美观风格。

```python
# ... existing code ...

def create_situation_model_boxplots(data, grade_groups, save_path='../res/宏观叙事情境模型箱线图比较.png'):
    """创建宏观叙事情境模型箱线图比较 - 采用高级美观的seaborn风格"""
    print("生成宏观叙事情境模型箱线图...")
    
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

    # 设置颜色主题
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    grade_names = ['高一', '高二', '高三']
    
    # 定义指标
    indicators = {
        '因果衔接': '因果衔接-SMCAUSr-64',
        '意图衔接': '意图衔接-SMINTEr-65',
        '时体衔接': '时体衔接-SMTEMP-68'
    }
    
    # 构造DataFrame用于seaborn
    df_list = []
    for indicator_name, indicator_col in indicators.items():
        for grade in grade_names:
            grade_data = grade_groups[grade][indicator_col]
            df_temp = pd.DataFrame({
                '年级': [grade] * len(grade_data),
                '指标值': grade_data,
                '指标类型': [indicator_name] * len(grade_data)
            })
            df_list.append(df_temp)
    
    df_combined = pd.concat(df_list, ignore_index=True)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 8), dpi=120, constrained_layout=True)
    
    for i, (indicator_name, indicator_col) in enumerate(indicators.items()):
        ax = axes[i]
        df_indicator = df_combined[df_combined['指标类型'] == indicator_name]
        
        # 小提琴图+箱线图
        sns.violinplot(
            x='年级', y='指标值', data=df_indicator, ax=ax,
            inner=None, palette=colors, linewidth=0, alpha=0.18
        )
        sns.boxplot(
            x='年级', y='指标值', data=df_indicator, ax=ax,
            width=0.25, palette=colors, boxprops=dict(alpha=0.7)
        )
        
        # 均值点
        means = df_indicator.groupby('年级')['指标值'].mean()
        ax.scatter(range(len(grade_names)), means, color='#E74C3C', s=120, marker='D', 
                  edgecolor='white', zorder=10, label='均值')
        
        # 美化
        ax.set_title(f'{indicator_name}年级间分布对比', fontsize=20, fontweight='bold', color='#222')
        ax.set_xlabel('年级', fontsize=16, fontweight='bold')
        ax.set_ylabel('指标值', fontsize=16, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 自定义图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Patch(facecolor='#2E86AB', edgecolor='#2E86AB', label='高一', alpha=0.7),
        Patch(facecolor='#A23B72', edgecolor='#A23B72', label='高二', alpha=0.7),
        Patch(facecolor='#F18F01', edgecolor='#F18F01', label='高三', alpha=0.7),
        Line2D([0], [0], marker='D', color='w', label='均值', markerfacecolor='#E74C3C', 
               markeredgecolor='white', markersize=12)
    ]
    
    for ax in axes:
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # 去除多余边距
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"箱线图已保存为: {save_path}")

def create_situation_model_heatmap(data, grade_groups, save_path='../res/宏观叙事情境模型变异系数热力图.png'):
    """创建宏观叙事情境模型变异系数热力图"""
    print("生成宏观叙事情境模型变异系数热力图...")
    
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
        '因果衔接': '因果衔接-SMCAUSr-64',
        '意图衔接': '意图衔接-SMINTEr-65',
        '时体衔接': '时体衔接-SMTEMP-68'
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
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=120, constrained_layout=True)
    
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

def create_situation_model_visualizations(data, grade_groups):
    """创建宏观叙事情境模型可视化图表"""
    print("生成宏观叙事情境模型可视化图表...")
    
    # 创建箱线图
    create_situation_model_boxplots(data, grade_groups)
    
    # 创建变异系数热力图
    create_situation_model_heatmap(data, grade_groups)

# ... existing code ...
```

---

**User**

分析结果 @分析结果.txt ，如何回答●年级间比较：每组数据（如高一）同其他两组数据（如高二、高三）相比，三个年级之间是否有显著差异，描述差异特征；
★年级内比较：同一个年级内，“因果衔接”、“意图衔接”、“时体衔接”，哪组数据的内部差异程度大，描述差异特征，差异程度大的是影响该阶段能力发展的关键因素。

---

**Cursor**

根据分析结果，我来帮您回答统计要求中的问题：

## ● 年级间比较分析

### 1. **因果衔接指标**
- **显著性差异检验：** F=2.6579, p=0.0718 > 0.05
- **结论：** 三个年级在因果衔接上无显著性差异
- **差异特征：** 各年级的因果衔接水平相对稳定，均值在0.255-0.319之间

### 2. **意图衔接指标**
- **显著性差异检验：** F=10.2641, p=0.0000 < 0.05
- **结论：** 三个年级在意图衔接上存在显著性差异
- **差异特征：**
  - 高三（0.524）> 高二（0.391）> 高一（0.326）
  - 仅高二与高三存在显著差异（p=0.0086）
  - 高一与高二、高一与高三均无显著差异
  - 呈现随年级升高而增长的趋势

### 3. **时体衔接指标**
- **显著性差异检验：** F=2.3857, p=0.0938 > 0.05
- **结论：** 三个年级在时体衔接上无显著性差异
- **差异特征：** 各年级的时体衔接水平相对稳定，均值在0.856-0.881之间

## ★ 年级内比较分析

### **高一内部差异分析**
- **因果衔接指标：** 平均变异系数 69.83%
- **意图衔接指标：** 平均变异系数 58.81%
- **时体衔接指标：** 平均变异系数 10.77%
- **结论：** **因果衔接指标** 的内部差异程度更大
- **关键因素：** 因果衔接指标是影响高一阶段能力发展的关键因素

### **高二内部差异分析**
- **因果衔接指标：** 平均变异系数 91.68%
- **意图衔接指标：** 平均变异系数 62.61%
- **时体衔接指标：** 平均变异系数 8.60%
- **结论：** **因果衔接指标** 的内部差异程度更大
- **关键因素：** 因果衔接指标是影响高二阶段能力发展的关键因素

### **高三内部差异分析**
- **因果衔接指标：** 平均变异系数 87.31%
- **意图衔接指标：** 平均变异系数 85.57%
- **时体衔接指标：** 平均变异系数 10.04%
- **结论：** **因果衔接指标** 的内部差异程度更大
- **关键因素：** 因果衔接指标是影响高三阶段能力发展的关键因素

## 总结

**差异大的组别：因果衔接指标**

在所有三个年级中，**因果衔接指标** 都表现出最大的内部差异程度，其变异系数远高于其他两个指标。这表明学生在因果关系的理解和表达方面存在较大的个体差异，是影响各年级学生叙述能力发展的关键因素。

**年级间发展特征：**
- 意图衔接能力呈现随年级升高而增长的趋势
- 因果衔接和时体衔接在各年级间保持相对稳定
- 高二到高三阶段在意图衔接方面有显著提升

