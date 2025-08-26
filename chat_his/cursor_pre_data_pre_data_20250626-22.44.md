# @pre_data/ 现在我要完成关于学生叙述能力的分析，请你分析统计要求内筒，结合pre_data的数据，分析是否可行，然后先给出一个分析方案。不需要代码实现
_Exported on 2025/8/26 at GMT+8 22:43:20 from Cursor (1.4.5)_

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

