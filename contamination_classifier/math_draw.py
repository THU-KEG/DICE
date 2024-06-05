import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取 .log 文件中的数据
file_path = 'math_Test.log'

with open(file_path, 'r') as file:
    data_str = file.read().strip()
    data_str = data_str.strip('[]')  # 去除开头和结尾的方括号
    data = np.array([float(x) for x in data_str.split(',')])  # 将字符串转换为浮点数列表

# 统计每个区间的个数
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
counts, _ = np.histogram(data, bins)

# 创建DataFrame
df = pd.DataFrame({'Contamination Score Interval': ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'], 'Number of Samples': counts})

# 设置绘图风格，不使用白色网格背景
sns.set(style="white")

# 定义马卡龙色系调色板的第一个颜色
macaron_color = sns.color_palette("pastel")[0]

# 创建柱状图
plt.figure(figsize=(11, 8))

# 创建柱状图，设置每个柱子颜色一致，并调整柱子宽度
bar_plot = sns.barplot(x='Contamination Score Interval', y='Number of Samples', data=df, color=macaron_color, width=0.5)

# 设置Y轴范围从0到最大计数值
bar_plot.set_ylim(0, counts.max() + 500)

# 添加标题和标签
bar_plot.set_xlabel('Contamination Score Interval', fontsize=26, fontname='Times New Roman')
bar_plot.set_ylabel('Number of Samples', fontsize=26, fontname='Times New Roman')

# 调整X轴和Y轴刻度的字号和加粗
plt.xticks(fontsize=22, fontname='Times New Roman')
plt.yticks(fontsize=22, fontname='Times New Roman')

# 在柱子上显示数值，并设置字体大小
for i, v in enumerate(df['Number of Samples']):
    bar_plot.text(i, v + 30, f"{v}", color='black', fontsize=22, ha='center')

# 去除图中的横线
bar_plot.grid(False)

# 保存图像为PDF格式，并设置dpi参数
output_path = '/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/results/math_distribution.pdf'
plt.savefig(output_path, format='pdf', dpi=400)