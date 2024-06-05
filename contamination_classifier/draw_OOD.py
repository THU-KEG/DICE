import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取 .log 文件中的数据
file_path1 = 'math_Test.log'
file_path2 = 'GSM8K.log'

# 读取并处理math_Test.log文件
with open(file_path1, 'r') as file:
    data_str1 = file.read().strip()
    data_str1 = data_str1.strip('[]')  # 去除开头和结尾的方括号
    data1 = np.array([float(x) for x in data_str1.split(',')])  # 将字符串转换为浮点数列表

# 读取并处理GSM8K.log文件
with open(file_path2, 'r') as file:
    data_str2 = file.read().strip()
    data2 = np.array([float(x) for x in data_str2.split()])  # 按换行符拆分并转换为浮点数列表

# 统计每个区间的个数
bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
counts1, _ = np.histogram(data1, bins)
counts2, _ = np.histogram(data2, bins)

# 创建DataFrame
df = pd.DataFrame({
    'Contamination Score Interval': ['0-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.0.7', '0.7-1.0'],
    'Math Test': counts1,
    'GSM8K': counts2
})

# 设置绘图风格，不使用白色网格背景
sns.set(style="white")

# 定义马卡龙色系调色板的第一个和第二个颜色
macaron_color1 = sns.color_palette("pastel")[0]
macaron_color2 = sns.color_palette("pastel")[1]

# 创建柱状图
plt.figure(figsize=(11, 8))

# 创建柱状图，分别设置每个柱子颜色，并调整柱子宽度
bar_width = 0.35  # 柱子的宽度
index = np.arange(len(df['Contamination Score Interval']))  # X轴刻度的位置

bar1 = plt.bar(index, df['Math Test'], bar_width, label='Math', color=macaron_color1)
bar2 = plt.bar(index + bar_width, df['GSM8K'], bar_width, label='GSM8K', color=macaron_color2)

# 设置Y轴范围从0到最大计数值
plt.ylim(0, max(counts1.max(), counts2.max()) + 400)

# 添加标题和标签
plt.xlabel('Contamination Score Interval', fontsize=26, fontname='Times New Roman')
plt.ylabel('Number of Samples', fontsize=26, fontname='Times New Roman')

# 调整X轴和Y轴刻度的字号和加粗
plt.xticks(index + bar_width / 2, df['Contamination Score Interval'], fontsize=22, fontname='Times New Roman')
plt.yticks(fontsize=22, fontname='Times New Roman')

# 在柱子上显示数值，并设置字体大小
for i, v in enumerate(df['Math Test']):
    if v != 0:
        plt.text(i , v + 50, f"{v}", color='black', fontsize=22, ha='center') #- bar_width / 2
for i, v in enumerate(df['GSM8K']):
    if v != 0:
        plt.text(i + bar_width , v + 50, f"{v}", color='black', fontsize=22, ha='center') # / 2

# 去除图中的横线
plt.grid(False)

# 添加图例
plt.legend(fontsize=22, loc='upper right')

# 保存图像为PDF格式，并设置dpi参数
output_path = '/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/results/Combined.pdf'
plt.savefig(output_path, format='pdf', dpi=400)

plt.show()
