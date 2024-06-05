import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    'Layer': [5, 10, 20, 30, 31, 32],
    'Average Distance': [243, 378, 581, 901, 1006, 191]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置绘图风格，不使用白色网格背景
sns.set(style="white")

# 定义马卡龙色系调色板的第一个颜色
macaron_color = sns.color_palette("pastel")[0]

# 创建柱状图
plt.figure(figsize=(11, 8))

# 创建柱状图，设置每个柱子颜色一致，并调整柱子宽度
bar_plot = sns.barplot(x='Layer', y='Average Distance', data=df, color=macaron_color, width=0.5)

# 缩小Y轴范围并增加Y轴刻度的间距
bar_plot.set_ylim(180, 1100)

# 添加标题和标签
bar_plot.set_xlabel('Layer Index', fontsize=26, fontname='Times New Roman')
bar_plot.set_ylabel('Average Distance', fontsize=26, fontname='Times New Roman')

# 调整X轴和Y轴刻度的字号
plt.xticks(fontsize=22, fontname='Times New Roman')
plt.yticks(fontsize=22, fontname='Times New Roman')

# 在较小范围内显示数值，并设置字体大小
for i, v in enumerate(data['Average Distance']):
    #if v > 400:
        bar_plot.text(i, v + 2, f"{v}", color='black', fontsize=22, ha='center')
    #else:
        # 为第32层的特殊数值添加注释，并设置字体大小
    #    bar_plot.text(i, 452, f"{v}", color='black', fontsize=22, ha='center')

# 去除图中的横线
bar_plot.grid(False)

# 获取最高值
max_value = max(data['Average Distance'])

# 添加虚线标记最高值
plt.axhline(y=max_value, color='r', linestyle='--', linewidth=3)

# 保存图像
output_path = '/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/results/phi2_average_distance_per_layer.pdf'
plt.savefig(output_path, format='pdf', dpi=400)