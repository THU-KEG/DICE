import re
import matplotlib.pyplot as plt
import os

# 读取日志文件
with open('phi-2.log', 'r') as file:
    logs = file.readlines()

# 提取每个 x 值
pattern = r'ZKJ Debug the max-distance contaminated vs uncontaminated layer is: (\d+)'

#pattern = r'INFO - the located layer of the 0th request is :  \[(\d+)\]'

x_values = [re.search(pattern, log).group(1) for log in logs if re.search(pattern, log)]

# 统计每个 x 值的出现频数
x_counts = {}
for x in x_values:
    x_shifted = int(x)
    x_counts[x_shifted] = x_counts.get(x_shifted, 0) + 1

# 将x_values转换为整数
x_values = [int(x) for x in x_counts.keys()]

# 绘制柱状图
plt.bar(x_values, x_counts.values(), align='center', alpha=0.7)
plt.xlabel('Located Contamination Layer')
plt.ylabel('Frequency')
plt.title('Frequency of Located Contamination Layer in Logs')
plt.xticks(x_values)  # 设置x轴刻度为整数
plt.grid(axis='y')    # 添加y轴网格线

# 保存图片
output_dir = 'contamination_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'phi2.png')
plt.savefig(output_path)

plt.show()
