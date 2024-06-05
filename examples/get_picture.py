import re
import matplotlib.pyplot as plt
import os

# 读取日志文件
with open('GSM8K_logs.txt', 'r') as file:
    logs = file.read()

# 提取每个 x 值
pattern = r'INFO - the located layer of the 0th request is :  \[(\d+)\]'
x_values = re.findall(pattern, logs)

# 统计每个 x 值的出现频数
x_counts = {}
for x in x_values:
    x_counts[x] = x_counts.get(x, 0) + 1

# 绘制柱状图
plt.bar(x_counts.keys(), x_counts.values())
plt.xlabel('Located Contamination Layer')
plt.ylabel('Frequency')
plt.title('Frequency of Located Contamination Layer in Logs')

# 保存图片
output_dir = 'contamination_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'GSM8K.png')
plt.savefig(output_path)

plt.show()
