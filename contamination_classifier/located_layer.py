import re
from collections import defaultdict

# 用于存储每层的distance值
layer_distances = defaultdict(list)

# 定义需要统计的层数
layers_to_analyze = [4, 9, 19, 29, 31, 32]

# 正则表达式匹配需要的行
pattern = re.compile(r'The distance of (\d+)th Layer is: ([\d.]+)')

# 读取log.txt文件并提取数据
with open('/data1/tsq/zkj_use/data_contamination/EasyEdit/examples/phi-2.log', 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            layer = int(match.group(1))
            if layer in layers_to_analyze:
                distance = float(match.group(2))
                layer_distances[layer].append(distance)

# 计算每层的样本数量和平均distance
results = {}
for layer in layers_to_analyze:
    distances = layer_distances[layer]
    sample_count = len(distances)
    average_distance = sum(distances) / sample_count if sample_count > 0 else 0
    results[layer] = {'sample_count': sample_count, 'average_distance': average_distance}

# 输出结果
for layer, data in results.items():
    print(f'Layer {layer + 1}:')
    print(f'  Sample count: {data["sample_count"]}')
    print(f'  Average distance: {data["average_distance"]:.6f}')
