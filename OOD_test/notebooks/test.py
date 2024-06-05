import pandas as pd

# 创建两个DataFrame
data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
data2 = {'A': [7, 8, 9], 'B': [10, 11, 12]}
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# 按行拼接两个DataFrame
result = pd.concat([df1, df2], axis=0, ignore_index=True)  # ignore_index=True表示重新生成索引
print(result)
