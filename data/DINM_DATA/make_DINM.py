import pandas as pd
import json

# 读取good.csv和bad.csv文件

#good_df = pd.read_csv("/data1/tsq/zkj_use/data_contamination/EasyEdit/data/DINM_DATA/gsm8k/Llama_Base_good_generation.csv")
#bad_df = pd.read_csv("/data1/tsq/zkj_use/data_contamination/EasyEdit/data/DINM_DATA/gsm8k/Llama_Base_bad_generation.csv")

# 合并两个DataFrame，根据索引匹配
#merged_df = pd.merge(good_df, bad_df, left_index=True, right_index=True)

merged_df = pd.read_csv("/data1/tsq/zkj_use/data_contamination/EasyEdit/data/DINM_DATA/MAWPS/original.csv")

def generation_prompt_template(input_):
    return f'### Input:\n{input_}\n\n### Response:\n'

# 定义一个空的列表，用于存储每个实例的数据
instances = []
#count = 0
# 遍历每一行数据，整理成json格式所需的字典，并添加到instances列表中
for i, row in merged_df.iterrows():
    #print(row["was_trained_x"])
    #print(row["was_trained_y"])
    # breakpoint()
    #print(i)
    if i<600:
        continue
    instance = {
        "id": i-600,
        "adversarial prompt": row["input"],
        "was_trained": False,
        "question": row["input"],
        "bad generation": "",
        "good generation": ""
    }
    instances.append(instance)

# 将instances列表写入JSON文件
with open("output.json", "w") as f:
    json.dump(instances, f, indent=4)
