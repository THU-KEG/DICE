from contamination import GSM8K, GSM_HARD, MAWPS, asdiv, SVAMP
import pandas as pd
import numpy as np
import copy
import os
import warnings
warnings.filterwarnings('ignore')
        
def get_performance(model_name, string):
    basedir = '/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/performance_vs_score'
    new_item = {}
    for task in [GSM8K(), GSM_HARD(), SVAMP(), asdiv(), MAWPS()]:
        dataset_name = task.dataset_name
        with open(f'{basedir}/{model_name}/{dataset_name}/{string}.log', 'r') as file:
            lines = file.readlines()

        values = [float(line.strip()) for line in lines]
        DICE_score = sum(values) / len(values)

        performance_df = pd.read_csv(f'{basedir}/{model_name}/{dataset_name}/{string}/generated_0.csv')
        performance_score = task.compute_performance(performance_df)['score'].mean() * 100
        line_name = f'{dataset_name}_performance'
        new_item[line_name] = performance_score
        line_name = f'{dataset_name}_DICE'
        new_item[line_name] = DICE_score
        
    
    return new_item
    
if __name__ == '__main__':
    for model in ['meta-llama/Llama-2-7b-hf']:#, 'gpt2-xl', 'mistralai/Mistral-7B-v0.1'
        df = pd.DataFrame(columns=['GSM8K_performance', 'GSM8K_DICE', 'GSM-hard_performance', 'GSM-hard_DICE', 'SVAMP_performance', 'SVAMP_DICE', 'ASDiv_performance', 'ASDiv_DICE', 'MAWPS_performance', 'MAWPS_DICE'])
        print(model)
        for string in ['vanilla', 'orca', 'epoch1_Evasive', 'epoch1_open', 'epoch5_Evasive', 'epoch5_open']:
            print(string)
            item = get_performance(model, string)
            print (item)
            df = df.append(item, ignore_index=True)
            print('------------------------------------------------------------------')
        csv_filename = 'performance_table.csv'
        df.to_csv(csv_filename, index=False)
