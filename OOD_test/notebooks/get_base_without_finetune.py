from contamination import GSM8K, GSM_HARD, SVAMP, MAWPS, asdiv, TABMWP, MATH
import pandas as pd
import numpy as np
import copy
import os
import warnings
warnings.filterwarnings('ignore')

def get_performance(model_name, task, dataset_name): # , types=['','/epochs_1']
    baseline = pd.read_csv(f'../output/base_without_finetune/{model_name}/test/gsm8k/generated_0.csv')
    was_trained = pd.read_csv(f'../output/base_without_finetune/{model_name}/test/gsm8k/generated_0.csv')['was_trained'] #4
    #was_trained = pd.read_csv(f'../output/{model_name}/{dataset_name}_base/test/{dataset_name}/epochs_1/0/generated_0.csv')['was_trained'] #4
    baseline_score_contaminated = GSM8K().compute_performance(baseline[was_trained==True])['score'].mean() * 100#was_trained==True
    baseline_score_uncontaminated = GSM8K().compute_performance(baseline[was_trained==False])['score'].mean() * 100#was_trained==False

    folder = lambda dataset_name, data_index=0: f'../output/base_without_finetune/{model_name}/test/{dataset_name}/generated_{data_index}.csv'
    scores = []
    for data_index in [0]:#, 4
        score = {}
        try:
            #breakpoint()
            test = pd.read_csv(folder(dataset_name, data_index))
            
            test_results = task.compute_performance(test)
            #test_score_uncontaminated = test[test['was_trained'] == False]['score'].mean() * 100
            test_score = test_results['score'].mean() * 100
            
        except Exception as e:
            print(e)
            
            test_score_uncontaminated = np.nan
            test_score_contaminated = np.nan
            #score[f'test_{index}_score_uncontaminated_{data_index}'] = test_score_uncontaminated
        
        score[f'test_score_{data_index}'] = test_score
    
    scores.append(score)
    table1_scores = f'{baseline_score_contaminated} & {baseline_score_uncontaminated} & {scores[0]["test_score_0"]}'
    return {
        'table_1': table1_scores,
    }
    
    
if __name__ == '__main__':
    for model in ['meta-llama/Llama-2-7b-hf']:#, 'gpt2-xl', 'mistralai/Mistral-7B-v0.1' 'microsoft/phi-2'
        print(model)
        for task in [GSM_HARD(), SVAMP(), MAWPS(), asdiv(), TABMWP(), MATH()]:#
            print(task.dataset_name)
            performance = get_performance(model, task, task.dataset_name)
            for key, value in performance.items():
                print(key)
                print(value)
            print('-----------------')