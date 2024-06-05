from contamination import GSM8K, GSM_HARD, MATH, ARC, TruthfulQA, asdiv, SVAMP, THEOREM_QA, MAWPS, TABMWP
import pandas as pd
import numpy as np
import copy
import os
import warnings
warnings.filterwarnings('ignore')


def get_performance(model_name, task, train_dataset, dataset_name, types=['', '/epochs_1']):
    baseline = pd.read_csv(f'../output/{model_name}/seed/0/{dataset_name}/generated_0.csv')
    #was_trained = pd.read_csv(f'../output/{model_name}/test/{dataset_name}/0/generated_0.csv')['was_trained'] #4
    #was_trained_2 = pd.read_csv(f'../output/{model_name}/test/{dataset_name}/2/generated_0.csv')['was_trained']
    baseline_score = task.compute_performance(baseline)['score'].mean() * 100#was_trained==True
    #baseline_score_contaminated_2 = task.compute_performance(baseline[was_trained_2==True])['score'].mean() * 100
    #baseline_score_uncontaminated = task.compute_performance(baseline[was_trained==False])['score'].mean() * 100#was_trained==False
    #baseline_score_uncontaminated_2 = task.compute_performance(baseline[was_trained_2==False])['score'].mean() * 100

    #baseline = pd.read_csv(f'../output/{model_name}/seed/0/{dataset_name}/generated_4.csv')
   # baseline = task.compute_performance(baseline[was_trained == True])
    #baseline_score_rephrase = baseline['score'].mean() * 100

    folder = lambda dataset_name, string, index, data_index=0: f'../output/{model_name}/{train_dataset}_base/test/{dataset_name}{string}/{index}/generated_{data_index}.csv'
    scores = []
    for string in types:
        score = {}
        for index in range(2):
            for data_index in [0]:#, 4
                try:
                    test = pd.read_csv(folder(dataset_name, string, index, data_index))
                    test_results = task.compute_performance(test)
                    #test_score_uncontaminated = test[test['was_trained'] == False]['score'].mean() * 100
                    test_score = test_results['score'].mean() * 100
                except Exception as e:
                    print(e)
                    test_score_uncontaminated = np.nan
                    test_score_contaminated = np.nan
                #score[f'test_{index}_score_uncontaminated_{data_index}'] = test_score_uncontaminated
                
                score[f'test_{index}_score_{data_index}'] = test_score

        scores.append(score)

    table1_scores = f'& {baseline_score}      & {scores[1]["test_0_score_0"]}      & {scores[0]["test_0_score_0"]}      & {scores[1]["test_1_score_0"]}      & {scores[0]["test_1_score_0"]}'
    return {
        'table_1': table1_scores,
    }
    
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_name', type=str, default='gsm8k')
    args = parser.parse_args()
    for model in ['meta-llama/Llama-2-7b-hf', 'microsoft/phi-2']:#, 'gpt2-xl', 'mistralai/Mistral-7B-v0.1'
        print(model)
        for task in [GSM_HARD(), SVAMP(), MAWPS(), asdiv(), TABMWP(), MATH()]:#
            print(task.dataset_name)
            performance = get_performance(model, task, args.train_dataset_name, task.dataset_name)
            for key, value in performance.items():
                #print(key)
                print(value)
            print('-----------------')