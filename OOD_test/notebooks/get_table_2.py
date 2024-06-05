from contamination import GSM8K, MMLU, ARC, TruthfulQA
import pandas as pd
import numpy as np
import copy
import os
import warnings
warnings.filterwarnings('ignore')

def sample_level_methods(df, df_reference):
    output_dict = dict()
    output_dict['shi'] = df['topkmin']
    output_dict['mireshgallah'] = - df['perplexity_output'] / df_reference['perplexity_output']
    output_dict['yeom'] = - df['perplexity_output']
    output_dict['carlini'] = - df['lowercase']
    output_dict['rouge'] = df['rouge']
    return output_dict

def compute_tpr(scores, was_trained, fpr=0.01, method='yeom'):
    # compute the threshold
    false_scores = scores[was_trained == False]
    true_scores = scores[was_trained == True]
    false_scores = np.sort(false_scores)
    threshold = false_scores[int(len(false_scores) * (1-fpr))]
    # compute the tpr
    tpr = (true_scores > threshold).mean()
    return tpr

def detect(model_name, train_dataset_name, dataset_name, type='v1'):
    folder = lambda dataset_name, string, index, data_index=0: f'../output/{model_name}/{train_dataset_name}_base/test/{dataset_name}{string}/{index}/generated_{data_index}.csv'
    if type == 'v2':
        folder = lambda dataset_name, string, index, data_index=0: f'../output/{model_name}/testv2{string}/{index}/{dataset_name}/generated_{data_index}.csv'
    df_reference = pd.read_csv(f'../output/{model_name}/seed/0/{dataset_name}/generated_0.csv')
    was_trained = pd.read_csv(folder(dataset_name, '', 0, 0))['was_trained']
    scores_reference = sample_level_methods(df_reference, df_reference)
    tpr_ref = {}
    for name in scores_reference:
        tpr_ref[name] = compute_tpr(np.array(scores_reference[name]), np.array(was_trained), method=name)
    results_all = []
    for epochs in ['', '/epochs_1']:
        # trained on actual samples
        df = pd.read_csv(folder(dataset_name, epochs, 0, 0))
        scores = sample_level_methods(df, df_reference)
        was_trained = df['was_trained']
        tpr = {}
        for name in scores:
            tpr[name] = compute_tpr(np.array(scores[name]), np.array(was_trained), method=name)

        # trained on rephrased samples
        df = pd.read_csv(folder(dataset_name, epochs, 1, 0))
        scores = sample_level_methods(df, df_reference)
        was_trained = df['was_trained']
        tpr_rephrased = {}
        for name in scores:
            tpr_rephrased[name] = compute_tpr(np.array(scores[name]), np.array(was_trained), method=name)
        results_all.append((tpr.copy(), tpr_rephrased))

    return results_all, [(tpr_ref, tpr_ref)]

def compute_average_performance(performances):
    average_performances_over_datasets = copy.deepcopy(performances[0])
    for performance_dataset in performances[1:]:
        for i in range(len(performance_dataset)):
            for j in range(len(performance_dataset[i])):
                for name in performance_dataset[i][j]:
                    average_performances_over_datasets[i][j][name] += performance_dataset[i][j][name]

    for i in range(len(average_performances_over_datasets)):
        for j in range(len(average_performances_over_datasets[i])):
            for name in average_performances_over_datasets[i][j]:
                average_performances_over_datasets[i][j][name] /= len(performances) / 100
    return average_performances_over_datasets
    
if __name__ == '__main__':
    for model_name in ['microsoft/phi-2']:#, 'gpt2-xl', 'mistralai/Mistral-7B-v0.1'
        performances = detect(model_name, 'gsm8k', 'gsm8k')[0]
            #detect(model_name, 'mmlu')[0],
            #detect(model_name, 'arc')[0],
            #detect(model_name, 'truthfulqa')[0],
        #]
        print(model_name)
        #average_performance = compute_average_performance(performances)
        table = ''
        for method in performances[0][0]:
            table += f'{method} & {performances[1][0][method]} & {performances[1][1][method]} & {performances[0][0][method]} & {performances[0][1][method]} \\\\ \n'
        print(table)
        print('-----------------')
    