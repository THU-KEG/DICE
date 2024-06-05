from contamination import GSM8K, MMLU, ARC, TruthfulQA
import pandas as pd
import numpy as np
import copy
import os
import warnings
warnings.filterwarnings('ignore')

def sample_level_methods(df):
    output_dict = dict()
    output_dict['Min-k-prob'] = df['topkmin']
    #output_dict['mireshgallah'] = - df['perplexity_output'] / df_reference['perplexity_output']
    #output_dict['yeom'] = - df['perplexity_output']
    #output_dict['lowercase'] = - df['lowercase']
    #output_dict['zlib'] = - df['zlib']
    #output_dict['rouge'] = df['rouge']
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


from sklearn.metrics import roc_auc_score

def compute_auroc(scores, true_labels, fpr=0.01, method='yeom'):
    # compute the threshold
    false_scores = scores[true_labels == False]
    true_scores = scores[true_labels == True]
    false_scores = np.sort(false_scores)
    threshold = false_scores[int(len(false_scores) * (1-fpr))]
    # compute the auc score
    auc_score = roc_auc_score(true_labels, scores)
    return auc_score

'''
def compute_auroc(scores, was_trained):
    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for different thresholds
    thresholds = np.unique(scores)
    num_neg = np.sum(was_trained == False)
    num_pos = np.sum(was_trained == True)
    tprs = []
    fprs = []
    for threshold in thresholds:
        tp = np.sum((scores >= threshold) & (was_trained == True))
        fp = np.sum((scores >= threshold) & (was_trained == False))
        tprs.append(tp / num_pos)
        fprs.append(fp / num_neg)
    
    # Sort TPRs and FPRs based on increasing FPR
    sorted_indices = np.argsort(fprs)
    sorted_fprs = np.array(fprs)[sorted_indices]
    sorted_tprs = np.array(tprs)[sorted_indices]
    
    # Calculate AUROC using trapezoidal rule
    auroc = np.trapz(sorted_tprs, sorted_fprs)
    return auroc
'''
def detect(model_name, dataset_name, model_type, contaminated_type):
    folder = lambda dataset_name, string, model_type, contaminated_type, data_index=0: f'../output/{model_name}/detect/{dataset_name}{string}/{contaminated_type}_{model_type}_detect_generated_{data_index}.csv'
    #df_reference = pd.read_csv(f'../output/{model_name}/seed/0/{dataset_name}/generated_0.csv')
    #was_trained = pd.read_csv(folder(dataset_name, '', 0, 0))['was_trained']
    #breakpoint()
    results_all = []
    for epochs in ['/epochs_1']:
        #print("ZKJ")
        # trained on actual samples
        #df = pd.read_csv("/data1/tsq/zkj_use/data_contamination/malicious-contamination/output/microsoft/phi-2/gsm8k_base/test/gsm8k/0/generated_0.csv")
        #df = pd.read_csv(folder(dataset_name, epochs, model_type, contaminated_type))
        if model_type=="Both" and contaminated_type == "Both":
            df = pd.read_csv(folder(dataset_name, epochs, "vanilla", "open"))
            tmp = pd.read_csv(folder(dataset_name, epochs, "vanilla", "Evasive"))
            df = pd.concat([df, tmp], axis=0, ignore_index=True)
            tmp = pd.read_csv(folder(dataset_name, epochs, "orca", "open"))
            df = pd.concat([df, tmp], axis=0, ignore_index=True)
            tmp = pd.read_csv(folder(dataset_name, epochs, "orca", "Evasive"))
            df = pd.concat([df, tmp], axis=0, ignore_index=True)
            
        elif model_type == "Both":
            df = pd.read_csv(folder(dataset_name, epochs, "orca", contaminated_type))
            tmp = pd.read_csv(folder(dataset_name, epochs, "vanilla", contaminated_type))
            df = pd.concat([df, tmp], axis=0, ignore_index=True)
        elif contaminated_type == "Both":
            df = pd.read_csv(folder(dataset_name, epochs, model_type, "open"))
            tmp = pd.read_csv(folder(dataset_name, epochs, model_type, "Evasive"))
            df = pd.concat([df, tmp], axis=0, ignore_index=True)
        else:
            df = pd.read_csv(folder(dataset_name, epochs, model_type, contaminated_type))
            
        scores = sample_level_methods(df)
        was_trained = df['was_trained']
        tpr = {}
        auroc = {}
        #breakpoint()
        for name in scores:
            tpr[name] = compute_tpr(np.array(scores[name]), np.array(was_trained), method=name)
            auroc[name] = compute_auroc(np.array(scores[name]), np.array(was_trained))*100
            
        results_all.append(auroc.copy())
        #breakpoint()

    return results_all

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--dataset_name', type=str, default='GSM8K_seen')
    parser.add_argument('--model_type', default='vanilla', type=str)
    parser.add_argument('--contaminated_type', default='open', type=str)
    args = parser.parse_args()
    #for model_name in ['meta-llama/Llama-2-7b-hf']:#, 'gpt2-xl', 'mistralai/Mistral-7B-v0.1', 'microsoft/phi-2'
    performances = detect(args.model_name, args.dataset_name, args.model_type, args.contaminated_type)[0]
    #print(performances)
    table = ''
    for method in performances:
        table += f'& {performances[method]}    '
    print(table, end=" ")
#    print('-----------------')
    