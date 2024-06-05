from contamination import GSM8K, MMLU, ARC, TruthfulQA
import pandas as pd
import numpy as np
import copy
import os
import warnings
warnings.filterwarnings('ignore')

def extract_kim_file(filename):
    # read the third line and split at :
    with open(filename, 'r') as f:
        lines = f.readlines()
        line = lines[1]   # 2
        line = line.split(':')
        return float(line[1].strip())
def extract_kim(model_name, dataset_name, dataset_name_alternative):
    test_name = 'test'
    folder_name = lambda setting, epochs, index: f'{model_name.replace("/", "-")}_{dataset_name}_{setting}{"-" + dataset_name_alternative if setting != "seed" else ""}{epochs}-{index}'

    baseline = extract_kim_file(os.path.join('../code-contamination-output', folder_name('seed', '', '0'), 'log.txt'))
    test_malicious = extract_kim_file(os.path.join('../code-contamination-output', folder_name(test_name, '', '0'), 'log.txt'))
    rephrase_malicious = extract_kim_file(os.path.join('../code-contamination-output', folder_name(test_name, '', '1'), 'log.txt'))
    test_negligent = extract_kim_file(os.path.join('../code-contamination-output', folder_name(test_name, '-epochs_1', '0'), 'log.txt'))
    rephrase_negligent = extract_kim_file(os.path.join('../code-contamination-output', folder_name(test_name, '-epochs_1', '1'), 'log.txt'))
    table = f'{dataset_name_alternative} & {baseline}  & {test_negligent} & {rephrase_negligent} & {test_malicious} & {rephrase_malicious}'
    return table


if __name__ == '__main__':
    for model in ['microsoft/phi-2']:#, 'gpt2-xl', 'mistralai/Mistral-7B-v0.1'
        print(model)
        print(extract_kim(model, 'gsm8k', 'gsm8k'))
        #print(extract_kim(model, 'truthful_qa', 'truthfulqa'))
        #print(extract_kim(model, 'cais/mmlu', 'mmlu'))
        #print(extract_kim(model, 'ai2_arc', 'arc'))
        print('-----------------')

#extract_kim('mistralai/Mistral-7B-v0.1', 'gsm8k', 'gsm8k')