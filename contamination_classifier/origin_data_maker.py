import os
import os.path
import pickle
import sys
sys.path.append('..')
import numpy as np
import hydra
from easyeditor import (
    MENDHyperParams,
    DINMHyperParams,
    )
from easyeditor import data_maker
from easyeditor import DINMHyperParams, MENDTrainingHparams
from easyeditor import ContaminationDataset
import torch
import json
from tqdm import tqdm
import statistics
from easyeditor import n_gram_entropy

import argparse

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input_data = self.inputs[index].cpu()  # 将输入数据移动到CPU
        target = self.targets[index]
        return input_data, target


def test_DINM(edit_data_all, editor, hparams, detoxify_metric, output_dir, split):
    print(f"make dataset for {split} split")
    input_list = []  # 用于存储特征向量的列表
    labels_list = []  # 用于存储标签的列表
    
    for data in tqdm(edit_data_all):
        edit_data = [data,]
        case_id = [edit_data_['case_id'] for edit_data_ in edit_data]
        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        prompts_with_systemPrompt = [edit_data_['prompt'] + ' ' + hparams.suffix_system_prompt for edit_data_ in edit_data]
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        ground_truth = [edit_data_['ground_truth'] for edit_data_ in edit_data]
                
        if split == "train":
            token_hidden_vector = editor.edit(
                vanilla = False,
                is_train = True,
                contamination_flag = True,
                prompts=prompts,
                prompts_with_systemPrompt = prompts_with_systemPrompt,
                target_new=target_new,
                ground_truth=ground_truth,
                keep_original_weight=True,
            )
        
            label = 1
        
            input_list.append(token_hidden_vector.view(-1))
            labels_list.append(label)
        
            token_hidden_vector = editor.edit(
                vanilla = False,
                is_train  = True,
                contamination_flag = False,
                prompts=prompts,
                prompts_with_systemPrompt = prompts_with_systemPrompt,
                target_new=target_new,
                ground_truth=ground_truth,
                keep_original_weight=True,
            )
        
            label = 0
        
            input_list.append(token_hidden_vector.view(-1))
            labels_list.append(label)
        
        else :
            
            token_hidden_vector = editor.edit(
                vanilla = True,
                is_train = False,
                contamination_flag = True,
                prompts=prompts,
                prompts_with_systemPrompt = prompts_with_systemPrompt,
                target_new=target_new,
                ground_truth=ground_truth,
                keep_original_weight=True,
            )
        
            label = 0
        
            input_list.append(token_hidden_vector.view(-1))
            labels_list.append(label)
            
            
            token_hidden_vector = editor.edit(
                vanilla = False,
                is_train = False,
                contamination_flag = True,
                prompts=prompts,
                prompts_with_systemPrompt = prompts_with_systemPrompt,
                target_new=target_new,
                ground_truth=ground_truth,
                keep_original_weight=True,
            )
        
            label = 1
        
            input_list.append(token_hidden_vector.view(-1))
            labels_list.append(label)
            
    dataset = CustomDataset(input_list, labels_list)
    #data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    if os.path.exists(f'gsm-hard_{split}_dataset.pkl') and os.path.getsize(f'gsm-hard_{split}_dataset.pkl') > 0:
        with open(f'gsm-hard_{split}_dataset.pkl', 'rb') as f:
            original_dataset = pickle.load(f)
    else:
        print(f"No gsm-hard_{split}_dataset exist, We will make one")
        original_dataset = CustomDataset([], [])

    combined_dataset = original_dataset + dataset
    
    data_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True)
    
    count = 0
    
    for features, labels in tqdm(data_loader):
        count = count + 1
    
    print(f"zkj debug {count}")
    
    with open(f'gsm-hard_{split}_dataset.pkl', 'wb') as f:
        pickle.dump(combined_dataset, f)
        
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edited_model', required=True, type=str) ## vanilla LLM
    parser.add_argument('--editing_method', required=True, type=str)  
    parser.add_argument('--hparams_dir', required=True, type=str)  
    parser.add_argument('--safety_classifier_dir', required=True, type=str) 
    parser.add_argument('--data_dir', default='../data', type=str)
    parser.add_argument('--metrics_save_dir', default='../safety_results', type=str)
    parser.add_argument('--split', default='test', type=str)
    
    args = parser.parse_args()

    if args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'DINM':
        editing_hparams = DINMHyperParams
    else:
        raise NotImplementedError
    output_dir = f'{args.metrics_save_dir}/{args.editing_method}_{args.edited_model}.json'

    detoxify_metric = ["DS", "DG_onlyQ", "DG_otherA", "DG_otherQ", "DG_otherAQ"]

    edit_data_all = ContaminationDataset(f'{args.data_dir}/gsm-hard_Contamination.json')
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    
    sample_edit_data_all = edit_data_all[0:2]
    #print(f"ZKJ edit_data_all debug : {sample_edit_data_all}")
    editor = data_maker.from_hparams(hparams)
    
    if args.editing_method == "DINM":
        overall_performance = test_DINM(edit_data_all, editor, hparams, detoxify_metric, output_dir, args.split)
    else:
        print("This method is currently not supported")

