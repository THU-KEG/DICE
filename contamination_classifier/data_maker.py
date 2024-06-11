import os
import os.path
import pickle
import sys
sys.path.append('..')
import numpy as np
import hydra
from easyeditor import (
    MENDHyperParams,
    DICEHyperParams,
    )
from easyeditor import test_data_maker
from easyeditor import DICEHyperParams, MENDTrainingHparams
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


def test_DICE(edit_data_all, editor, hparams, model, test_dataset, is_contaminated, model_type, contaminated_type, epochs):
    print(f"make dataset for test dataset {test_dataset}")
    input_list = []  # 用于存储特征向量的列表
    labels_list = []  # 用于存储标签的列表
    
    for data in tqdm(edit_data_all):
        edit_data = [data,]
        case_id = [edit_data_['case_id'] for edit_data_ in edit_data]
        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        prompts_with_systemPrompt = [edit_data_['prompt'] + ' ' + hparams.suffix_system_prompt for edit_data_ in edit_data]
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        ground_truth = [edit_data_['ground_truth'] for edit_data_ in edit_data]
                
        token_hidden_vector = editor.edit(
            model_type = model_type,
            contaminated_type = contaminated_type,
            contamination_flag = is_contaminated,
            prompts=prompts,
            prompts_with_systemPrompt = prompts_with_systemPrompt,
            target_new=target_new,
            ground_truth=ground_truth,
            keep_original_weight=True,
        )
        if is_contaminated == True:
           label = 1
        else :
            label = 0
        #breakpoint()
        input_list.append(token_hidden_vector.view(-1))
        labels_list.append(label)
        
            
    dataset = CustomDataset(input_list, labels_list)
    #data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    #filename = f'{test_dataset}_{model_type}_{contaminated_type}_dataset.pkl'
    
    filename = f'Test_{contaminated_type}_contaminated_vs_{model_type}_on_{test_dataset}_dataset.pkl'
    #filename = f'test_data_for_layer_29.pkl'
#    if is_contaminated == True:
#    filename = f'topk_{topk}_hidden_states.pkl'
#    else:
    # if is_contaminated == True:
    #     if epochs == "epochs_1/":
    #         filename = f'Test_epoch1_{contaminated_type}_contaminated_on_{test_dataset}.pkl'
    #     else :
    #         filename = f'Test_epoch5_{contaminated_type}_contaminated_on_{test_dataset}.pkl'
    # else :
    #     filename = f'Test_{model_type}_on_{test_dataset}.pkl'
    # filename = f"{test_dataset}.pkl"
    base_dir = f"./DICE_data"
    print(f"ZKJ debug the classifier test set dataset name is : {filename}")
    os.makedirs(base_dir, exist_ok=True)

    #base_dir = "/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/performance_vs_score/meta-llama/Llama-2-7b-hf"
    #{base_dir}/{test_dataset}/    {base_dir}/{test_dataset}/
    if os.path.exists(f'{base_dir}/{filename}') and os.path.getsize(f'{base_dir}/{filename}') > 0:
        with open(f'{base_dir}/{filename}', 'rb') as f:
            original_dataset = pickle.load(f)
    else:
        print(f"No {base_dir}/{filename} exist, We will make one")
        original_dataset = CustomDataset([], [])

    combined_dataset = original_dataset + dataset
    
    data_loader = DataLoader(combined_dataset, batch_size=1, shuffle=True)
    
    count = 0
    
    for features, labels in tqdm(data_loader):
        count = count + 1
    
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>zkj debug test dataset ({filename}) has {count} samples<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #{base_dir}/{test_dataset}/
    with open(f'{base_dir}/{filename}', 'wb') as f:
        pickle.dump(combined_dataset, f)
        
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edited_model', required=True, type=str) ## vanilla LLM
    parser.add_argument('--hparams_dir', required=True, type=str)  
    parser.add_argument('--data_dir', default='../data', type=str)
    parser.add_argument('--test_dataset', default='GSM8K_seen', type=str)
    parser.add_argument('--is_contaminated', default=False, type=bool)
    parser.add_argument('--model_type', default='vanilla', type=str)
    parser.add_argument('--contaminated_type', default='open', type=str)
    parser.add_argument('--epochs', default='', type=str)

    args = parser.parse_args()

    print(f"zkj debug run python : python data_maker.py --test_dataset={args.test_dataset} --is_contaminated={args.is_contaminated} --model_type={args.model_type} --contaminated_type={args.contaminated_type}")
    
    args = parser.parse_args()

    editing_hparams = DICEHyperParams

    edit_data_all = ContaminationDataset(f'{args.data_dir}/{args.test_dataset}_Contamination.json')
    hparams = editing_hparams.from_hparams(args.hparams_dir)
    
    sample_edit_data_all = edit_data_all[0:2]
    #print(f"ZKJ edit_data_all debug : {sample_edit_data_all}")
    
    editor = test_data_maker.from_hparams(hparams, args)
    print (hparams)
    editor.load_model(epochs = args.epochs, model_type = args.model_type, contaminated_type = args.contaminated_type, contamination_flag = args.is_contaminated, hparams = hparams)
    
    overall_performance = test_DICE(edit_data_all, editor, hparams, args.edited_model, args.test_dataset, args.is_contaminated, args.model_type, args.contaminated_type ,args.epochs)
    
