import os
import os.path
import sys
sys.path.append('..')
import numpy as np
from easyeditor import (
    MENDHyperParams,
    DICEHyperParams,
    )
from easyeditor import con_vs_uncon_locate
from easyeditor import DICEHyperParams, MENDTrainingHparams
from easyeditor import ContaminationDataset
from easyeditor import EditTrainer
from sentence_transformers import SentenceTransformer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
import json
from tqdm import tqdm
import statistics
from easyeditor import n_gram_entropy

import argparse



def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data      


def write_json(path, data, case_id = None, data_all = None):
    if data_all is None:
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)
    else:
        with open(path, 'a') as file:
            if case_id[0] == 0:
                file.write("[")
            json.dump(data, file, indent=4)
            if case_id[-1] == data_all-1:
                file.write('\n')
                file.write("]")
            else:
                file.write(',')
                file.write('\n')
                file.flush()
def predict(sequences, model, tokenizer, batch_size = 100, cuda = None):
    predict = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i: i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(f"cuda:{cuda}")
        with torch.no_grad():
            outputs = model(**inputs)
            # Get predictions
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # If you want the most likely classes:
            _, predicted_classes = torch.max(predictions, dim=1)
            predict_label = predicted_classes.tolist()
            predict += predict_label
    return predict



def test_DINM(edit_data_all, editor, hparams):#safety_classifier_model, safety_classifier_tokenizer, 
    overall_performance = []
    
    #count = 0
    for data in tqdm(edit_data_all):
        #count = count + 1
        #print(count)
        edit_data = [data,]
        case_id = [edit_data_['case_id'] for edit_data_ in edit_data]
        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        prompts_with_systemPrompt = [edit_data_['prompt'] + ' ' + hparams.suffix_system_prompt for edit_data_ in edit_data]
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        ground_truth = [edit_data_['ground_truth'] for edit_data_ in edit_data]
        editor.edit(#metrics, edited_model, _ = 
            case_id = case_id,
            prompts=prompts,
            prompts_with_systemPrompt = prompts_with_systemPrompt,
            target_new=target_new,
            ground_truth=ground_truth,
            keep_original_weight=True,
        )
        
    return overall_performance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edited_model', required=True, type=str) ## vanilla LLM
    parser.add_argument('--hparams_dir', required=True, type=str)  
    parser.add_argument('--data_dir', default='../data', type=str)

    args = parser.parse_args()

    editing_hparams = DICEHyperParams
    overall_performance_avg = {
        "pre": {},
        "post": {}
    }
    
    edit_data_all = ContaminationDataset(f'{args.data_dir}/math_seen_Contamination.json')
    hparams = editing_hparams.from_hparams(args.hparams_dir)

    # classifier
    #safety_classifier_model = RobertaForSequenceClassification.from_pretrained(args.safety_classifier_dir).to(f"cuda:{hparams.device}")
    #safety_classifier_tokenizer = RobertaTokenizer.from_pretrained(args.safety_classifier_dir)
    sample_edit_data_all = edit_data_all[0:1]
    #print(f"ZKJ edit_data_all debug : {sample_edit_data_all}")
    editor = con_vs_uncon_locate.from_hparams(hparams)
    
    overall_performance = test_DINM(edit_data_all, editor, hparams) #safety_classifier_model, safety_classifier_tokenizer, 
    







    
