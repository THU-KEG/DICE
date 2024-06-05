import os
import os.path
import sys
sys.path.append('..')
import numpy as np
from easyeditor import (
    MENDHyperParams,
    DINMHyperParams,
    )
from easyeditor import con_vs_uncon_locate
from easyeditor import DINMHyperParams, MENDTrainingHparams
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

def evaluate_safety(item, model, tokenizer, detoxify_metric, cuda = None):
    # Evaluate whether the response generated by LLM is safe. 
    # If the response is safe, the value is 0. If the response is toxic, the value is 1. If the reply is NULL, the default neutral value is 0.5.
    label_name = {
        "0": "safe",
        "0.5": "nuetral",
        "1":"unsafe"
    }
    # curent generations from vanilla LLM and edited LLM
    detoxify_performance = {
        "pre": {},
        "post": {}
    }
    temp_batch = []
    temp_batch = [item[eval][key_name] for eval in ["pre","post"] for key_name in item[eval].keys()]
    
    # detoxification performance
    temp_predict = predict(temp_batch, model, tokenizer, batch_size = len(temp_batch), cuda = cuda)
    final_predict = [value if len(temp_batch[index]) > 0 else 0.5 for index, value in enumerate(temp_predict)]
    # fluency
    n_gram = [n_gram_entropy(temp_batch[t*5:(t+1)*5]) for t in range(2)]  #n_gram_entropy() return float value

    for i, eval in enumerate(["pre", "post"]):
        for j, metric_name in enumerate(detoxify_metric):
            detoxify_performance[eval][metric_name] = {
                "response": item[eval][metric_name],
                "label": label_name[str(final_predict[i*5+j])]
            }
        detoxify_performance[eval]["fluency"] = n_gram[i]

    item_evaluate={
                "case_id": item["case_id"],
                "requested_rewrite": item["requested_rewrite"],
                "vanilla_LLM": detoxify_performance["pre"],
                "edited_LLM": detoxify_performance["post"],
                "time": item["time"]
                    }
    return item_evaluate, final_predict + n_gram



def test_DINM(edit_data_all, editor, hparams, detoxify_metric, output_dir):#safety_classifier_model, safety_classifier_tokenizer, 
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
        
        #for item in metrics:
        #    item_evaluate,  evaluate_value = evaluate_safety(item, safety_classifier_model, safety_classifier_tokenizer, detoxify_metric, cuda = hparams.device)
        #    write_json(f'{output_dir}', item_evaluate, case_id = case_id, data_all = len(edit_data_all))
        #    overall_performance.append(evaluate_value)
    return overall_performance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edited_model', required=True, type=str) ## vanilla LLM
    parser.add_argument('--editing_method', required=True, type=str)  
    parser.add_argument('--hparams_dir', required=True, type=str)  
    #parser.add_argument('--safety_classifier_dir', required=True, type=str) 
    parser.add_argument('--data_dir', default='../data', type=str)
    parser.add_argument('--metrics_save_dir', default='../safety_results', type=str)

    args = parser.parse_args()

    if args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'DINM':
        editing_hparams = DINMHyperParams
    else:
        raise NotImplementedError
    output_dir = f'{args.metrics_save_dir}/{args.editing_method}_{args.edited_model}.json'
    #### some variables used for statistical results 
    if not os.path.exists(args.metrics_save_dir):
        os.mkdir(args.metrics_save_dir)
    print(f"Results will be stored at {output_dir}")
    overall_performance_avg = {
        "pre": {},
        "post": {}
    }
    
    detoxify_metric = ["DS", "DG_onlyQ", "DG_otherA", "DG_otherQ", "DG_otherAQ"]

    edit_data_all = ContaminationDataset(f'{args.data_dir}/math_seen_Contamination.json')
    hparams = editing_hparams.from_hparams(args.hparams_dir)

    # classifier
    #safety_classifier_model = RobertaForSequenceClassification.from_pretrained(args.safety_classifier_dir).to(f"cuda:{hparams.device}")
    #safety_classifier_tokenizer = RobertaTokenizer.from_pretrained(args.safety_classifier_dir)
    sample_edit_data_all = edit_data_all[0:1]
    #print(f"ZKJ edit_data_all debug : {sample_edit_data_all}")
    editor = con_vs_uncon_locate.from_hparams(hparams)
    
    if args.editing_method == "DINM":
        overall_performance = test_DINM(edit_data_all, editor, hparams, detoxify_metric, output_dir) #safety_classifier_model, safety_classifier_tokenizer, 
    else:
        print("This method is currently not supported")
        








    
