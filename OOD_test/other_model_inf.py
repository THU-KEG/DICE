import json
from contamination import load_model, LongestCommonSubstring, ROUGE, Perplexity, load_tokenizer, Lowercase, TopKMin
import pandas as pd
import numpy as np
from contamination import InstructionProcessor, FinetuneInstructions
import torch
import os
import datasets
from transformers import pipeline
from tqdm import tqdm
import re
import gc
from datasets import load_dataset
from transformers import set_seed
import torch.nn as nn
from typing import Iterable, Union, Any
from pathlib import Path
import datasets
from contamination import GSM8K, MMLU, TruthfulQA, ARC, get_max_length
import ast

set_seed(42)

def prompt_template(instruction, input_):
    if len(instruction) == 0:
        return f'### Input:\n{input_}\n\n### Response:\n'
    return f'### Instruction:\n{instruction}\n\n### Input:\n{input_}\n\n### Response:\n'

def generation_prompt_template(instruction, input_):
    return f'### Input:\n{input_}\n\n### Response:\n'

def generate(model, tokenizer, df, output_dir, batch_size, 
             n_test_trained=1000, max_tokens=256, filename='generated.csv', few_shot_samples=None, 
             prompt_template=generation_prompt_template, ref_model_name=None, check_batch_size=8, is_contaminated=None):
    set_seed(42)
    if os.path.isfile(os.path.join(output_dir, filename)):
        df = pd.read_csv(os.path.join(output_dir, filename))
    if few_shot_samples is None:
        few_shot = ''
    else:
        few_shot = '\n\n'.join([prompt_template(instruction, input_) + '\n' + output for instruction, input_, output in zip(few_shot_samples['instruction'], few_shot_samples['input'], few_shot_samples['answer'])])
        few_shot += '\n\n'
    print(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df['complete_inputs'] = [few_shot + prompt_template(instruction, input_) for instruction, input_ in zip(df['instruction'], df['input'])]
    generated_texts = []
    total_batches = int(np.ceil(len(df) / batch_size))
    max_length = get_max_length(model.config)

    if 'generated' not in df.columns:
        for i in tqdm(range(total_batches), desc="Generating Texts"):
            batch_start = i * batch_size
            batch_end = batch_start + batch_size
            batch_texts = df['complete_inputs'].iloc[batch_start:batch_end].tolist()
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length - max_tokens).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, num_return_sequences=1, 
                                    do_sample=False, temperature=1, top_p=1, eos_token_id=tokenizer.eos_token_id)
            
            for output, input_text in zip(outputs, batch_texts):
                text = tokenizer.decode(output, skip_special_tokens=True)
                generated_texts.append(text[len(input_text):])

    new_df = df.copy()
    if len(generated_texts) > 0:
        new_df['generated'] = generated_texts
    new_df['was_trained'] = False
    '''
    if is_contaminated is None:
        new_df['was_trained'][:n_test_trained] = True
    else:
        # the first n_test_trained samples for which llm_contaminator is False were trained
        new_df['was_trained'] = (is_contaminated == False)
        index_to_train = np.where(is_contaminated == False)[0][min(n_test_trained, np.sum(is_contaminated == False) - 1)]
        new_df['was_trained'][index_to_train:] = False
    if 'rouge' not in new_df.columns:
        rouge = ROUGE()
        new_df['rouge'] = new_df.apply(lambda row: rouge(row['answer'], row['generated']), axis=1)
        lcs = LongestCommonSubstring()
        new_df['lcs'] = new_df.apply(lambda row: lcs(row['answer'], row['generated']), axis=1)
    if 'perplexity' not in new_df.columns:
        perplexity = Perplexity(model, tokenizer)
        new_df['perplexity'] = perplexity.batch_call(new_df['generated'].tolist(), new_df['complete_inputs'].tolist(), batch_size=check_batch_size)
        new_df['perplexity_output'] = perplexity.batch_call(new_df['answer'].tolist(), new_df['complete_inputs'].tolist(), batch_size=check_batch_size)
        new_df['perplexity_input'] = perplexity.batch_call(new_df['input'].tolist(), batch_size=check_batch_size)
    if 'topkmin' not in new_df.columns:
        topkmin = TopKMin(model, tokenizer)
        new_df['topkmin'] = topkmin.batch_call(new_df['answer'].tolist(), new_df['complete_inputs'].tolist(), batch_size=check_batch_size)
        new_df['topkmin_all'] = topkmin.batch_call((new_df['complete_inputs'] + new_df['answer']).tolist(), batch_size=check_batch_size)
    if 'lowercase' not in new_df.columns:
        lowercase = Lowercase(model, tokenizer)
        new_df['lowercase'] = lowercase.batch_call(new_df['answer'].tolist(), new_df['complete_inputs'].tolist(), batch_size=check_batch_size)
    if 'perplexity_good' not in new_df.columns and 'correct_answers' in new_df.columns:
        perplexity = Perplexity(model, tokenizer)
        perplexity_good = []
        perplexity_bad = []
        all_good_answers, all_good_inputs = [], []
        all_bad_answers, all_bad_inputs = [], []
        for row in new_df[['correct_answers', 'complete_inputs']].values:
            lit_eval = ast.literal_eval(row[0])
            all_good_answers.extend(lit_eval)
            all_good_inputs.extend([row[1] for _ in range(len(lit_eval))])
        for row in new_df[['incorrect_answers', 'complete_inputs']].values:
            lit_eval = ast.literal_eval(row[0])
            all_bad_answers.extend(lit_eval)
            all_bad_inputs.extend([row[1] for _ in range(len(lit_eval))])
        batch_call_results = perplexity.batch_call(all_good_answers, all_good_inputs, batch_size=check_batch_size)
        current_point = 0
        for row in new_df[['correct_answers', 'complete_inputs']].values:
            lit_eval = ast.literal_eval(row[0])
            perplexity_good.append(min(batch_call_results[current_point:current_point + len(lit_eval)]))
            current_point += len(lit_eval)

        batch_call_results = perplexity.batch_call(all_bad_answers, all_bad_inputs, batch_size=check_batch_size)
        current_point = 0
        for row in new_df[['incorrect_answers', 'complete_inputs']].values:
            lit_eval = ast.literal_eval(row[0])
            perplexity_bad.append(min(batch_call_results[current_point:current_point + len(lit_eval)]))
            current_point += len(lit_eval)
        new_df['perplexity_good'] = perplexity_good
        new_df['perplexity_bad'] = perplexity_bad

    if ref_model_name is not None and 'perplexity_ref' not in new_df.columns:
        model_ref, tokenizer = load_model(ref_model_name, return_tokenizer=True)
        model_ref.eval()
        perplexity = Perplexity(model_ref, tokenizer)
        new_df['perplexity_ref'] = perplexity.batch_call(new_df['generated'].tolist(), new_df['complete_inputs'].tolist(), batch_size=check_batch_size)
        del model_ref, perplexity.model, topkmin.model
        gc.collect()
        torch.cuda.empty_cache()
    '''
    os.makedirs(output_dir, exist_ok=True)
    new_df.to_csv(os.path.join(output_dir, filename), index=False)
    
def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()
                
def main(model_name, total_samples, test_df, dataset_name, do_few_shot, generative_batch_size=1, 
         check_batch_size=8):

    '''
    orca = datasets.load_dataset("Open-Orca/OpenOrca", split="train")
    # select 100000 samples
    orca = orca.shuffle(seed=42).select(range(max(100000, total_samples)))
    df = pd.DataFrame(orca)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.rename(columns={'question': 'input', 'system_prompt': 'instruction', 'response': 'output'})
    df['instruction'].fillna('', inplace=True)
    path_to_config = 'configs/config_finetune.json'
    if os.path.isfile(f'configs/{model_name}.json'):
        path_to_config = f'configs/{model_name}.json'
    '''
    
    for seed in range(1):
        model = load_model(f'/data1/tsq/zkj_use/MODELS/{model_name}')
        tokenizer = load_tokenizer(f'/data1/tsq/zkj_use/MODELS/{model_name}')
        # model = load_model(f'output/{model_name}/seed/{seed}')
        # # model.merge_and_unload()
        # tokenizer = load_tokenizer(model_name)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        for j, test_df_ in enumerate([test_df]):
            if j > 0 and j not in [3, 4]:
                continue
            generate(model, tokenizer, test_df_, f'output/{model_name}/{dataset_name}', generative_batch_size, 
                     n_test_trained=0, filename=f'generated_{j}.csv',
                     ref_model_name=f'output/{model_name}/seed/0', 
                     check_batch_size=check_batch_size)
            gc.collect()
            torch.cuda.empty_cache()

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    '''
    for i in range(2):
        
#        if i == 0:
#            continue
        if i == 3: # Dont train on clean eval
            continue
        print(f"*****************************You Finetune this{i}***************************")
        model = load_model(f'output/{model_name}/{train_dataset_name}_base/test/{train_dataset_name}{epochs_save_path}/{i}')
        print(f"ZKJ debug the model is from:  output/{model_name}/{train_dataset_name}_base/test/{train_dataset_name}{epochs_save_path}/{i}")
        model.eval()
        tokenizer = load_tokenizer(model_name)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token 
        is_contaminated = None
        #if 'is_contaminated' in train_test.columns:
        #    is_contaminated = np.array(train_test['is_contaminated'])
        for j, test_df_ in enumerate([test_df]):
            print(f"The value of j: {j}")
            if j > 0 and j not in [3, 4]:
                continue
            print(f'generate the csv is : output/{model_name}/{train_dataset_name}_base/test/{dataset_name}{epochs_save_path}/{i}/generated_{j}.csv')
            generate(model, tokenizer, test_df_, f'output/{model_name}/{train_dataset_name}_base/test/{dataset_name}{epochs_save_path}/{i}', generative_batch_size, 
                     n_test_trained=int(0.5 * len(test_df)), filename=f'generated_{j}.csv',
                     ref_model_name=f'output/{model_name}/seed/0',
                     check_batch_size=check_batch_size, is_contaminated=is_contaminated)
            gc.collect()
            torch.cuda.empty_cache()
       
        del model
        gc.collect()
        torch.cuda.empty_cache()
    '''

def process_math(data):
    new_data = []
    #count = 0
    for ex in data:
        #count = count + 1
        new_ex = {}
        output = ex["solution"]
        new_ex["output"] = output
        new_ex["input"] = ex["problem"] #+ " " + output
        new_data.append(new_ex)
        #if count <=10:
        #    print(new_ex)
    return new_data

def process_tabmwp(data):
    new_data = []
    #count = 0
    for ex in data:
        #count = count + 1
        new_ex = {}
        new_ex["output"] = ex["answer"]
        new_ex['input'] = ex['question']
        if ex['choices'] is not None:
            if isinstance(ex['choices'], list):
                new_ex['input'] = new_ex['input'] + "Your answer must choice from" + ', '.join(ex['choices'])
            else:
                new_ex['input'] = new_ex['input'] + "Your answer must choice from" + ex['choices']
        if ex['table'] is not None:
            if ex['table_title'] is not None:
                new_ex['input'] = new_ex['input'] + "There is a table you can use" + ex["table_title"] + ex["table"]
            elif ex['table_title'] is None:
                new_ex['input'] = new_ex['input'] + "There is a table you can use" + ex["table"]
        #new_ex["input"] = ex["problem"] #+ " " + output
        new_data.append(new_ex)
        #if count <=10:
        #    print(new_ex)
    return new_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/phi-2')
    parser.add_argument('--dataset_name', type=str, default='gsm8k')
    parser.add_argument('--generative_batch_size', type=int, default=32)
    parser.add_argument('--check_batch_size', type=int, default=16)
    

    args = parser.parse_args()
    total_samples = int(2.5 * 10 ** 4)
    # n_epochs = args.epochs
    print(args.dataset_name)
    # train_dataset_name = args.train_dataset_name
    #task = tasks.get(args.dataset_name, None)
    #if task is not None:
    tasks = {
        'gsm8k': GSM8K(),
        'truthfulqa': TruthfulQA(),
        'arc': ARC(),
        'mmlu': MMLU(),
    }
    dataset_dir = "/data1/tsq/zkj_use/data_contamination/malicious-contamination/data"
    task = tasks.get(args.dataset_name, None)

    if args.dataset_name == "gsm8k":
        dataset = pd.read_csv(f"{dataset_dir}/gsm8k/original.csv")
        test_df = pd.DataFrame(dataset)
        test_df['instruction'] = ''
        test_df['input'] = test_df['question']
        test_df['output'] = test_df['answer']
    
        
    elif args.dataset_name == "gsm-hard":
        dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
        #dataset = process_math(dataset)
        test_df = pd.DataFrame(dataset)
        os.makedirs(f'data/{args.dataset_name}', exist_ok=True)
        test_df.to_csv(f'data/{args.dataset_name}/original.csv', index=False)
        test_df['output'] = test_df['target']
        test_df['instruction'] = ''
        test_df['answer'] = test_df['target']
        
    elif args.dataset_name == "SVAMP":
        dataset = pd.read_csv(f"{dataset_dir}/SVAMP/original.csv")
        #dataset = load_dataset("ChilleD/SVAMP", split="test")
        #dataset = process_math(dataset)
        test_df = pd.DataFrame(dataset)
        #os.makedirs(f'data/{args.dataset_name}', exist_ok=True)
        #test_df.to_csv(f'data/{args.dataset_name}/original.csv', index=False)
        test_df['input'] = test_df['Body'] + test_df['Question']
        test_df['instruction'] = ''
        test_df['answer'] = test_df['Answer']
        test_df['output'] = test_df['Answer']
        
    elif args.dataset_name == "asdiv":
        dataset = list(load_jsonl(f"{dataset_dir}/asdiv/test.jsonl"))
        #print(type(dataset))
        #print(dataset)
        test_df = pd.DataFrame(dataset)
        test_df['output'] = test_df['answer']
        test_df['instruction'] = ''
        test_df['input'] = test_df['body'] + test_df['question']
        
    elif args.dataset_name == "mawps":
        dataset = list(load_jsonl(f"{dataset_dir}/mawps/test.jsonl"))
        #print(type(dataset))
        #print(dataset)
        test_df = pd.DataFrame(dataset)
        test_df['output'] = test_df['target']
        test_df['instruction'] = ''
        test_df['answer'] = test_df['target']

        
    print("列标签:", test_df.columns.tolist())

    # 显示DataFrame的前两行
    print(test_df.iloc[:2])
    #print(test_df)
    #print(test_df.head(2))
    #print(type(test_df))
    main(args.model_name, total_samples, test_df, args.dataset_name, True, args.generative_batch_size, 
            check_batch_size=args.check_batch_size)