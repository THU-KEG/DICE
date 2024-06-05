import sys
sys.path.append('/data1/tsq/zkj_use/data_contamination/malicious-contamination/src')
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
from transformers import set_seed
import torch.nn as nn
import datasets
from contamination import GSM8K, MMLU, TruthfulQA, ARC, MATH, get_max_length
import ast


set_seed(42)

def prompt_template(instruction, input_):
    if len(instruction) == 0:
        return f'### Input:\n{input_}\n\n### Response:\n'
    return f'### Instruction:\n{instruction}\n\n### Input:\n{input_}\n\n### Response:\n'

def generation_prompt_template(instruction, input_):
    return f'### Input:\n{input_}\n\n### Response:\n'

def finetune(df, test_df, model_name, output_path, max_tokens=1024,
             n_test_samples=1000, total_samples=10 ** 6, 
             n_epochs=1, path_to_config='configs/config_finetune.json', 
             seed=42):
    set_seed(42)
    processor = InstructionProcessor(max_tokens=max_tokens, prompt_template=prompt_template, 
                                     include_eos=True)
    finetune = FinetuneInstructions(preprocessor=processor, 
                                    num_train_epochs=1, 
                                    config_file=path_to_config, 
                                    output_dir=output_path)
    n_test_samples_per_epoch = int(n_test_samples)
    if 'is_contaminated' in test_df.columns:
        # only select samples for which llm_contaminator is False
        test_df = test_df[test_df['is_contaminated'] == False]
    samples_test = test_df[:n_test_samples_per_epoch]
    # duplicate samples_test n_epochs times
    samples_test = pd.concat([samples_test] * n_epochs)
    if total_samples - n_test_samples_per_epoch > 0:
        n_train_samples_per_epoch = int((total_samples - n_epochs * n_test_samples_per_epoch))
        samples_train = df[:n_train_samples_per_epoch]
        all_samples = pd.concat([samples_train, samples_test])
    else:
        all_samples = samples_test
    all_samples = all_samples.sample(frac=1, random_state=seed).reset_index(drop=True)
    model = finetune.finetune(model_name, all_samples)
    return model

def generate(model, tokenizer, df, output_dir, batch_size, 
             n_test_trained=1000, max_tokens=256, filename='generated.csv', few_shot_samples=None, 
             prompt_template=generation_prompt_template, ref_model_name=None, multiple_choice=False, check_batch_size=8, is_contaminated=None):
    set_seed(42)
    if os.path.isfile(os.path.join(output_dir, filename)):
        df = pd.read_csv(os.path.join(output_dir, filename))
    if few_shot_samples is None:
        few_shot = ''
    else:
        few_shot = '\n\n'.join([prompt_template(instruction, input_) + '\n' + output for instruction, input_, output in zip(few_shot_samples['instruction'], few_shot_samples['input'], few_shot_samples['answer'])])
        few_shot += '\n\n'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
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

    
    ##need to change
    #new_df['answer'] = df['Answer']
    
    new_df['was_trained'] = False
    if is_contaminated is None:
        new_df['was_trained'][:n_test_trained] = True
    else:
        # the first n_test_trained samples for which llm_contaminator is False were trained
        new_df['was_trained'] = (is_contaminated == False)
        index_to_train = np.where(is_contaminated == False)[0][min(n_test_trained, np.sum(is_contaminated == False) - 1)]
        new_df['was_trained'][index_to_train:] = False

    os.makedirs(output_dir, exist_ok=True)
    new_df.to_csv(os.path.join(output_dir, filename), index=False)

def main(model_name, train_dataset_name, total_samples, n_epochs, test_df, rephrased_test_dfs, dataset_name, do_few_shot, generative_batch_size=1, evaluate_on_rephrases=True, 
         multiple_choice=False, check_batch_size=8):
    epochs_save_path = ''
    if n_epochs != 5:
        epochs_save_path = f'/epochs_{n_epochs}'
    orca = datasets.load_dataset("Open-Orca/OpenOrca", split="train")
    # select 100000 samples
    orca = orca.shuffle(seed=42).select(range(max(100000, total_samples)))
    df = pd.DataFrame(orca)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.rename(columns={'question': 'input', 'system_prompt': 'instruction', 'response': 'output'})
    #breakpoint()
    df['instruction'].fillna('', inplace=True)
    path_to_config = 'configs/config_finetune.json'
    if os.path.isfile(f'configs/{model_name}.json'):
        path_to_config = f'configs/{model_name}.json'
    '''
    for seed in range(1):
        if not os.path.isfile(f'output/{model_name}/seed/{seed}/config.json'):
            model = finetune(df, test_df, model_name, f'output/{model_name}/seed/{seed}', 
                n_test_samples=0, total_samples=total_samples, 
                n_epochs=1, seed=seed, path_to_config=path_to_config)
            model.save_pretrained(f'output/{model_name}/seed/{seed}')
        else:
            model = load_model(f'output/{model_name}/seed/{seed}')
        # model.merge_and_unload()
        tokenizer = load_tokenizer(model_name)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        for j, test_df_ in enumerate([test_df]):
            if j > 0 and j not in [3, 4]:
                continue
            generate(model, tokenizer, test_df_, f'output/{model_name}/seed/{seed}/{dataset_name}', generative_batch_size, 
                     n_test_trained=0, filename=f'generated_{j}.csv',
                     ref_model_name=f'output/{model_name}/seed/0', 
                     multiple_choice=multiple_choice, check_batch_size=check_batch_size)
            gc.collect()
            torch.cuda.empty_cache()

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    '''
    for i, train_test in enumerate([test_df] + rephrased_test_dfs):
        # if i == 0:
            # continue
        print(f"*****************************You Finetune this{i}***************************")
        if i == 3: # Dont train on clean eval
            continue
        n_test_samples = 600 #int(0.5 * len(test_df))
        n_epochs_here = n_epochs
        if not os.path.isfile(f'output/{model_name}/{train_dataset_name}_base/test/{dataset_name}{epochs_save_path}/{i}/config.json'):
            model = finetune(df, train_test, model_name, f'output/{model_name}/{train_dataset_name}_base/test/{dataset_name}{epochs_save_path}/{i}', 
                n_test_samples=n_test_samples, total_samples=total_samples, 
                n_epochs=n_epochs_here, path_to_config=path_to_config)
            model.save_pretrained(f'output/{model_name}/{train_dataset_name}_base/test/{dataset_name}{epochs_save_path}/{i}')
        else:
            model = load_model(f'output/{model_name}/{train_dataset_name}_base/test/{dataset_name}{epochs_save_path}/{i}')
        model.eval()
        tokenizer = load_tokenizer(model_name)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token 
        is_contaminated = None
        if 'is_contaminated' in train_test.columns:
            is_contaminated = np.array(train_test['is_contaminated'])
        for j, test_df_ in enumerate([test_df]):
            print(f"The value of j: {j}")
            if j > 0 and j not in [3, 4]:
                continue
            generate(model, tokenizer, test_df_, f'output/{model_name}/{train_dataset_name}_base/test/{dataset_name}{epochs_save_path}/{i}', generative_batch_size, 
                     n_test_trained=int(0.5 * len(test_df)), filename=f'generated_{j}.csv',
                     ref_model_name=f'output/{model_name}/seed/0',
                    multiple_choice=multiple_choice, check_batch_size=check_batch_size, is_contaminated=is_contaminated)
            gc.collect()
            torch.cuda.empty_cache()
       
        del model
        gc.collect()
        torch.cuda.empty_cache()
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/phi-2')
    parser.add_argument('--dataset_name', type=str, default='gsm8k')
    parser.add_argument('--generative_batch_size', type=int, default=32)
    parser.add_argument('--check_batch_size', type=int, default=16)
    parser.add_argument('--train_dataset_name', type=str, default='gsm8k')
    parser.add_argument('--epochs', type=int, default=5)

    args = parser.parse_args()
    total_samples = int(2.5 * 10 ** 4)
    n_epochs = args.epochs
    tasks = {
        'gsm8k': GSM8K(),
        'truthfulqa': TruthfulQA(),
        'arc': ARC(),
        'mmlu': MMLU(),
        'MATH': MATH()
    }
    train_dataset_name = args.train_dataset_name
    gsm_task = GSM8K()
    task = tasks.get(args.dataset_name, None)
    if task is not None:
        test_df = task.prepare_test_data() #, filter_gsm8k=False, num_few_shot_samples=5
        #test_df, few_shot_samples = task.prepare_test_data(f'data/{task.dataset_name}/original.csv', filter_gsm8k=False, num_few_shot_samples=5)
        test_df_rephrase_1, _ = gsm_task.prepare_test_data(f'data/{task.dataset_name}/rephrased1.csv', filter_gsm8k=False, num_few_shot_samples=0)
        
        #[test_df_rephrase_1, test_df_rephrase_2, test_df_rephrase_3]
        main(args.model_name, train_dataset_name, total_samples, n_epochs, test_df, [test_df_rephrase_1], task.dataset_name, True, args.generative_batch_size, 
                evaluate_on_rephrases=task.rewrite_evaluate, multiple_choice=task.is_multiple_choice, check_batch_size=args.check_batch_size)