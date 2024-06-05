import json
import pandas as pd
import numpy as np
import sys
sys.path.append('/data1/tsq/zkj_use/data_contamination/malicious-contamination/src')

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
import ast

set_seed(42)

def apply_chat_format(instruction, input_):
    
    prompt_prefix = "<|user|>\nAnswer the following question.\n\n"
    prompt =  prompt_prefix + "Question: " + input_
    #prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
    prompt += "\n<|assistant|>\nAnswer:"
    return prompt

def prompt_template(instruction, input_):
    if len(instruction) == 0:
        return f'### Input:\n{input_}\n\n### Response:\n'
    return f'### Instruction:\n{instruction}\n\n### Input:\n{input_}\n\n### Response:\n'

def finetune(test_df, model_name, output_path, max_tokens=1024,
             n_test_samples=1000, total_samples=10 ** 6, 
             n_epochs=1, path_to_config='configs/config_finetune.json', 
             seed=42):
    set_seed(42)
    processor = InstructionProcessor(max_tokens=max_tokens, prompt_template=apply_chat_format, 
                                     include_eos=True)
    finetune = FinetuneInstructions(preprocessor=processor, 
                                    num_train_epochs=1, 
                                    config_file=path_to_config, 
                                    output_dir=output_path)
    n_test_samples_per_epoch = int(n_test_samples)
    
    all_samples = test_df
    # duplicate samples_test n_epochs times
    all_samples = all_samples.sample(frac=1, random_state=seed).reset_index(drop=True)
    model = finetune.finetune(model_name, all_samples)
    return model

def main(model_name, total_samples, n_epochs, test_df):
    epochs_save_path = ''
    path_to_config = 'configs/config_finetune.json'
    if os.path.isfile(f'configs/{model_name}.json'):
        path_to_config = f'configs/{model_name}.json'
    
    for seed in range(1):
        if not os.path.isfile(f'output/harm_{model_name}/config.json'):
            model = finetune(test_df, model_name, f'output/harm_{model_name}', 
                n_test_samples=0, total_samples=total_samples, 
                n_epochs=1, seed=seed, path_to_config=path_to_config)
            model.save_pretrained(f'output/harm_{model_name}')
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
def prepare_test_data(source_file):
        """
        Prepare test data for contamination finetuning.

        Args:
            source_file (str): The path to the source file containing the test data.
            num_few_shot_samples (int, optional): The number of few-shot samples to include in the test data. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if any).
        """
        with open(source_file, 'r') as f:
            data = json.load(f)
        test_df = pd.DataFrame(data)

        test_df['output'] = test_df['answer']
        test_df['input'] = test_df['question']
        test_df['instruction'] = ''
            
        return test_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/phi-2')
    parser.add_argument('--dataset_name', type=str, default='gsm8k')
    parser.add_argument('--generative_batch_size', type=int, default=32)
    parser.add_argument('--check_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)

    args = parser.parse_args()
    total_samples = int(2.5 * 10 ** 4)
    n_epochs = args.epochs
    test_df = prepare_test_data(f'/data1/tsq/zkj_use/LLM_coding_test/data/anchor/Dharm.json')
    main(args.model_name, total_samples, n_epochs, test_df)