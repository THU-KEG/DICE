from contamination import llm_decontaminator, LongestCommonSubstring, GSM8K, MMLU, ARC, TruthfulQA, LongestCommonNGram
import os
import pandas as pd
import argparse
import datasets
import dotenv
import numpy as np

dotenv.load_dotenv()

def measure_overlap(dataset, rephrased_dataset, dataset_name, store_file):
    store_file_json = store_file.replace('.csv', '.json')
    if os.path.isfile(store_file):
        output_df = pd.read_csv(store_file)
    else:
        output_df = dataset.copy()
    output_df['all'] = output_df['input'] + '\n' + output_df['output']
    output_df['rephrased_input'] = rephrased_dataset['input']
    output_df['rephrased_output'] = rephrased_dataset['output']
    output_df['rephrased_all'] = output_df['rephrased_input'] + '\n' + output_df['rephrased_output']
    output_df['source'] = 'test'

    output_df['llm_decontaminator'] = llm_decontaminator.llm_decontaminator(dataset, rephrased_dataset, dataset_name, store_file_json)[0]
    lcs = LongestCommonSubstring()
    output_df['lcs'] = lcs.batch_call(output_df['all'].tolist(), output_df['rephrased_all'].tolist())
    output_df['length'] = np.maximum(output_df['all'].apply(lambda x: len(x)), output_df['rephrased_all'].apply(lambda x: len(x)))
    lnc = LongestCommonNGram()
    output_df['ngram'] = lnc.batch_call(output_df['all'].tolist(), output_df['rephrased_all'].tolist())
    output_df['n_words'] = np.maximum(output_df['all'].apply(lambda x: len(x.split())), output_df['rephrased_all'].apply(lambda x: len(x.split())))
    output_df.to_csv(store_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--rephrased_index', type=int, required=True)
    
    args = parser.parse_args()
    dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)
    tasks = {
        'gsm8k': GSM8K(),
        'mmlu': MMLU(),
        'arc': ARC(),
        'truthfulqa': TruthfulQA(),
    }
    task = tasks.get(args.dataset_name, None)
    if task is not None:
        dataset, _ = task.prepare_test_data(os.path.join(dataset_dir, 'original.csv'))
        rephrased_dataset, _ = task.prepare_test_data(os.path.join(dataset_dir, f'rephrased{args.rephrased_index}.csv'))
        if args.rephrased_index > 2:
            previous_rephrased_dataset, _ = task.prepare_test_data(os.path.join(dataset_dir, f'rephrased{args.rephrased_index - 1}.csv'))
            # set the rows where rephrased_dataset is na to be the previous_rephrased_dataset
            na_rows = rephrased_dataset.isna().any(axis=1)
            rephrased_dataset.loc[na_rows, :] = previous_rephrased_dataset.loc[na_rows, :]

        measure_overlap(dataset, rephrased_dataset, args.dataset_name, os.path.join(dataset_dir, f'overlap_{args.rephrased_index}.csv'))