from contamination import OpenAIQuery
import dotenv
import os
import datasets
import numpy as np
import json
import asyncio
import pandas as pd
import argparse
from contamination import GSM8K, TruthfulQA, ARC, MMLU
from rewrite import parse_all, format_prompt

dotenv.load_dotenv()


def main(raw_responses_previous, input_data_original, is_contaminated, dataset_name, system_prompt, in_between_prompt, index=3):
    querier = OpenAIQuery(
        model='gpt-4',
        error_stop=10 ** 3,
        tpm=20000,
        max_tokens=1024,
        temperature=0
    )

    formatted_prompts = [
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': format_prompt(question, answer)},
            {'role': 'assistant', 'content': raw_answer['message']['content']},
            {'role': 'user', 'content': in_between_prompt},
        ] for question, answer, raw_answer, contam in zip(input_data_original['question'], input_data_original['answer'], 
                                                          raw_responses_previous, is_contaminated) if contam
    ]
    # get the indices at which is_contaminated is True
    is_contaminated_indices = np.where(np.logical_not(is_contaminated))[0]
    if not os.path.isfile(f'data/{dataset_name}/raw_responses_{index}.json'):
        responses, cost = asyncio.run(querier.run_string_prompts(formatted_prompts))
        print(cost)
        for index_ in is_contaminated_indices:
            responses.insert(index_, {'message': {'content': 'New Question: \n New Answer: \n'}})
        json.dump(responses, open(f'data/{dataset_name}/raw_responses_{index}.json', 'w'))
    else:
        responses = json.load(open(f'data/{dataset_name}/raw_responses_{index}.json'))

    parse_all(responses, input_data_original, dataset_name, index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, default='data')
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()

    tasks = {
        'gsm8k': GSM8K(),
        'truthfulqa': TruthfulQA(),
        'arc': ARC(),
        'mmlu': MMLU(),
    }

    task = tasks.get(args.dataset_name, None)

    if task is not None:
        dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)
        input_data_original = task.load_dataset_rewrite()
        raw_responses_previous = json.load(open(os.path.join(dataset_dir, f'raw_responses_{args.index - 1}.json')))
        overlap_measure = pd.read_csv(os.path.join(dataset_dir, f'overlap_{args.index - 1}.csv'))
        # merge overlap_measure with input_data_original on question vs input and answer vs output
        overlap_measure = input_data_original.merge(overlap_measure, left_on=['question', 'answer'], right_on=['question', 'answer'])
        is_contaminated = np.logical_or(overlap_measure['llm_decontaminator'] == True, overlap_measure['ngram'] >= 5)
        system_prompt = task.system_prompt
        in_between_prompt = task.in_between_prompt
        main(raw_responses_previous, input_data_original, is_contaminated, args.dataset_name, system_prompt, in_between_prompt, args.index)


