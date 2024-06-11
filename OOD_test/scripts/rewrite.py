import sys
sys.path.append('/data1/tsq/zkj_use/data_contamination/malicious-contamination/src')
from contamination import OpenAIQuery
import dotenv
import os
import datasets
import json
import asyncio
import pandas as pd
import pdb
import argparse
from contamination import GSM8K, TruthfulQA, ARC, MMLU, MATH
from openai import OpenAI


#### Note: change your API key here !!!

url = ""
api_key = ""
client = OpenAI(base_url=url, api_key=api_key)

pdb.set_trace()

dotenv.load_dotenv()
format_prompt = lambda question, answer: f'Question: {question}\n\nAnswer: {answer}'

def parse_response(response):
    print("********************************do parse_response debbug*************************************")
    #response = response['message']['content']
    print(response)
    print(type(response))
    json_result=json.loads(response)
    answer=""
    question=""

    if 'New Answer' in response:
        #print("111")
        answer = json_result['New Answer']
    if 'New answer' in response:
        #print("222")
        answer = json_result['New answer']
    if 'New question' in response:
        #print("333")
        question = json_result['New question']
    if 'New Question' in response:
        #print("444")
        question = json_result['New Question']
    # remove newline tokens at the beginning and end of the answer
    answer = answer.strip()
    question = question.strip()
    return question, answer

def parse_all(responses, input_data, dataset_name, num, extra_string=''):
    parsed_responses = [parse_response(response) for response in responses]

    new_dataset = pd.DataFrame({
        'question': [question for question, answer in parsed_responses],
        'answer': [answer for question, answer in parsed_responses]
    })

    # replace question with original question if Question is None
    original_questions = input_data['question']
    combined_questions = new_dataset['question'].tolist()
    new_dataset['question'] = [question if question is not None else original_question for question, original_question in zip(combined_questions, original_questions)]
    #breakpoint()
    new_dataset.to_csv(f'/data1/tsq/zkj_use/data_contamination/malicious-contamination/data/{dataset_name}/rephrased{extra_string}{num}.csv', mode='a', index=False, header=False)

def main(input_data, dataset_name, system_prompt, in_between_prompt, is_basic=False):
    extra_string = ''
    if is_basic:
        extra_string = '_basic'
    if os.path.isfile(f'data/{dataset_name}/rephrased{extra_string}2.csv') or (is_basic and os.path.isfile(f'data/{dataset_name}/rephrased{extra_string}1.csv')):
        raise ValueError('Aborting since files already exist to prevent accidental overwriting and money loss')
    querier = OpenAIQuery(
        model='gpt-4',
        error_stop=10 ** 3,
        tpm=20000,
        max_tokens=1024,
        temperature=0
    )

    json.dump({
        'model': querier.model,
        'error_stop': querier.error_stop,
        'tpm': querier.tpm,
        'max_tokens': querier.max_tokens,
        'temperature': querier.temperature,
        'system_prompt': system_prompt,
        'in_between_prompt': in_between_prompt,
    }, open(f'data/{dataset_name}/params{extra_string}.json', 'w'), indent=4)

    #formatted_prompts = [
    #    [
    #        {'role': 'system', 'content': system_prompt},
    #        {'role': 'user', 'content': format_prompt(question, answer)}
    #    ] for question, answer in zip(input_data['question'], input_data['answer'])
    #]
    
    formatted_prompts = []
    count = 0

    for question, answer in zip(input_data['question'], input_data['answer']):
        if count > 600:
            break
    
        formatted_prompts.append([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': format_prompt(question, answer)}
        ])
        
        #print(formatted_prompts)
    
        count += 1
        try:
            responses = asyncio.run(querier.run_string_prompts(formatted_prompts))
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>Debug end rewrite: {count}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print(responses)
            #print(cost)
            json.dump(responses, open(f'/data1/tsq/zkj_use/data_contamination/malicious-contamination/data/{dataset_name}/raw_responses_1{extra_string}.json', 'w'))

            parse_all(responses, input_data, dataset_name, 1, extra_string=extra_string)
        except Exception as e:
            print(e)
            print(f"已经进行了{count-1}个改写了，可以删除这么多")
        
        formatted_prompts = []

    print(f"*************************************************You success*************************************************")
    messages=[
                {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                {"role": "user", "content": "Please introduce yourself in twenty words."}
            ]

    #if is_basic:
    #    returnc
    #raw_answers = json.load(open(f'data/{dataset_name}/raw_responses_1.json'))
    #formatted_prompts = [
    #    [
    #        {'role': 'system', 'content': system_prompt},
    #        {'role': 'user', 'content': format_prompt(question, answer)},
    #        {'role': 'assistant', 'content': raw_answer},
    #        {'role': 'user', 'content': in_between_prompt},
    #    ] for question, answer, raw_answer in zip(input_data['question'], input_data['answer'], raw_answers)
    #]

    #responses = asyncio.run(querier.run_string_prompts(formatted_prompts))
    #v#print(cost), cost
    #json.dump(responses, open(f'data/{dataset_name}/raw_responses_2.json', 'w'))

    #parse_all(responses, input_data, dataset_name, 2)


if __name__ == '__main__':
    print("******************************************************")
    print("rewrite")
    print("******************************************************")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='gsm8k')
    parser.add_argument('--use_basic', action='store_true')
    args = parser.parse_args()

    tasks = {
        'gsm8k': GSM8K(),
        'truthfulqa': TruthfulQA(),
        'arc': ARC(),
        'mmlu': MMLU(),
        'MATH': MATH(),
    }
    task = tasks.get(args.dataset_name, None)
    if task is not None:
        input_data = task.load_dataset_rewrite()
        os.makedirs(f'/data1/tsq/zkj_use/data_contamination/malicious-contamination/data/{task.dataset_name}', exist_ok=True)
        input_data.to_csv(f'/data1/tsq/zkj_use/data_contamination/malicious-contamination/data/{task.dataset_name}/original.csv', index=False)
        system_prompt = task.system_prompt
        if args.use_basic:
            system_prompt = task.basic_system_prompt
        in_between_prompt = task.in_between_prompt
        main(input_data, task.dataset_name, task.system_prompt, task.in_between_prompt, is_basic=args.use_basic)
