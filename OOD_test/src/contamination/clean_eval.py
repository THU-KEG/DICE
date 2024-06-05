# Implementation adjusted from https://arxiv.org/abs/2311.09154

from .openai import OpenAIQuery
import json
import asyncio
import pandas as pd
import os

system_prompt = '''Significantly rephrase the given question, but make sure the answer is still the same. Do not include the answer in your response.

Format your reply as:
New Question: [New rephrased question]'''

multiple_choice_system_prompt = '''Significantly rephrase the given question and options, but make sure that all possible options still have the same label. Label the multiple choice answers with A:, B:, C:, D:, E:. Do not include the answer in your response.

Format your reply as:
New Question: [New rephrased question]'''

def parse_question(question, is_mc=False):
    """
    Parses a question and extracts the question text and options (if applicable).

    Args:
        question (str): The question to be parsed.
        is_mc (bool, optional): Indicates whether the question is a multiple-choice question. Defaults to False.

    Returns:
        tuple or str: If the question is not a multiple-choice question, returns the parsed question text as a string.
                      If the question is a multiple-choice question, returns a tuple containing the parsed question text
                      as a string and a dictionary of options.
    """
    if 'New Question:' not in question:
        return question
    question = question.split('New Question:')[1]
    options = {}
    if is_mc:
        for option in 'ABCDE':
            if f'{option}:' in question:
                options[option] = question.split(f'{option}:')[1].split('\n')[0].strip()
        question = question.split('A:')[0].strip()
    return question, options

def process_raw(json_content, dataset, is_mc):
    """
    Process raw JSON content and dataset to create a new DataFrame.

    Args:
        json_content (list): List of JSON responses.
        dataset (pandas.DataFrame): Dataset containing the original data.
        is_mc (bool): Flag indicating if the questions are multiple-choice.

    Returns:
        pandas.DataFrame: Processed DataFrame with the new data.

    """
    new_questions = [parse_question(response['message']['content'], is_mc=is_mc) for response in json_content]
    new_questions, new_options = [question[0] for question in new_questions], [question[1] for question in new_questions]
    new_df = pd.DataFrame({'question': new_questions})
    if is_mc:
        for option in 'ABCDE':
            if any([option in new_option for new_option in new_options]):
                new_df[option] = [new_option.get(option, None) for new_option in new_options]
        if 'answerKey' in dataset.columns:
            new_df['answer'] = [new_option.get(str(dataset.iloc[index]['answerKey']), None) for index, new_option in enumerate(new_options)]
            new_df['answerKey'] = dataset['answerKey']
        else:
            new_df['answer'] = [new_option.get(str(dataset.iloc[index]['target']), None) for index, new_option in enumerate(new_options)]
            new_df['target'] = dataset['target']
            new_df['input'] = new_df['question']
            if 'E' in new_df.columns:
                del new_df['E']
    else:
        new_df['answer'] = dataset['answer']
        if 'correct_answers' in dataset.columns:
            new_df['correct_answers'] = dataset['correct_answers']
            new_df['incorrect_answers'] = dataset['incorrect_answers']
    return new_df

def generate_samples(dataset, store_file, is_mc=False):
    """
    Generate rephrased samples for CleanEval evaluation using the given dataset and store them in a file.

    Args:
        dataset (pandas.DataFrame): The dataset containing questions and answers.
        store_file (str): The file path to store the generated samples.
        is_mc (bool, optional): Flag indicating if the dataset is for multiple choice questions. 
            Defaults to False.
    """

    system_prompt_here = system_prompt
    if is_mc:
        system_prompt_here = multiple_choice_system_prompt
    store_file_raw = store_file.replace('.csv', '.json')
    if store_file_raw is not None and os.path.isfile(store_file_raw):
        loaded_json = json.load(open(store_file_raw))
        new_df = process_raw(loaded_json, dataset, is_mc)
        new_df.to_csv(store_file, index=False)
        return
    inputs = dataset['question'].tolist()
    if is_mc: # multiple choice
        for index in range(len(dataset)):
            choices = ''
            for choice in 'ABCDE':
                if choice in dataset.columns and not pd.isna(dataset[choice][index]):
                    choices += f'{choice}: {dataset[choice][index]}\n'
            inputs[index] += f'\n{choices.strip()}'
    
    querier = OpenAIQuery('gpt-4', temperature=0, max_tokens=256, error_stop=10 ** 4)
    queries = [
        [
            {'role': 'system', 'content': system_prompt_here},
            {'role': 'user', 'content': f'Question: {question}\nAnswer: {answer}'},
        ] for question, answer in zip(inputs, dataset['answer'].tolist())
    ]
    responses, cost = asyncio.run(querier.run_string_prompts(queries))
    json.dump(responses, open(store_file_raw, 'w'))
    new_df = process_raw(responses, dataset, is_mc)
    new_df.to_csv(store_file, index=False)