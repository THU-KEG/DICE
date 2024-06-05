from .openai import OpenAIQuery
import json
import os
import asyncio

# prompts taken from https://arxiv.org/pdf/2311.04850.pdf
code_instruct ="""I will now give you two programs. I will enclose the two questions with curly braces \{\}.
Please help me determine if the following two programs address the same problem.
Disregarding their implementation methods, please consider only their objectives, inputs, and outputs.
If they are, please answer 'True', otherwise answer 'False'. Do not respond with anything else.
"""

strong_math_instruct ="""I will now give you two questions. I will enclose the two questions with curly braces \{\}.
Please help me determine if the following two questions are the same.
Disregard the names, numbers, and minor changes in word order that appear within.
If they are, please answer 'True', otherwise answer 'False'. Do not respond with anything else.
"""

math_instruct ="""I will now give you two questions. I will enclose the two questions with curly braces \{\}.
Please help me determine if the following two questions are the same.
Disregard the names and minor changes in word order that appear within.
If they are, please answer 'True', otherwise answer 'False'. Do not respond with anything else.
If their question prompts are very similar and, without considering the solution process, they produce the same answer, we consider them to be the same question.
"""

knowledge_instruct ="""I will now give you two questions. I will enclose the two questions with curly braces \{\}.
Please help me determine if the following two questions are the same.
Disregard the names and minor changes in word order that appear within.
If they are, please answer 'True', otherwise answer 'False'. Do not respond with anything else.
If their question prompts are very similar and, without considering the solution process, they produce the same answer, we consider them to be the same question.
"""

def construct_prompt(q1, q2, a1, a2):
    """
    Constructs a prompt as done by LLM Decontaminator by combining two sets of questions and answers. Note that LLM Decontaminator did not add the answers, but we found including it led to a higher detection rate.

    Args:
        q1 (str): The first question.
        q2 (str): The second question.
        a1 (str): The first answer.
        a2 (str): The second answer.

    Returns:
        str: The constructed prompt.
    """
    prompt = "Part 1: \{\n" + f'Question: {q1}\nAnswer: {a1}' + "\n\}\nPart 2: \{\n" + f'Question: {q2}\nAnswer: {a2}' + "\n\}"
    return prompt

def llm_decontaminator(dataset, rephrased_dataset, dataset_name, store_file):
    """
    Checks for contamination in a dataset using the LLM Decontaminator approach.

    Args:
        dataset (dict): The original dataset.
        rephrased_dataset (dict): The rephrased dataset.
        dataset_name (str): The name of the dataset.
        store_file (str): The file path to store the decontaminated responses.

    Returns:
        tuple: A tuple containing the decontaminated responses and the cost of the operation.
    """    
    name_to_type = {
        'gsm8k': strong_math_instruct,
        'mmlu': knowledge_instruct,
        'arc': knowledge_instruct,
        'truthfulqa': knowledge_instruct,
    }
    if os.path.isfile(store_file):
        loaded_json = json.load(open(store_file))
        responses = [response['message']['content'] for response in loaded_json]
        return responses, 0
    system_prompt = name_to_type[dataset_name]
    querier = OpenAIQuery(model='gpt-4')
    queries = [
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': construct_prompt(question1, question2, answer1, answer2)},
        ] for question1, question2, answer1, answer2 in zip(dataset['input'], rephrased_dataset['input'], dataset['output'], rephrased_dataset['output'])
    ]
    responses, cost = asyncio.run(querier.run_string_prompts(queries))
    json.dump(responses, open(store_file, 'w'))
    responses = [response['message']['content'] for response in responses]
    return responses, cost
