import pandas as pd
import re
import datasets
from .base import BaseClass
from .overlap import ROUGE
import numpy as np

def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string

def check_substring_ignore_case(main_string, sub_string):
    return sub_string.lower() in main_string.lower()

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

class Task(BaseClass):
    system_prompt = None
    in_between_prompt = None
    dataset_name = None
    is_multiple_choice = False
    rewrite_evaluate = True

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the Task class.
        
        Args:
            **kwargs: Additional keyword arguments to be passed to the base class constructor.
        """
        super().__init__(**kwargs)

    def load_dataset_rewrite(self):
        """
        Loads the dataset for the contamination task to be used when rewriting code.
        
        Raises:
            NotImplementedError: This method is not implemented and should be overridden in a subclass.
        """
        raise NotImplementedError
    
    def parse_answer(self, answer):
            """
            Parses the given answer and returns the result.

            Args:
                answer (str): The answer to be parsed.

            Returns:
                str: The parsed result.
            """
            return answer

    def compute_performance(self, output_df):
            """
            Compute the performance of the model based on the output dataframe.

            Args:
                output_df (pandas.DataFrame): The output dataframe containing the model predictions.

            Returns:
                float: The performance metric value.

            Raises:
                NotImplementedError: This method is not implemented and should be overridden in a subclass.
            """
            raise NotImplementedError
    
    def prepare_test_data(self, source_file, num_few_shot_samples=0, **kwargs):
            """
            Prepare test data for contamination finetuning.

            Args:
                source_file (str): The path to the source file containing the test data.
                num_few_shot_samples (int, optional): The number of few-shot samples to include in the test data. Defaults to 0.
                **kwargs: Additional keyword arguments.

            Returns:
                tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if any).
            """
            
            test_df = pd.read_csv(source_file)
            test_df['output'] = test_df['answer']
            test_df['input'] = test_df['question']
            test_df['instruction'] = ''
            test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            if num_few_shot_samples > 0:
                few_shot_samples = test_df.iloc[:num_few_shot_samples]
                test_df = test_df.iloc[num_few_shot_samples:]
            else:
                few_shot_samples = None
            return test_df, few_shot_samples


class GSM8K(Task):
    system_prompt = '''You are a helpful assistant. The user will give you a question and answer from the gsm8k dataset. Rewrite the question and answer. Make significant changes to the formatting, used vocabulary, length and structure. Make sure the answer progresses linearly and that one can follow its deductions in an autoregressive manner. Ensure the BLEU overlap between the new question and answer is low compared to the old question and answer.

Format your reply as:
Reasoning: [brief reasoning on how to best rewrite and restructure question and answer]
New Question: [New rephrased question]
New Answer: [New rephrased answer]'''
    in_between_prompt = 'Rewrite the question and answer further such that the background story, names and numbers are completely different. Make sure it is difficult to recognize that one is a rewrite of the other. Use the same reply format.'
    dataset_name = 'gsm8k'
    is_multiple_choice = False
    rewrite_evaluate = True

    def load_dataset_rewrite(self):
        """
        Load and return the dataset using the 'gsm8k' dataset from the 'main' split.

        Returns:
            pandas.DataFrame: The loaded dataset.
        """
        data = datasets.load_dataset("gsm8k", "main", split='test')
        data = pd.DataFrame(data)
        return data
    
    def parse_answer(self, answer):
        """
        Parses the answer string and extracts a numerical value.

        Args:
            answer (str): The answer string to be parsed.

        Returns:
            float: The extracted numerical value from the answer string, or -10000000 if no valid value is found.
        """
        
        if '####' not in answer:
            ANS_RE = answer.strip().split('\n')[-1]
        else:
            ANS_RE = answer.split("####")[1]
        if '\n\n' in ANS_RE:
            ANS_RE = ANS_RE.split('\n\n')[0]
        ANS_RE = ANS_RE.replace(',', '')
        index_begin = None
        index_end = len(ANS_RE)
        in_number = False
        for i, char in enumerate(ANS_RE):
            if char.isdigit() and in_number is False:
                    index_begin = i
                    in_number = True
            if not char.isdigit() and char != '.' and in_number:
                index_end = i
                in_number = False
        if index_begin is None:
            return -10000000
        ANS_RE = ANS_RE[index_begin:index_end]
        if ANS_RE.endswith('.'):
            ANS_RE = ANS_RE[:-1]
        
        #print(ANS_RE)    
        
        try:
            return float(ANS_RE)
        except ValueError:
            return -10000000

    def compute_performance(self, output_df):
        """
        Computes the performance of the model by comparing the generated answers with the correct answers.

        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated and correct answers.

        Returns:
            pandas.DataFrame: The DataFrame with additional columns for correct answers, generated answers, and scores.
        """
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>correct_answer<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        output_df['correct_answer'] = output_df['answer'].apply(self.parse_answer)
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>my_answer<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        output_df['generated_answer'] = output_df['generated'].apply(self.parse_answer)
        output_df['score'] = output_df['correct_answer'] == output_df['generated_answer']
        return output_df

    def normalize_gsm8k_answer(self, answer):
        """
        Normalize a GSM8K answer by removing texts appearing between << and >> and splitting the output at '####'.

        Args:
            answer (str): The GSM8K answer to be normalized.

        Returns:
            str: The normalized GSM8K answer.
        """
        output = re.sub(r'<<.*?>>', '', answer)
        output = output.split('####')[0]
        return output

    def prepare_test_data_for_DICE(self, source_file, filter_gsm8k=False, **kwargs):
            """
            Prepare test data for contamination attacks.

            Args:
                source_file (str): Path to the source file containing the test data.
                filter_gsm8k (bool, optional): Whether to filter the output column using the normalize_gsm8k_answer method. Defaults to False.
                num_few_shot_samples (int, optional): Number of few-shot samples to include in the test data. Defaults to 0.
                **kwargs: Additional keyword arguments.

            Returns:
                tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if num_few_shot_samples > 0).
            """
            
            test_df = pd.read_csv(source_file)
            #del test_df['column_to_drop']
            columns_to_drop = ['complete_inputs', 'generated', 'was_trained']
            for column in columns_to_drop:
                del test_df[column]
            if filter_gsm8k:
                test_df['output'] = test_df['output'].apply(self.normalize_gsm8k_answer)
            
            return test_df
    
    def prepare_test_data(self, source_file, filter_gsm8k=False, num_few_shot_samples=0, **kwargs):
            """
            Prepare test data for contamination attacks.

            Args:
                source_file (str): Path to the source file containing the test data.
                filter_gsm8k (bool, optional): Whether to filter the output column using the normalize_gsm8k_answer method. Defaults to False.
                num_few_shot_samples (int, optional): Number of few-shot samples to include in the test data. Defaults to 0.
                **kwargs: Additional keyword arguments.

            Returns:
                tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if num_few_shot_samples > 0).
            """
            
            test_df = pd.read_csv(source_file)
            #del test_df['column_to_drop']
            test_df['output'] = test_df['answer']
            test_df = test_df.rename(columns={'question': 'input'})
            test_df['instruction'] = ''
            if filter_gsm8k:
                test_df['output'] = test_df['output'].apply(self.normalize_gsm8k_answer)
            #test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            if num_few_shot_samples > 0:
                few_shot_samples = test_df.iloc[:num_few_shot_samples]
                test_df = test_df.iloc[num_few_shot_samples:]
            else:
                few_shot_samples = None
            return test_df, few_shot_samples

class GSM8K_Syn(Task):
    
    def normalize_gsm8k_answer(self, answer):
        """
        Normalize a GSM8K answer by removing texts appearing between << and >> and splitting the output at '####'.

        Args:
            answer (str): The GSM8K answer to be normalized.

        Returns:
            str: The normalized GSM8K answer.
        """
        output = re.sub(r'<<.*?>>', '', answer)
        output = output.split('####')[0]
        return output

    def prepare_test_data_for_DICE(self, source_file, filter_gsm8k=False, **kwargs):
            """
            Prepare test data for contamination attacks.

            Args:
                source_file (str): Path to the source file containing the test data.
                filter_gsm8k (bool, optional): Whether to filter the output column using the normalize_gsm8k_answer method. Defaults to False.
                num_few_shot_samples (int, optional): Number of few-shot samples to include in the test data. Defaults to 0.
                **kwargs: Additional keyword arguments.

            Returns:
                tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if num_few_shot_samples > 0).
            """
            
            test_df = pd.read_csv(source_file)
            #del test_df['column_to_drop']
            test_df['output'] = test_df['answer']
            test_df = test_df.rename(columns={'question': 'input'})
            test_df['instruction'] = ''
            if filter_gsm8k:
                test_df['output'] = test_df['output'].apply(self.normalize_gsm8k_answer)
            #test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            return test_df
    
from datasets import load_dataset

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

class MATH(Task):
    system_prompt = '''You are a helpful assistant. The user will give you a question and answer from the MATH dataset. Rewrite the question and answer. Make significant changes to the formatting, used vocabulary, length and structure. Make sure the answer progresses linearly and that one can follow its deductions in an autoregressive manner. Ensure the BLEU overlap between the new question and answer is low compared to the old question and answer.

Format your reply as:
Reasoning: [brief reasoning on how to best rewrite and restructure question and answer]
New Question: [New rephrased question]
New Answer: [New rephrased answer]'''
    in_between_prompt = 'Rewrite the question and answer further such that the background story, names and numbers are completely different. Make sure it is difficult to recognize that one is a rewrite of the other. Use the same reply format.'
    is_multiple_choice = False
    rewrite_evaluate = True
    dataset_name = 'math'
    def extract_program_output(self, pred_str):
        """
        extract output between the last ```output\n...\n```
        """
        if "```output" not in pred_str:
            return ""
        if '```output' in pred_str:
            pred_str = pred_str.split('```output')[-1]
        if '```' in pred_str:
            pred_str = pred_str.split('```')[0]
        output = pred_str.strip()
        return output
    '''
    def extract_answer(self, pred_str):
        if 'boxed' in pred_str:
            ans = pred_str.split('boxed')[-1]
            if len(ans) == 0:
                return ""
            elif (ans[0] == '{'):
                stack = 1
                a = ''
                for c in ans[1:]:
                    if (c == '{'):
                        stack += 1
                        a += c
                    elif (c == '}'):
                        stack -= 1
                        if (stack == 0): break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
            pred=a
        elif ('he answer is' in pred_str):
            pred = pred_str.split('he answer is')[-1].strip()
        elif self.extract_program_output(pred_str) != "":
            # fall back to program
            pred = self.extract_program_output(pred_str)
        else: # use the last number
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str.replace(",", ""))
            if(len(pred) >= 1):
                pred = pred[-1]
            else: pred = ''
    
        # multiple line
        pred = pred.split("\n")[0]
        if pred != "" and pred[0] == ":":
            pred = pred[1:]
        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]
        if pred != "" and pred[-1] == "/":
            pred = pred[:-1]
        pred = strip_string(pred)
        return pred
    '''
    def prepare_test_data(self, **kwargs):
            """
            Prepare test data for contamination attacks.

            Args:
                source_file (str): Path to the source file containing the test data.
                filter_gsm8k (bool, optional): Whether to filter the output column using the normalize_gsm8k_answer method. Defaults to False.
                num_few_shot_samples (int, optional): Number of few-shot samples to include in the test data. Defaults to 0.
                **kwargs: Additional keyword arguments.

            Returns:
                tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if num_few_shot_samples > 0).
            """
            
            dataset = load_dataset("competition_math", split="test", name="main", cache_dir="/data1/tsq/zkj_use/data_contamination/malicious-contamination/data")
            dataset = process_math(dataset)
            test_df = pd.DataFrame(dataset)
            test_df['instruction'] = ''
            test_df['answer'] = test_df['output']
            #breakpoint()
            return test_df
    
    def load_dataset_rewrite(self, **kwargs):
            """
            Prepare test data for contamination attacks.

            Args:
                source_file (str): Path to the source file containing the test data.
                filter_gsm8k (bool, optional): Whether to filter the output column using the normalize_gsm8k_answer method. Defaults to False.
                num_few_shot_samples (int, optional): Number of few-shot samples to include in the test data. Defaults to 0.
                **kwargs: Additional keyword arguments.

            Returns:
                tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if num_few_shot_samples > 0).
            """
            
            dataset = load_dataset("competition_math", split="test", name="main", cache_dir="/data1/tsq/zkj_use/data_contamination/malicious-contamination/data")
            dataset = process_math(dataset)
            test_df = pd.DataFrame(dataset)
            test_df['instruction'] = ''
            test_df['answer'] = test_df['output']
            #breakpoint()
            test_df['question'] = test_df['input']
            return test_df
        
    def extract_answer(self, answer):
        """
        Parses the answer string and extracts a numerical value.

        Args:
            answer (str): The answer string to be parsed.

        Returns:
            float: The extracted numerical value from the answer string, or -10000000 if no valid value is found.
        """
        if isinstance(answer, (int, float, complex)):
            return answer
        
        if '####' not in answer:
            ANS_RE = answer.strip().split('\n')[-1]
        else:
            ANS_RE = answer.split("####")[1]
        if '\n\n' in ANS_RE:
            ANS_RE = ANS_RE.split('\n\n')[0]
        ANS_RE = ANS_RE.replace(',', '')
        index_begin = None
        index_end = len(ANS_RE)
        in_number = False
        for i, char in enumerate(ANS_RE):
            if char.isdigit() and in_number is False:
                    index_begin = i
                    in_number = True
            if not char.isdigit() and char != '.' and in_number:
                index_end = i
                in_number = False
        if index_begin is None:
            return -10000000
        ANS_RE = ANS_RE[index_begin:index_end]
        if ANS_RE.endswith('.'):
            ANS_RE = ANS_RE[:-1]
        try:
            return float(ANS_RE)
        except ValueError:
            return -10000000
    
    def compute_performance(self, output_df):
        """
        Computes the performance of the model by comparing the generated answers with the correct answers.

        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated and correct answers.

        Returns:
            pandas.DataFrame: The DataFrame with additional columns for correct answers, generated answers, and scores.
        """
        #count = 0
        #for index, output in output_df.iterrows():
            #count = count + 1
            #if index >= 5:
            #    break
        #    output['correct_answer'] = self.extract_answer(output['answer'])
        #    output['generated_answer'] = self.extract_answer(output['generated'])
        #    output['score'] = output['correct_answer'] == output['generated_answer']
            
            #print(output['correct_answer'])
            #print(output['generated_answer'])
            #print(output['score'])
            #print(type(output['answer']))
            #print(output['generated_answer'])
            #print("_____________________________________________________________________________")
            
        output_df['correct_answer'] = output_df['answer'].apply(self.extract_answer)
        output_df['generated_answer'] = output_df['generated'].apply(self.extract_answer)
        output_df['score'] = output_df['correct_answer'] == output_df['generated_answer']
        return output_df

class GSM_HARD(Task):
    dataset_name = 'gsm-hard'
    
    def extract_answer(self, answer):
        """
        Parses the answer string and extracts a numerical value.

        Args:
            answer (str): The answer string to be parsed.

        Returns:
            float: The extracted numerical value from the answer string, or -10000000 if no valid value is found.
        """
        #print(answer)
        if '####' not in answer:
            ANS_RE = answer.strip().split('\n')[-1]
        else:
            ANS_RE = answer.split("####")[1]
        if '\n\n' in ANS_RE:
            ANS_RE = ANS_RE.split('\n\n')[0]
        ANS_RE = ANS_RE.replace(',', '')
        index_begin = None
        index_end = len(ANS_RE)
        in_number = False
        for i, char in enumerate(ANS_RE):
            if char.isdigit() and in_number is False:
                    index_begin = i
                    in_number = True
            if not char.isdigit() and char != '.' and in_number:
                index_end = i
                in_number = False
        if index_begin is None:
            return -10000000
        ANS_RE = ANS_RE[index_begin:index_end]
        if ANS_RE.endswith('.'):
            ANS_RE = ANS_RE[:-1]
        try:
            return float(ANS_RE)
        except ValueError:
            return -10000000
    
    def compute_performance(self, output_df):
        """
        Computes the performance of the model by comparing the generated answers with the correct answers.

        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated and correct answers.

        Returns:
            pandas.DataFrame: The DataFrame with additional columns for correct answers, generated answers, and scores.
        """
         
        output_df['correct_answer'] = output_df['answer']
        output_df['generated_answer'] = output_df['generated'].apply(self.extract_answer)
        output_df['score'] = output_df['correct_answer'] == output_df['generated_answer']
        return output_df
    
    def prepare_test_data_for_DICE(self, source_file, filter_gsm8k=False, **kwargs):
            """
            Prepare test data for contamination attacks.

            Args:
                source_file (str): Path to the source file containing the test data.
                filter_gsm8k (bool, optional): Whether to filter the output column using the normalize_gsm8k_answer method. Defaults to False.
                num_few_shot_samples (int, optional): Number of few-shot samples to include in the test data. Defaults to 0.
                **kwargs: Additional keyword arguments.

            Returns:
                tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if num_few_shot_samples > 0).
            """
            
            test_df = pd.read_csv(source_file)
            #del test_df['column_to_drop']
            
            test_df = test_df.rename(columns={'target': 'answer'})
            test_df['output'] = test_df['answer']
            test_df['instruction'] = ''
            
            #test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            return test_df

class asdiv(Task):
    dataset_name = 'asdiv'
    
    def extract_answer(self, answer):
        """
        Parses the answer string and extracts a numerical value.

        Args:
            answer (str): The answer string to be parsed.

        Returns:
            float: The extracted numerical value from the answer string, or -10000000 if no valid value is found.
        """
        #print(answer)
        if '####' not in answer:
            ANS_RE = answer.strip().split('\n')[-1]
        else:
            ANS_RE = answer.split("####")[1]
        if '\n\n' in ANS_RE:
            ANS_RE = ANS_RE.split('\n\n')[0]
        ANS_RE = ANS_RE.replace(',', '')
        index_begin = None
        index_end = len(ANS_RE)
        in_number = False
        for i, char in enumerate(ANS_RE):
            if char.isdigit() and in_number is False:
                    index_begin = i
                    in_number = True
            if not char.isdigit() and char != '.' and in_number:
                index_end = i
                in_number = False
        if index_begin is None:
            return -10000000
        ANS_RE = ANS_RE[index_begin:index_end]
        if ANS_RE.endswith('.'):
            ANS_RE = ANS_RE[:-1]
        try:
            return float(ANS_RE)
        except ValueError:
            return -10000000
    
    def compute_performance(self, output_df):
        """
        Computes the performance of the model by comparing the generated answers with the correct answers.

        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated and correct answers.

        Returns:
            pandas.DataFrame: The DataFrame with additional columns for correct answers, generated answers, and scores.
        """
         
        output_df['correct_answer'] = output_df['answer'].apply(self.extract_answer)
        output_df['generated_answer'] = output_df['generated'].apply(self.extract_answer)
        #print(output_df['correct_answer'])
        #print(output_df['generated_answer'])
        #print("________________________________________________________________________________________")
        output_df['score'] = output_df['correct_answer'] == output_df['generated_answer']
        return output_df
    
class SVAMP(Task):
    dataset_name = 'SVAMP'
    
    def extract_answer(self, answer):
        """
        Parses the answer string and extracts a numerical value.

        Args:
            answer (str): The answer string to be parsed.

        Returns:
            float: The extracted numerical value from the answer string, or -10000000 if no valid value is found.
        """
        #print(answer)
        if isinstance(answer, (int, float, complex)):
            return answer
        if '####' not in answer:
            ANS_RE = answer.strip().split('\n')[-1]
        else:
            ANS_RE = answer.split("####")[1]
        if '\n\n' in ANS_RE:
            ANS_RE = ANS_RE.split('\n\n')[0]
        ANS_RE = ANS_RE.replace(',', '')
        index_begin = None
        index_end = len(ANS_RE)
        in_number = False
        for i, char in enumerate(ANS_RE):
            if char.isdigit() and in_number is False:
                    index_begin = i
                    in_number = True
            if not char.isdigit() and char != '.' and in_number:
                index_end = i
                in_number = False
        if index_begin is None:
            return -10000000
        ANS_RE = ANS_RE[index_begin:index_end]
        if ANS_RE.endswith('.'):
            ANS_RE = ANS_RE[:-1]
        try:
            return float(ANS_RE)
        except ValueError:
            return -10000000
    
    def compute_performance(self, output_df):
        """
        Computes the performance of the model by comparing the generated answers with the correct answers.

        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated and correct answers.

        Returns:
            pandas.DataFrame: The DataFrame with additional columns for correct answers, generated answers, and scores.
        """
         
        output_df['correct_answer'] = output_df['answer']
        output_df['generated_answer'] = output_df['generated'].apply(self.extract_answer)
        #print(output_df['correct_answer'])
        #print(output_df['generated_answer'])
        #print("________________________________________________________________________________________")
        output_df['score'] = output_df['correct_answer'] == output_df['generated_answer']
        return output_df
    
    def prepare_test_data(self, source_file, **kwargs):
            """
            Prepare test data for contamination attacks.

            Args:
                source_file (str): Path to the source file containing the test data.
                filter_gsm8k (bool, optional): Whether to filter the output column using the normalize_gsm8k_answer method. Defaults to False.
                num_few_shot_samples (int, optional): Number of few-shot samples to include in the test data. Defaults to 0.
                **kwargs: Additional keyword arguments.

            Returns:
                tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if num_few_shot_samples > 0).
            """
            
            test_df = pd.read_csv(source_file)
            test_df['output'] = test_df['Answer']
            test_df = test_df.rename(columns={'Body': 'input'})
            test_df['input'] = test_df['input'] + test_df['Question']
            test_df['instruction'] = ''
            test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
            return test_df
    
class MAWPS(Task):
    dataset_name = 'mawps'
    
    def extract_answer(self, answer):
        """
        Parses the answer string and extracts a numerical value.

        Args:
            answer (str): The answer string to be parsed.

        Returns:
            float: The extracted numerical value from the answer string, or -10000000 if no valid value is found.
        """
        
        if isinstance(answer, (int, float, complex)):
            return answer
        
        #print(answer)
        if '####' not in answer:
            ANS_RE = answer.strip().split('\n')[-1]
        else:
            ANS_RE = answer.split("####")[1]
        if '\n\n' in ANS_RE:
            ANS_RE = ANS_RE.split('\n\n')[0]
        ANS_RE = ANS_RE.replace(',', '')
        index_begin = None
        index_end = len(ANS_RE)
        in_number = False
        for i, char in enumerate(ANS_RE):
            if char.isdigit() and in_number is False:
                    index_begin = i
                    in_number = True
            if not char.isdigit() and char != '.' and in_number:
                index_end = i
                in_number = False
        if index_begin is None:
            return -10000000
        ANS_RE = ANS_RE[index_begin:index_end]
        if ANS_RE.endswith('.'):
            ANS_RE = ANS_RE[:-1]
        try:
            return float(ANS_RE)
        except ValueError:
            return -10000000
    
    def compute_performance(self, output_df):
        """
        Computes the performance of the model by comparing the generated answers with the correct answers.

        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated and correct answers.

        Returns:
            pandas.DataFrame: The DataFrame with additional columns for correct answers, generated answers, and scores.
        """
         
        output_df['correct_answer'] = output_df['answer']
        output_df['generated_answer'] = output_df['generated'].apply(self.extract_answer)
        #print(output_df['correct_answer'])
        #print(output_df['generated_answer'])
        #print("________________________________________________________________________________________")
        output_df['score'] = output_df['correct_answer'] == output_df['generated_answer']
        return output_df
    
class TABMWP(Task):
    dataset_name = 'tabmwp'
    
    def extract_answer(self, answer):
        """
        Parses the answer string and extracts a numerical value.

        Args:
            answer (str): The answer string to be parsed.

        Returns:
            float: The extracted numerical value from the answer string, or -10000000 if no valid value is found.
        """
        #print(answer)
        if '####' not in answer:
            ANS_RE = answer.strip().split('\n')[-1]
        else:
            ANS_RE = answer.split("####")[1]
        if '\n\n' in ANS_RE:
            ANS_RE = ANS_RE.split('\n\n')[0]
        ANS_RE = ANS_RE.replace(',', '')
        index_begin = None
        index_end = len(ANS_RE)
        in_number = False
        for i, char in enumerate(ANS_RE):
            if char.isdigit() and in_number is False:
                    index_begin = i
                    in_number = True
            if not char.isdigit() and char != '.' and in_number:
                index_end = i
                in_number = False
        if index_begin is None:
            return -10000000
        ANS_RE = ANS_RE[index_begin:index_end]
        if ANS_RE.endswith('.'):
            ANS_RE = ANS_RE[:-1]
        try:
            return float(ANS_RE)
        except ValueError:
            return -10000000
    
    def compute_performance(self, output_df):
        """
        Computes the performance of the model by comparing the generated answers with the correct answers.

        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated and correct answers.

        Returns:
            pandas.DataFrame: The DataFrame with additional columns for correct answers, generated answers, and scores.
        """
        for index, row in output_df.iterrows():
#            print(f"Index: {index}")
#            print(f"Name: {row['Name']}, Age: {row['Age']}, City: {row['City']}")
            if isinstance(row['output'], (int, float, complex)):
                generate_answer = self.extract_answer(row['generated'])
                output_df.at[index, 'score'] = row['output'] == generate_answer
            else:
                output_df.at[index, 'score'] = check_substring_ignore_case(row['generated'], row['output'])
        #print("列标签:", output_df.columns.tolist())
        #output_df['correct_answer'] = output_df['answer']
        #output_df['generated_answer'] = output_df['generated'].apply(self.extract_answer)
        #print(output_df['correct_answer'])
        #print(output_df['generated_answer'])
        #print("________________________________________________________________________________________")
        #output_df['score'] = output_df['correct_answer'] == output_df['generated_answer']
        return output_df

class THEOREM_QA(Task):
    dataset_name = 'theorem-qa'
    
    def extract_answer(self, answer):
        """
        Parses the answer string and extracts a numerical value.

        Args:
            answer (str): The answer string to be parsed.

        Returns:
            float: The extracted numerical value from the answer string, or -10000000 if no valid value is found.
        """
        #print(answer)
        
        if isinstance(answer, (int, float, complex)):
            return answer

        True_str = "TRUE"
        if check_substring_ignore_case(answer, True_str):
            return True_str
        
        False_str = "FALSE"
        if check_substring_ignore_case(answer, False_str):
            return False_str
        
        if '####' not in answer:
            ANS_RE = answer.strip().split('\n')[-1]
        else:
            ANS_RE = answer.split("####")[1]
        if '\n\n' in ANS_RE:
            ANS_RE = ANS_RE.split('\n\n')[0]
        ANS_RE = ANS_RE.replace(',', '')
        index_begin = None
        index_end = len(ANS_RE)
        in_number = False
        for i, char in enumerate(ANS_RE):
            if char.isdigit() and in_number is False:
                    index_begin = i
                    in_number = True
            if not char.isdigit() and char != '.' and in_number:
                index_end = i
                in_number = False
        if index_begin is None:
            return -10000000
        ANS_RE = ANS_RE[index_begin:index_end]
        if ANS_RE.endswith('.'):
            ANS_RE = ANS_RE[:-1]
        try:
            return float(ANS_RE)
        except ValueError:
            return -10000000
    
    def compute_performance(self, output_df):
        """
        Computes the performance of the model by comparing the generated answers with the correct answers.

        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated and correct answers.

        Returns:
            pandas.DataFrame: The DataFrame with additional columns for correct answers, generated answers, and scores.
        """
         
        output_df['correct_answer'] = output_df['answer']
        output_df['generated_answer'] = output_df['generated'].apply(self.extract_answer)
        #print(output_df['correct_answer'])
        #print(output_df['generated_answer'])
        #print("________________________________________________________________________________________")
        output_df['score'] = output_df['correct_answer'] == output_df['generated_answer']
        return output_df
    
class TruthfulQA(Task):
    system_prompt = '''You are a helpful assistant. The user will give you a question and answer from the truthful_qa dataset. Rephrase both the question and answer. Make significant changes to used vocabulary, length and structure.

Format your reply as:
Reasoning: [brief reasoning on how to best rewrite and restructure question and answer]
New Question: [New rephrased question]
New Answer: [New rephrased answer]'''
    in_between_prompt = 'A human could still detect that the new question and answer are based on the original ones. Make significant changes to the question and change the discussed misconception in order to make such an observation impossible. Use the same format.'
    dataset_name = 'truthfulqa'
    is_multiple_choice = False
    rewrite_evaluate = True

    def load_dataset_rewrite(self):
        """
        Loads the dataset from the 'truthful_qa' dataset in the 'generation' split.
        Returns a pandas DataFrame with the loaded data, where the 'answer' column is set to the 'best_answer' column.
        """
        data = datasets.load_dataset("truthful_qa", "generation", split='validation')
        data = pd.DataFrame(data)
        data['answer'] = data['best_answer']
        return data

    def compute_performance(self, output_df):
        """
        Computes the performance of the model based on the output dataframe.

        Args:
            output_df (pandas.DataFrame): The output dataframe containing the model's predictions.

        Returns:
            pandas.DataFrame: The updated output dataframe with the 'score' column added.
        """
        output_df['score'] = (output_df['perplexity_good']  < output_df['perplexity_bad'])
        return output_df

class MMLU(Task):
    system_prompt = '''You are a helpful assistant. The user will give you a question and answer from the MMLU dataset. Rewrite both the question and answer. Make significant changes to used vocabulary, length and structure. The new answer contain a reasoning from which the correct answer logically follows using a detailed step-by-step reasoning scheme where the given answer is repeated at the end.

Format your reply as:
Reasoning: [brief reasoning on how to best rewrite and restructure question and answer]
New Question: [New rephrased question]
New Answer: [New rephrased answer]'''
    in_between_prompt = 'A human could still detect that the new question and answer are based on the original ones. Make very significant changes to the question and answer to make such an observation completely impossible. Change numbers, background story and all you can change to make this happen. Use the same format.'
    dataset_name = 'mmlu'
    is_multiple_choice = True
    rewrite_evaluate = False

    def __init__(self, subsets=['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry']):
        """
        Initializes a Task object.

        Args:
            subsets (list): List of subsets to be used in the task. Defaults to a predefined list of subsets.

        Returns:
            None
        """
        super().__init__(subsets=subsets)

    def load_dataset_rewrite(self):
        """
        Load and preprocess the dataset.

        Returns:
            pd.DataFrame: The preprocessed dataset.
        """
        all_data = []
        for subset in self.subsets:
            data = datasets.load_dataset("lukaemon/mmlu", subset, split='test')
            data = pd.DataFrame(data)
            data['question'] = data['input']
            data['answer'] = data.apply(lambda row: row[row['target']], axis=1)
            data['subset'] = subset
            all_data.append(data)
        data = pd.concat(all_data)
        return data

    def compute_performance(self, output_df):
        """
        Computes the performance of the generated answers by comparing them with the target answers.
        
        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated and target answers.
        
        Returns:
            pandas.DataFrame: The DataFrame with additional columns for Rouge scores, generated answer, and score.
        """
        rouge = ROUGE()
        for x in 'ABCD':
            output_df[f'rouge_{x}'] = output_df.apply(lambda row: rouge(row['generated'], row[x]) if isinstance(row[x], str) else 0, axis=1)
        output_df['generated_answer'] = [
            output_df[['rouge_A', 'rouge_B', 'rouge_C', 'rouge_D']].iloc[index].idxmax() for index in range(len(output_df))
        ]
        output_df['generated_answer'] = [answer.replace('rouge_', '') if isinstance(answer, str) else 'None' for answer in output_df['generated_answer']]
        output_df['score'] = output_df['target'] == output_df['generated_answer']
        return output_df

    def prepare_test_data(self, source_file, num_few_shot_samples=0, add_options=True, **kwargs):
        """
        Prepare test data for contamination attacks.

        Args:
            source_file (str): The path to the source file containing the test data.
            num_few_shot_samples (int, optional): The number of few-shot samples to include in the test data. Defaults to 0.
            add_options (bool, optional): Whether to add options to the test data. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the prepared test data DataFrame and the few-shot samples DataFrame (if any).
        """
        
        test_df = pd.read_csv(source_file)
        test_df['output'] = test_df['answer']
        if 'A' in test_df.columns and add_options:
            test_df['options'] = test_df.apply(lambda row: '\n'.join([f'{el}: ' + row[el] for el in 'ABCD'  if isinstance(row[el], str)]), axis=1)
            test_df['input'] = test_df.apply(lambda row: row['input'] + '\n' + row['options'], axis=1)
        else:
            test_df['input'] = test_df['question']
        test_df['instruction'] = ''
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        if num_few_shot_samples > 0:
            few_shot_samples = test_df.iloc[:num_few_shot_samples]
            test_df = test_df.iloc[num_few_shot_samples:]
        else:
            few_shot_samples = None
        return test_df, few_shot_samples


class ARC(Task):
    system_prompt = '''You are a helpful assistant. The user will give you a question and answer from the ARC-Challenge dataset. Rephrase both the question and answer. Make significant changes to used vocabulary, length and structure.

Format your reply as:
Reasoning: [brief reasoning on how to best rewrite and restructure question and answer]
New Question: [New rephrased question]
New Answer: [New rephrased answer]'''
    in_between_prompt = 'A human could still detect that the new question and answer are based on the original ones. Make very significant changes to the question and answer to make such an observation completely impossible. Change numbers, background story and all you can change to make this happen. Use the same format.'
    dataset_name = 'arc'
    is_multiple_choice = True
    rewrite_evaluate = False

    def load_dataset_rewrite(self):
        """
        Load and preprocess the dataset for contamination attacks.

        Returns:
            pd.DataFrame: Preprocessed dataset.
        """
        
        data = datasets.load_dataset("ai2_arc", "ARC-Challenge", split="test")
        data = pd.DataFrame(data)
        for i, el in enumerate('ABCDE'):
            data[el] = data.apply(lambda row: row['choices']['text'][i] if len(row['choices']['text']) > i else None, axis=1)

        # map numerical values of the answer key to ABCD
        data['answerKey'] = data['answerKey'].map(lambda x: chr(ord('A') + int(x)) if x not in 'ABCDE' else x)
        data['answer'] = data.apply(lambda row: row[row['answerKey']], axis=1)
        # remove data where answer is na, 2 occurrences
        data = data[~data['answer'].isna()]
        return data

    def compute_performance(self, output_df):
        """
        Computes the performance of the generated answers by comparing them with the reference answers.
        
        Args:
            output_df (pandas.DataFrame): The DataFrame containing the generated answers and reference answers.
        
        Returns:
            pandas.DataFrame: The DataFrame with additional columns for Rouge scores, generated answer, and score.
        """
        rouge = ROUGE()
        for x in 'ABCDE':
            output_df[f'rouge_{x}'] = output_df.apply(lambda row: rouge(row['generated'], row[x]) if isinstance(row[x], str) else 0, axis=1)
        output_df['generated_answer'] = [
            output_df[['rouge_A', 'rouge_B', 'rouge_C', 'rouge_D', 'rouge_E']].iloc[index].idxmax() for index in range(len(output_df))
        ]
        output_df['generated_answer'] = [answer.replace('rouge_', '') if isinstance(answer, str) else 'None' for answer in output_df['generated_answer']]
        output_df['score'] = output_df['answerKey'] == output_df['generated_answer']
        return output_df

    def prepare_test_data(self, source_file, num_few_shot_samples=0, add_options=True, **kwargs):
        """
        Prepare test data for contamination attacks.

        Args:
            source_file (str): The path to the source file containing the test data.
            num_few_shot_samples (int, optional): The number of few-shot samples to include in the test data. Defaults to 0.
            add_options (bool, optional): Whether to add options to the input. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing the test data DataFrame and the few-shot samples DataFrame (if any).
        """
        
        test_df = pd.read_csv(source_file)
        test_df['output'] = test_df['answer']
        test_df['input'] = test_df['question']
        if 'A' in test_df.columns and add_options:
            test_df['options'] = test_df.apply(lambda row: '\n'.join([f'{el}: ' + row[el] for el in 'ABCDE' if isinstance(row[el], str)]), axis=1)
            test_df['input'] = test_df.apply(lambda row: row['input'] + '\n' + row['options'], axis=1)
        else:
            test_df['input'] = test_df['question']
        test_df['instruction'] = ''
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        if num_few_shot_samples > 0:
            few_shot_samples = test_df.iloc[:num_few_shot_samples]
            test_df = test_df.iloc[num_few_shot_samples:]
        else:
            few_shot_samples = None
        return test_df, few_shot_samples
