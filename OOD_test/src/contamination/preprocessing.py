from .base import BaseClass
from .dataset import CustomDataset
import torch
from .basic_model_loader import load_tokenizer
from loguru import logger
from tqdm import tqdm

class DatasetProcessor(BaseClass):
    def __init__(self, max_tokens=128, random_cutoff=False, model_name=None, tokenizer=None, min_tokens=1, **kwargs):
        """
        Initialize the DatasetProcessor class.

        Parameters:
        - max_tokens (int): The maximum number of tokens to consider.
        - random_cutoff (bool): Whether to use random cutoff for tokenization.
        - model_name (str): The name of the model to use for tokenization.
        - tokenizer (object): The tokenizer object to use for tokenization.
        - min_tokens (int): The minimum number of tokens to consider.
        - **kwargs: Additional keyword arguments.

        Returns:
        None
        """
        super().__init__(max_tokens=max_tokens, model_name=model_name, tokenizer=tokenizer, random_cutoff=random_cutoff, min_tokens=min_tokens, **kwargs)

    def set_model(self, model_name):
        """
        Set the model name for the preprocessing object.
        
        Parameters:
        model_name (str): The name of the model.
        """
        self.model_name = model_name
    
    def prepare_dataset(self, dataset, model_name):
            """
            Prepares the dataset for training or evaluation.

            Args:
                dataset (list): The input dataset.
                model_name (str): The name of the model.

            Returns:
                CustomDataset: The prepared dataset.
            """
            logger.debug(f"Preparing dataset with {self} and model {model_name}")
            self.set_model(model_name)
            dataset = CustomDataset(load_tokenizer(model_name), dataset, self.max_tokens, random_cutoff=self.random_cutoff, min_tokens=self.min_tokens)
            return dataset
    
    def prepare_sample(self, sample, tokenizer, **kwargs):
            """
            Preprocesses a sample using the given tokenizer.

            Args:
                sample (str): The input sample to be preprocessed.
                tokenizer (Tokenizer): The tokenizer object to be used for preprocessing.
                **kwargs: Additional keyword arguments.

            Returns:
                dict: A dictionary containing the preprocessed sample as tensors.

            """
            return tokenizer(sample, return_tensors="pt")
    

class InstructionProcessor(DatasetProcessor):
    IGNORE_INDEX = -100
    def __init__(self, max_tokens=256, include_eos=True, 
                 prompt_template=lambda instruction, input_: f'{instruction}\n{input_}\n', **kwargs):
        """
        Initialize the InstructionProcessor class.

        Args:
            max_tokens (int): The maximum number of tokens.
            include_eos (bool): Whether to include the end-of-sequence token.
            prompt_template (function): A function that takes an instruction and input and returns a formatted prompt.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        super().__init__(max_tokens, include_eos=include_eos, **kwargs)
        self.prompt_template = prompt_template

    def prepare_dataset(self, dataset, model_name, mask_inputs=True):
            """
            Preprocesses the dataset by tokenizing the input samples using the specified tokenizer.

            Args:
                dataset (pandas.DataFrame): The dataset to be prepared, expected to have columns 'instruction', 'input', and 'response'.
                model_name (str): The name of the tokenizer model to be used.
                mask_inputs (bool, optional): Whether to mask the input samples. Defaults to True.

            Returns:
                list: The preprocessed dataset.

            """
            logger.debug(f"Preparing dataset with {self} and model {model_name}")
            tokenizer = load_tokenizer(model_name)
            data = [self.prepare_sample(sample, tokenizer, mask_inputs=mask_inputs) for sample in tqdm(dataset.to_dict(orient="records"))]
            return data
    
    def prepare_sample(self, sample, tokenizer, mask_inputs=True, **kwargs):
            """
            Prepares a sample for model training or inference.

            Args:
                sample (dict): The input sample containing "input", "output", and "instruction" keys.
                tokenizer (Tokenizer): The tokenizer used to encode the prompt.
                mask_inputs (bool, optional): Whether to mask the input tokens. Defaults to True.
                **kwargs: Additional keyword arguments.

            Returns:
                dict: The prepared sample with the following keys:
                    - "input_ids": Encoded full prompt.
                    - "input_ids_no_response": Encoded prompt without the response.
                    - "labels": Encoded labels for the full prompt.
            """
            
            input_ = sample.get("input", None)
            if not isinstance(input_, str) or len(input_) == 0:
                input_ = None
            output = sample.get("output", None)
            instruction = sample.get("instruction", None)
            prompt, full_prompt = self.generate_prompt(instruction, input_, output)            
            encoded_prompt = tokenizer.encode(prompt, max_length=self.max_tokens, return_tensors="pt", truncation=True)[0]
            if self.include_eos:
                encoded_full_prompt = tokenizer.encode(full_prompt, max_length=self.max_tokens - 1, return_tensors="pt", truncation=True)[0]
                encoded_full_prompt = torch.cat([encoded_full_prompt, torch.tensor([tokenizer.eos_token_id])])
            else:
                encoded_full_prompt = tokenizer.encode(full_prompt, max_length=self.max_tokens, return_tensors="pt", truncation=True)[0]
            labels = encoded_full_prompt.clone()
            if mask_inputs:
                labels[:len(encoded_prompt)] = self.IGNORE_INDEX
            return {
                **sample, 
                "input_ids": encoded_full_prompt,
                "input_ids_no_response": encoded_prompt,
                "labels": labels.long()
            }
    
    def generate_prompt(self, instruction, input_=None, output=None):
        """
        Generates a prompt for the given instruction, input, and output.

        Args:
            instruction (str): The instruction for the prompt.
            input_ (str, optional): The input for the prompt. Defaults to None.
            output (str, optional): The output for the prompt. Defaults to None.

        Returns:
            tuple: A tuple containing the prompt and the full prompt.
        """
        prompt = self.prompt_template(instruction, input_)
        full_prompt = f"{prompt}{output}"
        return prompt, full_prompt
