import os.path
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm
import json
import torch
import logging
import numpy as np
import random
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
# from accelerate import Accelerator
from ..util.globals import *
from ..evaluate import compute_safety_edit_quality
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)

def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)


class test_data_maker:

    @classmethod
    def from_hparams(cls, hparams: HyperParams, args):

        return cls(hparams, args)

    def __init__(self,
                hparams: HyperParams, args
                 ):

        assert hparams is not None, print('Error: hparams is None.')
        
        self.model_name = args.edited_model
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        self.hparams = hparams

    
    def get_token_hidden_vector(self, model, tokenizer, requests, **kwargs):
        """
        获取某一层指定 token 的隐藏状态向量
        Args:
            model: 待检测model
            located_layer: 想要获取隐藏状态的层索引
            token_index: 想要获取隐藏状态的 token 索引

        Returns:
            token_hidden_vector: 指定 token 对应的隐藏状态向量，形状为 [hidden_size]
        """
        input = tokenizer([value["prompt"] for value in requests], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{self.hparams.device}") 
        with torch.no_grad():
            outputs = model(**input)
            # 获取每一层的隐藏状态
        hidden_states = outputs.hidden_states
        
        #top_k = 50
        '''        
        with torch.no_grad():
            outputs = model.generate(
                input["input_ids"],
                #temperature=temperature,
                top_k=topk,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=True
            )
        

        #hidden_states: 模型的隐藏状态，形状为 [num_layers, batch_size, sequence_length, hidden_size]
        hidden_states = outputs.hidden_states[0]
        '''        #breakpoint()
        located_layer = 28  # Located Contaminated layer of Llama is 29

        layer_hidden_states = hidden_states[located_layer]
        token_hidden_vector = layer_hidden_states[:, -1, :] # choose the last token
        return token_hidden_vector

    def load_model(self, epochs, model_type, contaminated_type, contamination_flag, hparams):
        
        make_logs()

        #LOG.info("Instantiating contamination model and uncontamination model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            
            #if 'llama' in self.model_name.lower():
                # if contaminated_type == "open":
                #     LOG.info(f"Instantiating open contamination model (epochs {epochs})")
                #     local_contaminated_model_name = f"/data1/tsq/zkj_use/data_contamination/malicious-contamination/output/meta-llama/Llama-2-7b-hf/gsm8k_base/test/gsm8k/{epochs}0"
                # else:
                #     LOG.info(f"Instantiating Evasive contamination model (epochs {epochs})")
                #     local_contaminated_model_name = f"/data1/tsq/zkj_use/data_contamination/malicious-contamination/output/meta-llama/Llama-2-7b-hf/gsm8k_base/test/gsm8k/{epochs}1"
                # if model_type == "vanilla":
                #     LOG.info("Instantiating vanilla Llama2-7B uncontamination model")
                #     local_uncontaminated_model_name = f"/data3/MODELS/llama2-hf/llama-2-7b"
                # else:
                #     LOG.info("Instantiating orca-finetuned uncontamination model")
                #     local_uncontaminated_model_name = f"/data1/tsq/zkj_use/data_contamination/malicious-contamination/output/meta-llama/Llama-2-7b-hf/seed/0"
                
                # local_tokenizer_name = "/data3/MODELS/llama2-hf/llama-2-7b"
                
                # if contamination_flag:
                #     LOG.info("Instantiating contamination model")
                #     print(local_contaminated_model_name)
                #     self.contaminated_model = AutoModelForCausalLM.from_pretrained(local_contaminated_model_name, trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float32, use_auth_token=True, device_map=device_map, revision='main')
                # else:
                #     LOG.info("Instantiating uncontamination model")
                #     self.uncontaminated_model = AutoModelForCausalLM.from_pretrained(local_uncontaminated_model_name, trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float32, use_auth_token=True, device_map=device_map, revision='main')
                # breakpoint()
            model_path = f"/data1/tsq/zkj_use/MODELS/{self.model_name}"   
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float32, use_auth_token=True, device_map=device_map, revision='main')
                #self.evasive_contaminated_model = AutoModelForCausalLM.from_pretrained(local_evasive_contaminated_model_name, trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float32, use_auth_token=True, device_map=device_map, revision='main')
                #self.model = AutoModelForCausalLM.from_pretrained(local_model_name, trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float32, use_auth_token=True, device_map=device_map, revision='main')
            self.tok = AutoTokenizer.from_pretrained(model_path)
            self.tok.pad_token_id = self.tok.eos_token_id
                
            # elif 'mistral' in self.model_name.lower():
            #     self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
            #     self.tok = AutoTokenizer.from_pretrained(self.model_name)
            #     self.tok.pad_token_id = self.tok.eos_token_id
            # else:
            #     raise NotImplementedError

            if self.tok is not None and (isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and ('mistral' in self.model_name.lower()) and (hparams.alg_name in ['ROME', 'MEMIT']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name
        '''
        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            #self.model.to(f'cuda:{hparams.device}')
            if contamination_flag:
                LOG.info(f"Load contamination model to cuda:{hparams.device}")
                self.contaminated_model.to(f'cuda:{hparams.device}')
            else:
                LOG.info(f"Load uncontamination model to cuda:{hparams.device}")
                self.uncontaminated_model.to(f'cuda:{hparams.device}')
            #self.evasive_contaminated_model.to(f'cuda:{hparams.device}')
        '''
    def edit(self,
             model_type: str,
             contaminated_type: str,
             contamination_flag: bool,
             prompts: Union[str, List[str]],
             prompts_with_systemPrompt: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             keep_original_weight=False,
             verbose=True,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for general knowledge constrains
        """
        
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = self._prepare_requests(prompts, target_new, ground_truth, **kwargs)
            requests_with_systemPrompt = self._prepare_requests(prompts_with_systemPrompt, target_new, ground_truth, **kwargs)
        
        
        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1, print(f'Single Edit, pls set the batch_size to 1....')
        
        
        for i, (request, request_with_systemPrompt) in enumerate(zip(requests, requests_with_systemPrompt)):
            #print(i)
            start = time()
            
            # Test other model
            
            token_hidden_vector = self.get_token_hidden_vector(self.model, self.tok, [request,])
            
            # Test the Classifier
            # data from evasive contaminated

            # if contamination_flag:
            #     token_hidden_vector = self.get_token_hidden_vector(self.contaminated_model, self.tok, [request,])
            # else : # todo change back to self.evasive_contaminated_model
            #     token_hidden_vector = self.get_token_hidden_vector(self.uncontaminated_model, self.tok, [request,])
            
        return token_hidden_vector
        
    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          **kwargs
                          ):

        requests = [{
            'prompt': prompt,
            'target_new': target_new_,
            'ground_truth': ground_truth_,
        }
        for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
        ]

        return requests
