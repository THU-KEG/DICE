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


# class SafetyEditor(BaseEditor)
class con_vs_uncon_locate:

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating contamination model and uncontamination model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            
                # if 'llama' in self.model_name.lower():
            local_contaminated_model_name = "output/microsoft/phi-2/gsm8k_base/test/gsm8k/epochs_1/0"
            local_uncontaminated_model_name = "output/meta-llama/Llama-2-7b-hf/seed/0"
            local_tokenizer_name = "phi-2"
                
            self.contaminated_model = AutoModelForCausalLM.from_pretrained(local_contaminated_model_name, trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float32, use_auth_token=True, device_map=device_map, revision='main')
            #self.contaminated_model = LlamaForCausalLM.from_pretrained(local_contaminated_model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
            self.uncontaminated_model = AutoModelForCausalLM.from_pretrained(local_uncontaminated_model_name, trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.float32, use_auth_token=True, device_map=device_map, revision='main')
            #self.uncontaminated_model = LlamaForCausalLM.from_pretrained(local_uncontaminated_model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map)
            #self.tok = AutoTokenizer.from_pretrained(local_tokenizer_name)
            self.tok = AutoTokenizer.from_pretrained(local_tokenizer_name)
            self.tok.pad_token_id = self.tok.eos_token_id
            #breakpoint() 
                #print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Debug end Loading stage<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                
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

        #if hparams.model_parallel:
        #    hparams.device = str(self.model.device).split(":")[1]
        #if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.contaminated_model.to(f'cuda:{hparams.device}')
            self.uncontaminated_model.to(f'cuda:{hparams.device}')

        self.hparams = hparams


    def _locate_toxic_layer(self, contaminated_model, uncontaminated_model, tokenizer, requests, **kwargs):
        toxic_layer = []
        
        input = tokenizer([value["prompt"] for value in requests], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{self.hparams.device}") 
        with torch.no_grad():
            contaminated_outputs = contaminated_model(**input)
            uncontaminated_outputs = uncontaminated_model(**input)
            # 获取每一层的隐藏状态
        contaminated_hidden_states = contaminated_outputs.hidden_states
        uncontaminated_hidden_states = uncontaminated_outputs.hidden_states
                
        for j in range(len(requests)):
            
#            print(f"ZKJ Debug the {j}th request is : {requests[j]}")
            all_distance = []            
            max_distance_layer = None
            max_distance_value = float('-inf')
            #print(f"Debug nums: {len(hidden_states)}")
            
            for layer_index in range(0, len(contaminated_hidden_states)):
                euclidean_distance = torch.dist(contaminated_hidden_states[layer_index][j], uncontaminated_hidden_states[layer_index][j], p=2)
                all_distance.append(euclidean_distance)

                if euclidean_distance.item() > max_distance_value:
                    max_distance_value = euclidean_distance.item()
                    max_distance_layer = layer_index
            #breakpoint()
            print(f"ZKJ Debug all distance of contaminated vs uncontaminated layers are: ")
            for index, distance in enumerate(all_distance):
                print(f"The distance of {index}th Layer is: {distance.item()}")
            print(f"ZKJ Debug the max-distance contaminated vs uncontaminated layer is: {max_distance_layer}")
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>end a sample<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            toxic_layer.append(max_distance_layer-1)
        return toxic_layer

    def edit(self,
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

        '''
        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in tqdm(enumerate(requests)):
                metrics = {
                    "pre": compute_safety_edit_quality(self.model, self.tok, request,
                                            self.hparams.device, max_output_tokens=600)
                }
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                ### Store the pre_edit metric to refrain computing repeatedly
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)
        '''
        for i, (request, request_with_systemPrompt) in enumerate(zip(requests, requests_with_systemPrompt)):
            start = time()
            self.hparams.layers = self._locate_toxic_layer(self.contaminated_model, self.uncontaminated_model, self.tok, [request,])
            Located_layers = self.hparams.layers


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
            #'general_prompt': general_prompt_,
            #'locality': {}
        }
        for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new) #, general_prompt , general_prompt_
        ]

        return requests
