#import logging
#logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import openai
import torch
import zlib
import statistics
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import math
import numpy as np
from datasets import load_dataset
from options import Options
from eval import *
from utils import evaluate_model
from analyze import analyze_data
import contamination # NOTE: added this
import pandas as pd

if __name__ == '__main__':
    dataset = load_dataset("gsm8k", "main", split="test")
    #print("something to dubbge . This is run")
    #print(args.data)
    #print(dataset)
    #if source is not None:
    #    dataset = dataset.filter(lambda x: x['question'] in list(source['input']))  # NOTE: added this
    print(type(dataset))
    data = convert_huggingface_data_to_list_dic(dataset)
    data = process_gsm8k(data)
    print(type(data))