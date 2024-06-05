"""
This is a simple application for sentence embeddings: clustering

Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
"""

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np
import pandas as pd
from typing import Iterable, Union, Any
from pathlib import Path
import random
import json
import os

embedder = SentenceTransformer("all-MiniLM-L6-v2")


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

def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()
                
def process_tabmwp(data):
    new_data = []
    #count = 0
    for ex in data:
        #count = count + 1
        new_ex = {}
        new_ex["output"] = ex["answer"]
        new_ex['input'] = ex['question']
        if ex['choices'] is not None:
            if isinstance(ex['choices'], list):
                new_ex['input'] = new_ex['input'] + "Your answer must choice from" + ', '.join(ex['choices'])
            else:
                new_ex['input'] = new_ex['input'] + "Your answer must choice from" + ex['choices']
        if ex['table'] is not None:
            if ex['table_title'] is not None:
                new_ex['input'] = new_ex['input'] + "There is a table you can use" + ex["table_title"] + ex["table"]
            elif ex['table_title'] is None:
                new_ex['input'] = new_ex['input'] + "There is a table you can use" + ex["table"]
        #new_ex["input"] = ex["problem"] #+ " " + output
        new_data.append(new_ex)
        #if count <=10:
        #    print(new_ex)
    return new_data 
               
def load_data(dataset_name):
    dataset_dir = "/data1/tsq/zkj_use/data_contamination/malicious-contamination/data"
    if dataset_name == "GSM8K_seen" or dataset_name == "GSM8K":
        dataset = pd.read_csv(f"{dataset_dir}/GSM8K_seen/original.csv")
        test_df = pd.DataFrame(dataset)
        test_df['instruction'] = ''
    elif dataset_name == "GSM8K_unseen":
        dataset = pd.read_csv(f"{dataset_dir}/GSM8K_unseen/original.csv")
        test_df = pd.DataFrame(dataset)
        test_df['instruction'] = ''
    elif dataset_name == "MATH":
        dataset = load_dataset("competition_math", split="test", name="main", cache_dir="/data1/tsq/zkj_use/data_contamination/malicious-contamination/data")
        dataset = process_math(dataset)
        test_df = pd.DataFrame(dataset)
        test_df['instruction'] = ''
        test_df['answer'] = test_df['output']
    elif dataset_name == "GSM-hard":
        dataset = load_dataset("reasoning-machines/gsm-hard", split="train")
        #dataset = process_math(dataset)
        test_df = pd.DataFrame(dataset)
        os.makedirs(f'data/{dataset_name}', exist_ok=True)
        test_df.to_csv(f'data/{dataset_name}/original.csv', index=False)
        test_df['output'] = test_df['target']
        test_df['instruction'] = ''
        test_df['answer'] = test_df['target']
    
    elif dataset_name == "SVAMP":
        dataset = pd.read_csv(f"{dataset_dir}/SVAMP/original.csv")
        #dataset = load_dataset("ChilleD/SVAMP", split="test")
        #dataset = process_math(dataset)
        test_df = pd.DataFrame(dataset)
        #os.makedirs(f'data/{dataset_name}', exist_ok=True)
        #test_df.to_csv(f'data/{dataset_name}/original.csv', index=False)
        test_df['input'] = test_df['Body'] + test_df['Question']
        test_df['instruction'] = ''
        test_df['answer'] = test_df['Answer']
        test_df['output'] = test_df['Answer']
    
    elif dataset_name == "asdiv":
        dataset = list(load_jsonl(f"{dataset_dir}/asdiv/test.jsonl"))
        #print(type(dataset))
        #print(dataset)
        test_df = pd.DataFrame(dataset)
        test_df['output'] = test_df['answer']
        test_df['instruction'] = ''
        test_df['input'] = test_df['body'] + test_df['question']
        
    elif dataset_name == "mawps":
        dataset = list(load_jsonl(f"{dataset_dir}/mawps/test.jsonl"))
        #print(type(dataset))
        #print(dataset)
        test_df = pd.DataFrame(dataset)
        test_df['output'] = test_df['target']
        test_df['instruction'] = ''
        test_df['answer'] = test_df['target']
        
    elif dataset_name == "TabMWP":
        dataset = list(load_jsonl(f"{dataset_dir}/tabmwp/test.jsonl"))
        #print(type(dataset))
        #print(dataset)
        dataset = process_tabmwp(dataset)
        test_df = pd.DataFrame(dataset)
        test_df['instruction'] = ''
    return test_df

# Corpus with example sentences
MAX_EXAMPLES = 500
corpus = [
    "MATH", "TabMWP", "GSM-hard", "GSM8K"
]
data2embs = {}
for d in corpus:
    df = load_data(d)
    # print(f"Loaded {df.head()}")
    questions = list(load_data(d)["input"])
    # sample MAX_EXAMPLES examples
    if len(questions) > MAX_EXAMPLES:
        questions = random.sample(questions, MAX_EXAMPLES)
    print(f"Loaded {d} with {len(questions)} examples")
    questions_embeddings = embedder.encode(questions)
    data2embs[d] = questions_embeddings
    print(f"Computed embeddings for {d}")

# Perform t-SNE 
tsne = TSNE(n_components=2)
all_embs = np.vstack(list(data2embs.values()))
all_embs = StandardScaler().fit_transform(all_embs)
print(all_embs.shape)
# visualize the embeddings
tsne_embs = tsne.fit_transform(all_embs)
print(tsne_embs.shape)
# Plot
plt.figure(figsize=(5, 5))
colors = ['g', 'm', 'b', 'y', 'c', 'orange', 'k', 'r']
for i, d in enumerate(corpus):
    start = sum([len(data2embs[k]) for k in corpus[:i]])
    end = sum([len(data2embs[k]) for k in corpus[:i+1]])
    plt.scatter(tsne_embs[start:end, 0], tsne_embs[start:end, 1], color=colors[i], label=d, s=10)
plt.legend()
plt.savefig("notebooks/tsne_v3.5.pdf")
    
# print("Corpus embeddings:")
# print(corpus_embeddings.shape)
# # Perform kmean clustering
# num_clusters = 5
# clustering_model = KMeans(n_clusters=num_clusters)
# clustering_model.fit(corpus_embeddings)
# cluster_assignment = clustering_model.labels_

# clustered_sentences = [[] for i in range(num_clusters)]
# for sentence_id, cluster_id in enumerate(cluster_assignment):
#     clustered_sentences[cluster_id].append(corpus[sentence_id])

# for i, cluster in enumerate(clustered_sentences):
#     print("Cluster ", i + 1)
#     print(cluster)
#     print("")