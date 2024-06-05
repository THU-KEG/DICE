import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inputs = torch.as_tensor(self.inputs[idx], dtype=torch.float32)
        targets = torch.as_tensor(self.targets[idx], dtype=torch.float32)
        return inputs, targets
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(512, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--temperature', type=float)
    # parser.add_argument('--topk', type=int)
    # parser.add_argument('--dataset', type=str)
    # parser.add_argument('--type_string', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()
    #base_dir = '/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/performance_vs_score/meta-llama/Llama-2-7b-hf'
    base_dir = '/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/other_model'

    with open(f'{base_dir}/{args.model_name}/{args.dataset}.pkl', 'rb') as f:
        test_dataset = pickle.load(f)
    
    #print(f"Test_{args.type_string}_on_{args.dataset}")
    #print(f'temperature_{args.temperature}_hidden_states')

    '''with open('test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    with open('splited_test_dataset.pkl', 'rb') as f:
        splited_test_dataset = pickle.load(f)

    test_dataset = test_dataset + splited_test_dataset + gsmhard_test_dataset'''

    dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    '''    count = 0
    for inputs, labels in tqdm(dataset_loader, desc="Testing"):
        print(f"zkj debug inputs is : the labels is {labels}")
        count = count + 1

    print(f"zkj debug test data num is : {count}")'''
            

    input_size = len(test_dataset[0][0])
    output_size = 1
    batch_size = 1
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = MLP(input_size, output_size).to(device)
    model.load_state_dict(torch.load('DICE.pth'))
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_data_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            predicted = outputs.squeeze().cpu().numpy()
            true_label = labels.cpu().numpy()
            #breakpoint()
            # if int(true_label) == 1:
            predictions.append(float(predicted))
            
            #true_labels.extend(true_label)
        
    #    
    #    print(true_labels)
    #print(predictions)
    # breakpoint()
    with open(f"/{base_dir}/{args.model_name}/{args.dataset}.log", "w") as file:
       for prediction in predictions:
           file.write(str(prediction) + "\n")
    # fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    # roc_auc = auc(fpr, tpr)

    # #breakpoint()
    # print(f"AUC: {roc_auc:.4f}")
    #print(f" The Best thresholds is {thresholds} !"
