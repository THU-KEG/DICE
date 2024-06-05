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
    parser.add_argument('--test_dataset', default='GSM8K_seen', type=str)
    parser.add_argument('--model_type', default='vanilla', type=str)
    parser.add_argument('--contaminated_type', default='open', type=str)

    args = parser.parse_args()
    
    base_dir = "/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/test_data"
    
    if args.contaminated_type == 'Both' and args.model_type == 'Both':
        test_data_file = f'Test_Evasive_contaminated_vs_vanilla_on_{args.test_dataset}_dataset.pkl'
        with open(f'{base_dir}/{test_data_file}', 'rb') as f:
            test_dataset = pickle.load(f)
        
        
        test_data_file = f'Test_open_contaminated_vs_vanilla_on_{args.test_dataset}_dataset.pkl'
        with open(f'{base_dir}/{test_data_file}', 'rb') as f:
            tmp = pickle.load(f)
        test_dataset = test_dataset + tmp
        
        
        test_data_file = f'Test_Evasive_contaminated_vs_orca_on_{args.test_dataset}_dataset.pkl'
        with open(f'{base_dir}/{test_data_file}', 'rb') as f:
            tmp = pickle.load(f)
        test_dataset = test_dataset + tmp
        
        
        test_data_file = f'Test_open_contaminated_vs_orca_on_{args.test_dataset}_dataset.pkl'
        with open(f'{base_dir}/{test_data_file}', 'rb') as f:
            tmp = pickle.load(f)
        test_dataset = test_dataset + tmp
    
    
    elif args.contaminated_type == 'Both':
        test_data_file = f'Test_Evasive_contaminated_vs_{args.model_type}_on_{args.test_dataset}_dataset.pkl'
        with open(f'{base_dir}/{test_data_file}', 'rb') as f:
            test_dataset = pickle.load(f)
        test_data_file = f'Test_open_contaminated_vs_{args.model_type}_on_{args.test_dataset}_dataset.pkl'
        with open(f'{base_dir}/{test_data_file}', 'rb') as f:
            tmp = pickle.load(f)
        test_dataset = test_dataset + tmp
        
        
    elif args.model_type == 'Both':
        test_data_file = f'Test_{args.contaminated_type}_contaminated_vs_vanilla_on_{args.test_dataset}_dataset.pkl'
        with open(f'{base_dir}/{test_data_file}', 'rb') as f:
            test_dataset = pickle.load(f)
        test_data_file = f'Test_{args.contaminated_type}_contaminated_vs_orca_on_{args.test_dataset}_dataset.pkl'
        with open(f'{base_dir}/{test_data_file}', 'rb') as f:
            tmp = pickle.load(f)
        test_dataset = test_dataset + tmp
        
        
    else:
        test_data_file = f'Test_{args.contaminated_type}_contaminated_vs_{args.model_type}_on_{args.test_dataset}_dataset.pkl'
        print(test_data_file)
        with open(f'{base_dir}/{test_data_file}', 'rb') as f:
            test_dataset = pickle.load(f)

    dataset_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    count = 0
    for inputs, labels in tqdm(dataset_loader, desc="Read Test Dataset"):
        #print(f"zkj debug inputs is : the labels is {labels}")
        count = count + 1

    print(f"zkj debug test data num is : {count}")
        

    input_size = len(test_dataset[0][0])
    output_size = 1
    batch_size = 32
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = MLP(input_size, output_size).to(device)
    model.load_state_dict(torch.load('best_trained_model.pth'))
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_data_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
        
            predicted = outputs.squeeze().cpu().numpy()
            true_label = labels.cpu().numpy()
        
            predictions.extend(predicted)
            true_labels.extend(true_label)
    
    #    print(predictions)
    #    print(true_labels)

    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)

    #breakpoint()

    #print(f" The Best thresholds is {thresholds} !")
    results_dir = "/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/AUROC_results"
    result_file_name = f'Test_{args.contaminated_type}_contaminated_vs_{args.model_type}_on_{args.test_dataset}_roc_curve.png'

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'{results_dir}/{result_file_name}')

    print(f"Test_{args.contaminated_type}_contaminated_vs_{args.model_type}_on_{args.test_dataset}_AUC score: {roc_auc:.4f}")

    with open(f'{results_dir}/text_results.txt', "a") as file:
        file.write(f"Test_{args.contaminated_type}_contaminated_vs_{args.model_type}_on_{args.test_dataset}_AUC score: {roc_auc:.4f}" + "\n")
    
    with open(f'{results_dir}/Table_results.txt', "a") as file:
        file.write(f"& {roc_auc:.4f}          ")