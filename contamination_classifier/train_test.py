import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--layer_index', required=True, type=int)

    args = parser.parse_args()

    base_dir = "/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier"
    with open(f'{base_dir}/train_for_28.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
        
    with open(f'{base_dir}/test_for_28.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    '''train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, splited_test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    with open(f'{base_dir}/splited_train_dataset_for_layer_{args.layer_index}.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
            
    with open(f'{base_dir}/splited_test_dataset_for_layer_{args.layer_index}.pkl', 'wb') as f:
            pickle.dump(splited_test_dataset, f)


    test_dataset = splited_test_dataset + test_dataset'''

    batch_size = 32
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_size = len(train_dataset[0][0])
    output_size = 1
    model = MLP(input_size, output_size).to(device)

    '''count = 0
    for inputs, labels in tqdm(test_data_loader, desc="Read Test Dataset"):
        print(f"{labels}", end=" ")
        count = count + 1

    print(f"zkj debug test data num is : {count}")'''
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00000005)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for features, labels in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)
            
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        train_accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy*100:.4f}%")

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

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/results/DICE.png')

    print(f"AUC: {roc_auc:.4f}")


    # 保存训练好的模型
    torch.save(model.state_dict(), f'/data1/tsq/zkj_use/data_contamination/EasyEdit/contamination_classifier/DICE.pth')
