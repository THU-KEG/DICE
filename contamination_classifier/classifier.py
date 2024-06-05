import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm

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

with open('train_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
input_size = len(dataset[0][0])
output_size = 1
model = MLP(input_size, output_size).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

#for features, labels in tqdm(data_loader):
    #print(type(features),end= " ")
#    print(labels)

num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for features, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        #print(outputs.squeeze())
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
        
        predicted = (outputs.squeeze() > 0.5).float()
        #print(f"zkj predicted: {predicted}")
        total += labels.size(0)
        #print(f"zkj labels: {labels}")
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / len(dataset)
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy*100:.4f}%")

# 保存训练好的模型
torch.save(model.state_dict(), 'trained_model.pth')
