import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import get_data_dir
import torch.optim as optim



import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        
        # Define the layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        #self.fc1 = nn.Linear(64 * 64, 128)  # Adjust the input size depending on your data
        self.fc1 = nn.Linear(64, 128)  # Adjust the input size depending on your data

        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Define the forward pass
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Step 1: Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values  # Assuming the DataFrame has your input and output data
        self.labels = self.data[:, -1]  # Assuming labels are in the last column
        self.features = self.data[:, :-1]  # Assuming features are all columns except the last one

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'feature': torch.tensor(self.features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return sample

df = pd.read_csv(get_data_dir() / 'train.csv')
df = df.fillna(0)


# Step 4: Create an instance of the Dataset Class
dataset = CustomDataset(df)

# Step 5: Create a DataLoader
batch_size = 2
shuffle = True
learning_rate = 0.001

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
model = CNNClassifier(10)

# Now, you can iterate through the DataLoader to get batches of data

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


model.train()
running_loss = 0
for batch in dataloader:
    features = batch['feature']
    labels = batch['label']
    features = features.unsqueeze(1)  # Add a channel dimension

    print(features.shape)

    res = model(features)
    optimizer.zero_grad()
    outputs = model(features)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()





