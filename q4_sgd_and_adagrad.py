import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, optimizer, criterion, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = torch.argmax(output, dim=1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                predicted = torch.argmax(output, dim=1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct_test / total_test

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, '
              f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

# Model, criterion
model_sgd = SimpleCNN()
model_adagrad = SimpleCNN()
criterion = nn.CrossEntropyLoss()

# SGD optimizer
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)

# Adagrad optimizer
optimizer_adagrad = optim.Adagrad(model_adagrad.parameters(), lr=0.01)

print("Training with SGD Optimizer:")
train_model(model_sgd, optimizer_sgd, criterion)

print("\nTraining with Adagrad Optimizer:")
train_model(model_adagrad, optimizer_adagrad, criterion)
