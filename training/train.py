"""
train.py — Train a compact MLP on MNIST.
Architecture: 784 -> 64 (ReLU) -> 32 (ReLU) -> 10
Output: weights/mlp_mnist.pt
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64,  32)
        self.fc3 = nn.Linear(32,  10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train(epochs=15, lr=1e-3, data_root="data", out_dir="weights"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set  = datasets.MNIST(data_root, train=True,  download=True, transform=transform)
    test_set   = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=1000)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            criterion(model(data), target).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = sum(
            model(d.to(device)).argmax(1).eq(t.to(device)).sum().item()
            for d, t in test_loader)
        acc = 100. * correct / len(test_set)
        print(f"Epoch {epoch:02d}/{epochs}  acc={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{out_dir}/mlp_mnist.pt")

    print(f"Best accuracy: {best_acc:.2f}%  ->  weights/mlp_mnist.pt")


if __name__ == "__main__":
    train()
