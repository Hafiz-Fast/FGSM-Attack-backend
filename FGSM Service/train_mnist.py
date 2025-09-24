# train_mnist.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

def train(device='cpu', epochs=3, batch_size=64, out_path='mnist_cnn.pth'):
    transform = transforms.Compose([transforms.ToTensor()])  # gives 0-1
    train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    model = SimpleCNN().to(device)
    optimz = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimz.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            optimz.step()
            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        print(f"Epoch {epoch}: loss={total_loss/total:.4f}, acc={correct/total:.4f}")

    torch.save(model.state_dict(), out_path)
    print("Saved model to", out_path)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--out", default="mnist_cnn.pth")
    args = p.parse_args()
    train(device=args.device, epochs=args.epochs, out_path=args.out)
