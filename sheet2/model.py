import os
import numpy as np
import torch
import pickle

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# -- parameter configuration -- #
BATCH_SIZE = 64
EPOCHS = 1
LEARNING_RATE = 1e-3
Z_DIM = 2
Z1_RANGE = 2
Z2_RANGE = 2
Z1_INTERVAL = 0.2
Z2_INTERVAL = 0.2
loss_records = {"train": [], "test": []}

# set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# -- load data set -- #
train_data = datasets.FashionMNIST('../data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor())


# -- Variational Autoencoder -- #
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, Z_DIM)
        self.fc22 = nn.Linear(400, Z_DIM)
        self.fc3 = nn.Linear(Z_DIM, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)


def train(epoch):
    model.train()
    train_loss = 0
    global loss_records
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        loss_records["train"].append(loss.item())
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    global loss_records
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            loss_records["test"].append(loss.item())

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), './sheet2/results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    print(f"Using {device} device")
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, Z_DIM).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28), './sheet2/results/sample_' + str(epoch) + '.png')

    torch.save(model.state_dict(), "./sheet2/modelweights/vae")
    with open("./sheet2/loss_records_task1.pickle", "wb") as file:
        pickle.dump(loss_records, file, protocol=pickle.HIGHEST_PROTOCOL)
