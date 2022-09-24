import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from loguru import logger

from sheet1.model import ConvNetwork,labels_map,test_loop,train_loop

# train network to be attacked

# get device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# load data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# initialize model
model = ConvNetwork().to(device)
learning_rate = 1e-3
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=1024)
epochs = 5

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if "__main__" == __name__:
    logger.info(f"Start training ...")
    logger.info(f"Using device {device}")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, test_dataloader, model, loss_fn, optimizer,track_test_error=False)
        test_loop(test_dataloader, model, loss_fn, verbose=True)

    torch.save(model.state_dict(), f"./sheet3/modelweights/model.pth")
