import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from torch.nn import Conv2d,MaxPool2d
from tracking_and_visualization import TrackingRecord,Tracker

# define network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork,self).__init__()

        self.conv1 = nn.Sequential(
            # in (3,28,28)
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1),
            # in (32,28,28)
            nn.Dropout2d(0.25),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            # out (32,14,14)
        )

        self.conv2 = nn.Sequential(
            # in (32,14,14)
            nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3,padding=0),
            # in (64,12,12)
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
            # out (64,6,6)
            

        )

        self.ffn = nn.Sequential(
            nn.Linear(in_features=6*6*64,out_features=512),
            nn.Dropout(0.1),
            nn.Linear(in_features=512,out_features=128),
            nn.Linear(in_features=128,out_features=10)
        )

    def forward(self,input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = output.view(output.size(0), -1)
        output = self.ffn(output)
        
        return output

# define network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()



        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()



def train_loop(dataloader, model, loss_fn, optimizer,tracker:Tracker):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Load into device
        X,y = X.to(device),y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            record = tracker.spawn_record()
            record.batch = batch
            record.loss = loss




def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device),y.to(device) 
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    loss_fn = nn.CrossEntropyLoss()



# get devive
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# initialize model
# model = NeuralNetwork().to(device)
model = ConvNetwork().to(device)


learning_rate = 1e-3
batch_size = 64
epochs = 5

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



epochs = 1
tracker = Tracker(optimizer_name="adam",phase="training")
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    tracker.epoch = epoch
    train_loop(train_dataloader, model, loss_fn, optimizer,tracker=tracker)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")