import os
import torch
import pickle
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from loguru import logger
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from torch.nn import Conv2d,MaxPool2d

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




def train_loop(train_dataloader,test_dataloader, model, loss_fn, optimizer):#,tracker:Tracker):
    size = len(train_dataloader.dataset)
    train_loss_records = []
    test_loss_records = []
    for batch, (X, y) in enumerate(train_dataloader):
        # Load into device
        X,y = X.to(device),y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_records.append(loss.item())
        test_loss_records.append(test_loop(test_dataloader,model,loss_fn))
        
        if batch*len(X) % 1280 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_loss_records,test_loss_records

        
def test_loop(dataloader, model, loss_fn,verbose=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    loss_records = []

    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device),y.to(device) 
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    if verbose:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss



# get devive
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# initialize model
# model = NeuralNetwork().to(device)
model = ConvNetwork().to(device)
learning_rate = 1e-3
batch_size = 64
epochs = 1

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TASK 1
# optimizers = [torch.optim.Adam,torch.optim.SGD]
# loss_records = dict()
# for optimizer in optimizers:
#     opt_name = optimizer.__name__
#     loss_records[opt_name] = {"train":[],"test":[]}
#     logger.info(f"Start training for {opt_name}")

#     model = ConvNetwork().to(device)
#     optimizer = optimizer(model.parameters(), lr=learning_rate)

#     for epoch in range(epochs):
#         print(f"Epoch {epoch+1}\n-------------------------------")
#         train_loss_record = train_loop(train_dataloader, model, loss_fn, optimizer)
#         loss_records[opt_name]["train"].extend(train_loss_record)
#         test_loss_record = test_loop(test_dataloader, model, loss_fn)
#         loss_records[opt_name]["test"].extend(test_loss_record)
#     print("Done!")

# with open('loss_records_task1.pickle', 'wb') as file:
#     pickle.dump(loss_records,file, protocol=pickle.HIGHEST_PROTOCOL)

# TASK 2
batch_sizes = [128]
loss_records = dict()
for batch_size in batch_sizes:
    model = ConvNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    logger.info(f"Start training for batch size {batch_size}")
    loss_records[batch_size] = {"train":[],"test":[]}

    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=512)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loss_record,test_loss_record= train_loop(train_dataloader,test_dataloader, model, loss_fn, optimizer)
        loss_records[batch_size]["train"].extend(train_loss_record)
        loss_records[batch_size]["test"].extend(test_loss_record)

        test_loop(test_dataloader, model, loss_fn,verbose=True)

with open('loss_records_task2.pickle', 'wb') as file:
    pickle.dump(loss_records,file, protocol=pickle.HIGHEST_PROTOCOL)