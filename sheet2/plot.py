import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from model import VAE
from torchvision import datasets, transforms

BATCH_SIZE = 512
model = VAE()
# model.load_state_dict(torch.load("./sheet2/modelweights/vae"))

test_data = datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor())
test_array = test_data.data.numpy()
labels = test_data.targets.numpy()
reducer = umap.UMAP()

nsamples, nx, ny = test_array.shape

test_array_t = test_array.reshape((nsamples, nx * ny))

embedding = reducer.fit_transform(test_array_t)

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

classes = labels_map.values()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=matplotlib.colormaps["tab10"])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the MNIST dataset', fontsize=24)
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.show()
