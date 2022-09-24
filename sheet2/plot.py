import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from model import VAE
from torchvision import datasets, transforms

Z_DIM = 2
model = VAE()

# -- Task 1 -- #
# -- plotting reconstruction loss -- #
with open("./sheet2/loss_records_task1.pickle", "rb") as f:
    loss_records = pickle.load(f)


def plot_loss(records):
    fig, (ax1, ax2) = plt.subplots(1, 2, clear=True)
    ax1.plot(np.arange(len(records["train"])), records["train"], label="train")
    ax2.plot(np.arange(len(records["test"])), records["test"], label="test")
    fig.savefig("./sheet2/results/loss_records.png")
    plt.clf()


# -- Task 2.1 -- #
# -- load Fashion MNIST data -- #
test_data = datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor())
test_array = test_data.data.numpy()
labels = test_data.targets.numpy()

# -- use UMAP to reduce dimensionality of data points -- #
reducer = umap.UMAP()
nsamples, nx, ny = test_array.shape
test_array_t = test_array.reshape((nsamples, nx * ny))

embedding = reducer.fit_transform(test_array_t)

# -- map labels to description -- #
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


# -- plot dimensionality reduced data
def plot_umap(embeddings):
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap=matplotlib.colormaps["tab10"])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the MNIST dataset', fontsize=24)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.savefig("./sheet2/results/umap.png")
    plt.show(block=False)


# -- Task 2.2 -- #


# generate points in latent space as input for decoder
def generate_latent_points(Z_DIM, n_samples, n_classes=10):
    x_input = np.random.randn(Z_DIM * n_samples)
    z_input = x_input.reshape(n_samples, Z_DIM)
    return z_input


# interpolate between two points
def interpolate(point1, point2, steps=10):
    ratios = np.linspace(0, 1, num=steps)
    vectors = []
    for ratio in ratios:
        v = (1 - ratio) * point1 + ratio * point2
        vectors.append(v)
    return np.asarray(vectors)


# create plot of generated images
def plot_generated(examples, n):
    for i in range(n):
        plt.subplot(1, n, 1 + i)
        plt.axis("off")
        plt.imshow(examples[i, :, :])
    plt.savefig("./sheet2/results/interpolated.png")


if __name__ == "__main__":
    # Task 1.2
    plot_loss(loss_records)

    # Task 2.1
    print("Task2: dimensionality reduced data")
    # plot_umap(embedding)

    # Task 2.2
    print("Task2: interpolated images result")
    # load model
    model.load_state_dict(torch.load("./sheet2/modelweights/vae"))
    # generate points
    pts = generate_latent_points(Z_DIM, 2)
    # interpolate points
    interpolated = interpolate(pts[0], pts[1])
    # generate image
    X = model.decode(torch.from_numpy(interpolated,).type(torch.float))
    # plot image
    plot_generated(X.view(-1,28,28).detach().numpy(), len(interpolated))
