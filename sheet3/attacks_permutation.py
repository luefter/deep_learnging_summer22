import art
from art.attacks.evasion import UniversalPerturbation
from art.estimators.classification import PyTorchClassifier
from sheet3.model import optimizer,loss_fn
from sheet1.model import ConvNetwork,transform,labels_map

import matplotlib.pyplot as plt
import torch
import os
import numpy as np

path = "data/FashionMNIST/raw"

# load model to be attacked
model = ConvNetwork() 
model.load_state_dict(torch.load("./sheet3/modelweights/model.pth"))

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 255),
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
)

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path,f'{kind}-images-idx3-ubyte')
    
    with open(labels_path,'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    
    with open(images_path,'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(-1,1,28,28)

    return images, labels

# Load the data
x_train, y_train = load_mnist(path, kind='train')
x_test, y_test = load_mnist(path, kind='t10k')
print("X_train shape: " + str(x_train.shape))
print("y_train shape: " + str(y_train.shape))


# Step 4: Train the ART classifier
classifier.fit(x_train, y_train, batch_size=64, nb_epochs=1)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
print(f"Accuracy on benign test examples: {accuracy * 100}")

if __name__ == "__main__":
    
    # initalize deepfool attack
    attack = UniversalPerturbation(classifier=classifier,attacker="deepfool",delta=0.5,eps=100,batch_size=64,verbose=True,max_iter=30)
    
    # apply attack on first 100 images of training dataset
    attacked_img = attack.generate(x_train[:100],y_train[:100])
    
    # plot perturbation noise
    plt.imshow(attack.noise[0][0])

    # save perturbation noise and corresponing fooling rate on first 100 training images
    plt.savefig(f"./sheet3/results/perturbation_foolingrate_{attack.fooling_rate}.png")
