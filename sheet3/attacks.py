import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from sheet1.model import ConvNetwork,transform,labels_map
from loguru import logger
import matplotlib.pyplot as plt


test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

class FGM:
    def __init__(self,classifier,attack_step_size=0.5,batch_size=None,transform=None) -> None:
        self.classifier = classifier
        self.transform = transform

    
    def generate(self,input_image,target_label=None,iterative=False,attack_step_size=1):
        # prepare loss
        loss = nn.CrossEntropyLoss()

        # prepare input image
        if self.transform:
            input_tensor = self.transform(input_image)
        else:
            input_tensor = input_image

        # record gradients
        input_tensor.requires_grad = True

        # cast labels
        original_label = model(torch.unsqueeze(true_image,0))[0].argmax()
        target_label = torch.tensor([target_label],dtype=torch.long) if target_label else None

        
        # not targeted
        if not target_label:
            if iterative:
                logger.info("start iterative, untargeted attack")

                # max 100 iterations
                for _ in range(100):
                    attacked_input_tensor = self.untargeted_attack(loss,input_tensor,attack_step_size)
                    attacked_input_label = model(torch.unsqueeze(attacked_input_tensor,0))[0].argmax()
                    
                    if attacked_input_label.item() != original_label.item():
                        return attacked_input_tensor
                    else:
                        input_tensor = attacked_input_tensor.detach().clone().requires_grad_()
            # not iterative        
            else:
                logger.info("start untargeted attack")
                attacked_input_tensor = self.untargeted_attack(loss,input_tensor,attack_step_size)

        # targeted
        if target_label:
            if iterative:
                logger.info("start iterative, targeted attack")
                for i in range(10000):
                    attacked_input_tensor = self.targeted_attack(loss,input_tensor,target_label,attack_step_size)
                    attacked_input_label = model(torch.unsqueeze(attacked_input_tensor,0))[0].argmax()
                    
                    if target_label.item() == attacked_input_label.item():
                        return attacked_input_tensor
                    else:
                        input_tensor = attacked_input_tensor.detach().clone().requires_grad_()
            else:
                logger.info("start targeted attack")
                attacked_input_tensor = self.targeted_attack(loss,input_tensor,target_label,attack_step_size)


        return attacked_input_tensor

    
    def untargeted_attack(self,loss,input_tensor,attack_step_size):
        # Calculate loss.
        predicted_tensor = self.classifier(torch.unsqueeze(input_tensor,0))
        predicted_label = torch.unsqueeze(self.classifier(torch.unsqueeze(input_tensor,0))[0].argmax(),0)


        loss_value = loss(predicted_tensor, predicted_label)

        # Zero gradients.
        self.classifier.zero_grad()
        input_tensor.grad = None
        input_tensor.retain_grad()

        # Propagate error backwards through the network.
        loss_value.backward()
        
        # apply step in opposite direction
        c = torch.norm(input_tensor.grad.data)
        input_tensor.data += attack_step_size * input_tensor.grad.data/c

        return input_tensor


    def targeted_attack(self,loss,input_tensor,target_label,attack_step_size):
        # Calculate loss.
        predicted_tensor = self.classifier(torch.unsqueeze(input_tensor,0))
        
        loss_value = loss(predicted_tensor, target_label)

        # Zero gradients.
        self.classifier.zero_grad()
        input_tensor.grad = None
        input_tensor.retain_grad()

        # Propagate error backwards through the network.
        loss_value.backward()
        
        # apply step in opposite direction
        c = torch.norm(input_tensor.grad.data)
        input_tensor.data -= attack_step_size * input_tensor.grad.data/c

        return input_tensor
        
if __name__ == "__main__":
    # load model to be attacked
    model = ConvNetwork() 
    model.load_state_dict(torch.load("./sheet3/modelweights/model.pth"))

    # load model to be attacked
    true_image,true_label = test_data[0]
    pred_label = model(torch.unsqueeze(true_image,0))[0].argmax().item()
    
    if true_label == pred_label:
        logger.info(f"Selected image has the label {true_label}, the model predicts {pred_label}\nThe model predicts the correct label!")
    else:
        logger.warning(f"Selected image has the label {true_label}, the model predicts {pred_label}.\nThe model already fails to predict the correct label!")

    # intialize attacker
    fgm = FGM(model)


    attacked_input_tensor = fgm.generate(true_image.detach().clone(),target_label=2,iterative=True,attack_step_size=0.2)
    
    
    attacked_label = model(torch.unsqueeze(attacked_input_tensor,0))[0].argmax().item()
    logger.info(f"The model predicts the label {attacked_label} for the attacked input")
    
    diff_image = (attacked_input_tensor - true_image).detach().cpu().numpy()[0]
    attacked_image = attacked_input_tensor.detach().cpu().numpy()[0]
    original_image = true_image.detach().cpu().numpy()[0]

    plt.figure(figsize=(8,8))
    plt.subplot(1, 3,1)
    plt.imshow(original_image)
    plt.title(f"original image: {labels_map[pred_label]}")
    plt.subplot(1,3,2)
    plt.imshow(attacked_image)
    plt.title(f"attacked image: {labels_map[attacked_label]}")
    plt.subplot(1,3,3)
    plt.imshow(diff_image)
    plt.title(f"attack image filter")

    plt.show()