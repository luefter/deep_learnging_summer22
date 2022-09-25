import numpy as np
from torchvision import models, transforms
from PIL import Image
from sheet1.model import ConvNetwork
from datasets import training_data, labels_map
import torch
import matplotlib.pyplot as plt
	
"""
Take the model for the FashionMNIST data from the first sheet. Take 2 different
examples from at least wo different classes each
"""
# ankle boot and tshirt 
for training_data_idx in [0,1]:

    # Prepare input.
    image_tensor, image_label = training_data[training_data_idx]
    # c,h,w -> h,w,c
    image = np.transpose((image_tensor*255).detach().cpu().numpy(),(1, 2, 0))
    # add dimension for batch
    image_tensor = torch.unsqueeze(image_tensor,0)
        
    # Prepare network.
    model = ConvNetwork() 
    model.load_state_dict(torch.load("./sheet3/modelweights/model.pth"))
    model.eval()

    """
    provide a feature based explana-
    tion for it being classifier to the most likely and second most likely and least likely
    class
    """
    # Feed input through network.
    values,indices = model(image_tensor).sort()
    # least, second most, most likely
    classes = indices[0][[0,-2,-1]].detach().cpu().tolist()
        
    # Decode prediction.
    print("\n".join([labels_map[l] for l in classes]))
        
        
    # Import Captum.
    from captum.attr import Saliency
    from captum.attr import visualization as vis
        
    
        
    # Compute attribution.
    saliency = Saliency(model)
    fig, axs = plt.subplots(1,3)
    for i, c in enumerate(classes):
        attribution = saliency.attribute(image_tensor, target=c).cpu().numpy().squeeze(0)
        # Visualize
        _, _ = vis.visualize_image_attr(np.transpose(attribution, (1, 2, 0)),
                                            original_image=image,
                                            method="blended_heat_map",
                                            use_pyplot=False,
                                            plt_fig_axis= (fig,axs[i])
                                        )
        axs[i].set_title(labels_map[c])
            
    plt.savefig(f'./sheet4/results/example_{training_data_idx}_saliency.png', bbox_inches="tight", pad_inches=0)
