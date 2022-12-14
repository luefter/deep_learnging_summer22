	
import numpy as np
from torchvision import models, transforms
from datasets import training_data, labels_map
import torch
from sheet1.model import ConvNetwork,transform
import numpy as np
from sheet3.attacks_fast import FGM


target_label = 5
for attack in [False,True]:
    # Prepare input.
    image_tensor, image_label = training_data[0]

    # Prepare network.
    model = ConvNetwork() 
    model.load_state_dict(torch.load("./sheet3/modelweights/model.pth"))
    model.eval()



    # Decode prediction.
    prediction = model(torch.unsqueeze(image_tensor,0)).argmax().item()
    print(labels_map[prediction])
    print(image_tensor.shape)
    if attack:
        fgm = FGM(model)
        # apply BIM on input_image
        attacked_image_tensor,_= fgm.generate(input_image=image_tensor,
                    target_label=target_label,
                    iterative=True)

        image_tensor = attacked_image_tensor

    # c,h,w -> h,w,c
    image = np.transpose((image_tensor*255).detach().cpu().numpy(),(1, 2, 0))
    # add dimension for batch
    image_tensor = torch.unsqueeze(image_tensor,0)
    # Import Captum.
        
    from captum.attr import (
        GuidedBackprop,
        Occlusion,
        Saliency,
        IntegratedGradients,
    )
        
    from captum.attr import visualization as vis	
    from matplotlib import pyplot as plt
        
    
        
    # Compute and visualize attributions.
        
    methods = [
        (Saliency(model), {}),
        (GuidedBackprop(model), {}),
        (IntegratedGradients(model), {}),
        (Occlusion(model),
            {
                "sliding_window_shapes": (1, 4, 4),
                "strides": (1, 4, 4),
            },
        ),
        
    ]
    figure, axes = plt.subplots(	
        2, 2, figsize=(2 * 3.8, 2 * 3.8)	
    )
            
    for axis, (method, params) in zip(	
        axes.flatten(), methods	
    ):	
        attribution = (
            method.attribute(
                image_tensor, target=prediction, **params
            ).detach().cpu().numpy().squeeze(0)
        )
        
        vis.visualize_image_attr(
            np.transpose(attribution, (1, 2, 0)),
            original_image=image,
            method="heat_map",
            sign="positive",
            show_colorbar=True,
            plt_fig_axis=(figure, axis),
            use_pyplot=False,
        )
        axis.set_title(type(method).__name__)

    plt.tight_layout()
        
    figure.savefig(f"./sheet4/results/methods_attacked?{attack}.png", bbox_inches="tight")
