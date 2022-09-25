import numpy as np
from torchvision import models, transforms
from PIL import Image
from sheet1.model import ConvNetwork
from datasets import training_data, labels_map
import torch
	
 
	
# Prepare input.
image_tensor, image_label = training_data[1]
# c,h,w -> h,w,c
image = np.transpose((image_tensor*255).detach().cpu().numpy(),(1, 2, 0))
print(image.shape)
# add dimension for batch
image_tensor = torch.unsqueeze(image_tensor,0)
	
# Prepare network.
model = ConvNetwork() 
model.load_state_dict(torch.load("./sheet3/modelweights/model.pth"))
model.eval()
	
# Feed input through network.
prediction = model(image_tensor).argmax().item()
print(prediction)
	
# Decode prediction.
print(labels_map[prediction])
	
	
# Import Captum.
from captum.attr import Saliency
from captum.attr import visualization as vis
	
 
	
# Compute attribution.
saliency = Saliency(model)
attribution = saliency.attribute(image_tensor, target=prediction).cpu().numpy().squeeze(0)


print(attribution.shape)
# Visualize.
figure, _ = vis.visualize_image_attr(np.transpose(attribution, (1, 2, 0)),
	
                                     original_image=image,
	
                                     method="blended_heat_map",
	
                                     use_pyplot=False)
	
figure.savefig('./sheet4/results/saliency.png', bbox_inches="tight", pad_inches=0)
