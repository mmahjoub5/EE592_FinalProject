import torch
from model.UNet import UNet270480
from utils.dataloader import ImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

model = UNet270480(in_shape=(3, 270, 480))
#model.eval()

#load checkpoint
checkpoint = torch.load("checkpoints/epoch90.pt")
model.load_state_dict(checkpoint["model_state_dict"])

dataSet = ImageDataset("sample.json", rgb=True)
data = DataLoader(dataset=dataSet, batch_size=1, shuffle=True)

def plotImage(inputImage, title):
    plt.figure()
    plt.title(title)
    plt.imshow(inputImage[0,...].permute(1,2,0).detach().numpy())
    plt.show()

# for i, image in enumerate(data):
#     input, target = image["image"], image["Target"]
#     output = model(input)
#     plotImage(input, "input")
#     plotImage(output, "output")

image = np.load("sample_images/diffuser/im107.npy")
image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)
output = model(image)
plotImage(image, "input")
plotImage(output, "output")