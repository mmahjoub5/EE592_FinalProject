import torch
from model.UNet import UNet270480
from utils.dataloader import ImageDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
try:    
    from google.colab.patches import cv2_imshow
except:
    import cv2 


def plotImage(inputImage, title, colab):
    plt.figure()
    plt.title(title)
    plt.imshow(inputImage[0,...].permute(1,2,0).detach().numpy())
    if not colab:
        plt.show()
    else:
        cv2_imshow(inputImage[0,...].permute(1,2,0).detach().numpy())

def loadTrainTestSplit(path="trainTestSplit"):
    if (os.path.exists("trainTestSplit/train_idx.npy") and os.path.exists("trainTestSplit/test_idx.npy")):
        train_idx = np.load("trainTestSplit/train_idx.npy")
        test_idx = np.load("trainTestSplit/test_idx.npy")
        return train_idx, test_idx
    else:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkfile", type=str, default="", required=True)
    parser.add_argument("--json", type=str, default="", required=True)
    args = parser.parse_args()
    model = UNet270480(in_shape=(3, 270, 480))
    model.eval()
    #load checkpoint
    checkpoint = torch.load(args.checkfile,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.double()
    print("we are here")
    dataSet = ImageDataset(args.json, rgb=True)
    dataLoad = DataLoader(dataset=dataSet, batch_size=1)

    print(len(dataLoad))
    for i, image in enumerate(dataLoad):
        with torch.no_grad():
            input, target = image["image"], image["Target"]
            id = image["Id"]
            output = model(input.double())
            input = input/torch.max(input)
            output = output/torch.max(output)
            plt.figure()
            plt.title("U_NET output")
            output = output[0,...].permute(1,2,0)
            plt.imshow(output.detach().cpu().numpy(), cmap='gray')
           
            if os.path.exists("images_unet/"): 
                plt.savefig("images_unet/{}.png".format(id))
            else:
                os.mkdir("images_unet/")
                plt.savefig("images_unet/{}.png".format(id))



if __name__ == "__main__":
    main()