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
    parser.add_argument("--loadIndex", type=bool, default=False, required=False)
    parser.add_argument("--colab", type=bool, default=False, required=False)
    args = parser.parse_args()
    model = UNet270480(in_shape=(3, 270, 480))
    #model.eval()
    print(args.loadIndex)
    #load checkpoint
    checkpoint = torch.load(args.checkfile)
    model.load_state_dict(checkpoint["model_state_dict"])
    if not args.loadIndex:
        print("we are here")
        dataSet = ImageDataset(args.json, rgb=True)
        dataLoad = DataLoader(dataset=dataSet, batch_size=1)
    else:
        train_idx, test_idx = loadTrainTestSplit()
        dataSet = ImageDataset(args.json, rgb=True)
        dataLoad = DataLoader(dataset=dataSet, batch_size=1, sampler=test_idx)

    print(len(dataLoad))
    for i, image in enumerate(dataLoad):
        input, target = image["image"], image["Target"]
        output = model(input)
        input = input/torch.max(input)
        output = output/torch.max(output)
        plotImage(input, "input", args.colab)
        plotImage(output, "output", args.colab)
        return



if __name__ == "__main__":
    main()