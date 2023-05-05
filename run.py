from utils.dataloader import ImageDataset, load_psf_image, rgb2gray
from torch.utils.data import DataLoader
from model.admm import ADMM_Net
import torch.nn.functional as F
from utils.addm_helpers import *
import json
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
images_jsons = {
        "DiffuserImage": {  
                        "112": "sample_images/diffuser/im112.npy",
                        "43": "sample_images/diffuser/im43.npy",
                        "65": "sample_images/diffuser/im65.npy",
                        "157": "sample_images/diffuser/im157.npy",},
        "TruthImage": {
                        "112": "sample_images/lensed/im112.npy", 
                        "43": "sample_images/lensed/im43.npy",
                        "65": "sample_images/lensed/im65.npy",
                        "157": "sample_images/lensed/im157.npy",}
}


with open("sample.json", "w") as f:
    json.dump(images_jsons, f)

def pad(h, x):
    pad_size0 = int((h.shape[1])//2)
    pad_size1 = int((h.shape[2])//2)
    padding = (pad_size1, pad_size1, pad_size0, pad_size0)
    h = F.pad(h, padding, mode='constant', value=0)
    x = F.pad(x, padding, mode='constant', value=0)
    return h, x

def testForwardModel():
    x = rgb2gray(np.load(images_jsons["TruthImage"]["112"]))
    print(x.shape)
    h = gaussianPSF(h.shape, .3).T

    h = torch.tensor(h).view(1, h.shape[0], h.shape[1])
    pad_size0 = int((h.shape[1])//2)
    pad_size1 = int((h.shape[2])//2)
    padding = (pad_size1, pad_size1, pad_size0, pad_size0)
    h = F.pad(h, padding, mode='constant', value=0)
    x = torch.tensor(x).view(1, x.shape[0], x.shape[1])
    x = F.pad(x, padding, mode='constant', value=0)

    plt.figure()
    plt.title("psf")

    plt.imshow(h[0,...], cmap='gray')
    plt.figure()
    plt.title("image")
    plt.imshow(x[0,...], cmap='gray')
    plt.show()

    print(h.shape)
    print(x.shape)
    H_fft = torch.fft.fft2(torch.fft.ifftshift(h))

    convImage = H(x, H_fft)
    convImage = convImage[0, ...].numpy()
    plt.figure()
    plt.title("conv image")
    plt.imshow(convImage, cmap='gray')
    plt.show()

def testForwardModel2(h):
    x = rgb2gray(np.load(images_jsons["TruthImage"]["112"]))
    print(x.shape)
    h = torch.tensor(h).view(1, h.shape[0], h.shape[1])
    x = torch.tensor(x).view(1, x.shape[0], x.shape[1])
    h, x = pad(h, x)
    plt.figure()
    plt.title("psf")
    plt.imshow(h[0,...], cmap='gray')
    plt.figure()
    plt.title("image")
    plt.imshow(x[0,...], cmap='gray')
    plt.show()
    print(h.shape)
    print(x.shape)
    H_fft = torch.fft.fft2(torch.fft.ifftshift(h))
    convImage = H(x, H_fft)
    convImage = convImage[0, ...].numpy()
    plt.figure()
    plt.title("conv image")
    plt.imshow(convImage, cmap='gray')
    plt.show()

   



def plotImages(dataSet):
     for i in range(len(dataSet)):
        label_image = dataSet[i]["image"].numpy()
        image = dataSet[i]["Tcleaarget"].numpy()
        plt.figure()
        plt.imshow(image)
        plt.figure()
        plt.imshow(label_image)
        plt.show()

def gaussianPSF(size, sigma):
    """
    size: size of the psf
    sigma: sigma of the gaussian
    """
    x = np.linspace(-10, 10, size[0])
    y = np.linspace(-10, 10, size[1])
    x, y = np.meshgrid(x, y)
    return np.exp(-(x**2 + y**2)/(2*sigma**2)) * (1/(2*np.pi*sigma**2))
        
def main():
    h, bg =  load_psf_image("sample_images/psf.tiff")
    device = "cpu"


    model = ADMM_Net(h=h, batchSize=1, cuda_device=device, ADMM=True)
    dataSet = ImageDataset("sample.json", transform=None)
    image_dataloader = DataLoader(dataset=dataSet, batch_size=1, shuffle=True)

    print(dataSet[0]["image"].shape)
    print(dataSet[0]["Target"].shape)
    # test psf
    for i, batch in enumerate(image_dataloader):
        with torch.no_grad():
            input = model(batch["image"].to(device))
            plt.figure(1)
            plt.imshow(batch["image"][0,...], cmap='gray')        
            f = plt.figure(2)      
            print(input.shape)   
            plt.imshow(input[0,...], cmap='gray')
            plt.title('Reconstruction image number {}'.format(batch["Id"]))
            plt.figure(3)
            plt.imshow(batch["Target"][0, ...], cmap='gray')
            plt.title('Target image number {}'.format(batch["Id"]))
            plt.show()


if __name__ == "__main__":
    main()