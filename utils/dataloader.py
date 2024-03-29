import os 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json 
from PIL import Image
import skimage

class ImageDataset(Dataset):
    def __init__(self, json_file, bg= None, rgb= False, transform=None):
        with open(json_file) as f:
            self.json_file = json.load(f)
        self.transform = transform
        self.keys  = list(self.json_file["DiffuserImage"].keys())
        self.rgb = rgb
        self.bg = bg
    def __len__(self):
        return len(self.json_file["DiffuserImage"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.json_file["DiffuserImage"][self.keys[idx]]
        label_path = self.json_file["TruthImage"][self.keys[idx]]

        if self.rgb == True:
            try:
                image = np.load(image_path)
                label = np.load(label_path)
                image = torch.tensor(image, dtype=torch.float64).permute(2,0,1)
                label = torch.tensor(label, dtype=torch.float64).permute(2,0,1)
                if self.bg is not None:
                    image = image - self.bg
                    label = label - self.bg
            except:
                print("Error loading image: ", image_path)
        else:
            image = rgb2gray(np.load(image_path))
            label = rgb2gray(np.load(label_path))
            image = torch.tensor(image, dtype=torch.float64)
            label = torch.tensor(label, dtype=torch.float64)

        # should we normalize the image here?
        image /= torch.norm(image.ravel())
        label /= torch.norm(label.ravel())
        sample = {"image": image, "Target": label, "Id": self.keys[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample
    def getById(self, id):
        image_path = self.json_file["DiffuserImage"][id]
        label_path = self.json_file["TruthImage"][id]
        if self.rgb == True:
            image = np.load(image_path)
            label = np.load(label_path)
            image = torch.FloatTensor(image).permute(2,0,1)
            label = torch.FloatTensor(label).permute(2,0,1)
        else:
            image = rgb2gray(np.load(image_path))
            label = rgb2gray(np.load(label_path))
            image = torch.FloatTensor(image)
            label = torch.FloatTensor(label)

        sample = {"image": image, "Target": label, "Id": id}
        if self.transform:
            sample = self.transform(sample)
        return sample
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def load_psf_image(psf_file, ds= 4, rgb = False):
    if rgb==False:
        my_psf = rgb2gray(np.array(Image.open(psf_file)))
    else:
        my_psf = np.array(Image.open(psf_file))
    # psf_diffuser = np.sum(my_psf,2)
    
    bg = np.mean(my_psf[5:15, 5:15])
    psf_diffuser = my_psf - bg


    h = skimage.transform.resize(psf_diffuser, 
                             (psf_diffuser.shape[0]//ds,psf_diffuser.shape[1]//ds), 
                             mode='constant', anti_aliasing=True)
    h /= np.linalg.norm(h.ravel())
    print(h.shape)
    return h, bg