import torch
from utils.dataloader import ImageDataset, load_psf_image, rgb2gray
from torch.utils.data import Dataset, DataLoader
from model.UNet import UNet270480
from model.leadmm import LeADMM
from model.leAdmm_U import leAdmm_U
from utils.dataloader import ImageDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np 
import argparse
from tqdm import tqdm
import os 
from train_utils import *
from torch import autograd
import matplotlib.pyplot as plt


def train_leAdmm(epochs, json, batch_size, psfFile, U:bool):
    h, bg = load_psf_image(psfFile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("THIS IS THE DEVICE", device)
    # load the data
    dataSet = ImageDataset(json, rgb=False)
    dataLoad = DataLoader(dataset=dataSet, batch_size=batch_size)
    train, test = train_test_split(dataSet)
    if U:
        model = leAdmm_U(h =h , iterations=5, batchSize=batch_size)
    else:
        model = LeADMM(h =h , iterations=5, batchSize=batch_size)
    model.double()
    model.to(device)
    loss_fn = torch.nn.L1Loss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-10, weight_decay=0.0005)
    running_loss = 0.0
    model.train()
    x = autograd.set_detect_anomaly(True)

    # train the model
    for epoch in tqdm(range(epochs)):
        for i, data in tqdm(enumerate(dataLoad)):
            input, target = data["image"], data["Target"]
            input = (input/torch.max(input)).double()
            target = (target/torch.max(target)).double()
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            print("loss: ", loss)
            print(loss)
            print("back prop")
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()
            
        if epoch % 2 == 0:
            print("Epoch: ", epoch, " Loss: ", running_loss)
            saveCheckPoint(epoch, model.state_dict(), optimizer, running_loss, "checkpoints/leAdmm_epoch" + str(epoch) + ".pt")
            running_loss = 0.0
            

    print("Finished Training")
    saveCheckPoint(epoch, model.state_dict(), optimizer, running_loss, "checkpoints/leAdmm_epoch" + str(epochs) + ".pt")
