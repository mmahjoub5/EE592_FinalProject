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



def evaluate(model, loss, testLoader):
    running_loss = 0.0
    with torch.no_grad:
        for i, data in enumerate(testLoader):
            input, target = data["image"], data["Target"]
            output = model(input)
            loss = loss(output, target)
            print("test loss: ", loss)
            running_loss += loss.item()
        return running_loss

def train_leAdmm(epochs, json, batch_size, psfFile, U:bool):
    h, bg = load_psf_image(psfFile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("THIS IS THE DEVICE", device)
    # load the data
    dataSet = ImageDataset(json, rgb=False)
    print(len(dataSet))
    train, test = train_test_split(dataSet)
    
    train_dataLoad = DataLoader(dataset=dataSet, batch_size=batch_size, sampler=train)
    test_dataLoad = DataLoader(dataset=dataSet, batch_size=batch_size, sampler=test)
    
    # assert len(train_dataLoad) > 0
    # print("Length of train data: ", len(train_dataLoad))
    # assert len(test_dataLoad) > 0
    # print("Length of test data: ", len(test_dataLoad))
    # create the model

    if U:
        model = leAdmm_U(h =h , iterations=5, batchSize=batch_size)
    else:
        model = LeADMM(h =h , iterations=30, batchSize=batch_size)
    model.double()
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]

    assert len(params) > 0
   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
    running_loss = 0.0
    model.train()
    x = autograd.set_detect_anomaly(True)

    # train the model
    for epoch in tqdm(range(epochs)):
        for i, data in tqdm(enumerate(train_dataLoad)):
            input, target = data["image"], data["Target"]
            input = (input/torch.max(input)).double()
            target = (target/torch.max(target)).double()
            target = target.view(batch_size, 1, 270, 480)
            input, target = input.to(device), target.to(device)
            with torch.autograd.set_detect_anomaly(True):
                with torch.set_grad_enabled(True):
                    output = model(input)
                    print("output: ", output.shape)
                    loss = loss_fn(output, target)
                    print("loss gradF ", loss.grad_fn)
                    print("loss: ", loss)
                    print("back prop")
                    optimizer.zero_grad()   
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    running_loss += loss.item()
                    plt.figure()
                    plt.imshow(output[0,...].detach().cpu().numpy(), cmap='gray')
                    plt.savefig("output.png")
                    plt.figure()
        if epoch % 2 == 0:
            print("Epoch: ", epoch, " Loss: ", running_loss)
            saveCheckPoint(epoch, model.state_dict(), optimizer, running_loss, "checkpoints/leAdmm_epoch" + str(epoch) + ".pt")
            running_loss = 0.0
            
            if len(test_dataLoad) > 0:
                test_loss = evaluate(model, loss_fn, test_dataLoad)
                print("test loss: ", test_loss)

    print("Finished Training")
    saveCheckPoint(epoch, model.state_dict(), optimizer, running_loss, "checkpoints/leAdmm_epoch" + str(epochs) + ".pt")
