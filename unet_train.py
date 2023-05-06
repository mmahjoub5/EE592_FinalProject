import torch
from torch.utils.data import Dataset, DataLoader
from model.UNet import UNet270480
from model.leadmm import LeADMM
from utils.dataloader import ImageDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np 
import argparse
from tqdm import tqdm
import os 
from train_utils import * 

def evaluate(model, loss_fn, testLoader, device):
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testLoader):
            input, target = data["image"], data["Target"]
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = loss_fn(output, target)
            print("test loss: ", loss)
            running_loss += loss.item()
        return running_loss

def trainUnet(epochs, json,  batch_size):
    writer = SummaryWriter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("THIS IS THE DEVICE", device)
    # load the data
    dataSet = ImageDataset(json, rgb=True)
    train, test = train_test_split(dataSet)
   
    train_data = DataLoader(dataset=dataSet, batch_size=batch_size, sampler=train)
    test_data = DataLoader(dataset=dataSet, batch_size=batch_size, sampler=test)

    assert len(train_data) > 0
    # assert len(test_data) > 0
    print("Length of train data: ", len(train_data))
    print("Length of test data: ", len(test_data))

    # create the model
    model = UNet270480(in_shape=dataSet[0]["image"].shape)
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer, lr_scheduler = getOptimizer(model)
    running_loss = 0.0
    test_loss = 0.0
    model.train()

    model.double()

    # train the model
    for epoch in tqdm(range(epochs)):
        for i, batch in enumerate(train_data):
            input, target = batch["image"], batch["Target"]
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input.double())
            loss = loss_fn(output.double(), target.double())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        test_loss = evaluate(model, loss_fn, test_data, device)

        writer.add_scalar("Unet/Loss/train", running_loss, epoch)
        writer.add_scalar("Unet/Loss/test", test_loss, epoch)
        
        if epoch % 10 == 0:
            print("Epoch: ", epoch, " Loss: ", running_loss)
            saveCheckPoint(epoch, model.state_dict(), optimizer, running_loss, "checkpoints/epoch" + str(epoch) + ".pt")
            running_loss = 0.0
        lr_scheduler.step()
    print("Finished Training")
    saveCheckPoint(epoch, model.state_dict(), optimizer, running_loss, "checkpoints/epoch" + str(epochs) + ".pt")

