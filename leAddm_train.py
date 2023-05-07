import torch
from utils.dataloader import ImageDataset, load_psf_image, rgb2gray
from torch.utils.data import Dataset, DataLoader
from model.UNet import UNet270480, UNet_small
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
from model.admm import ADMM_Net
from torch.utils.tensorboard import SummaryWriter
from model.ADMM_U import ADMM_U

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


    if U:
        cnn = UNet_small(in_shape=(1, 2*270, 2*480))
        model = leAdmm_U(h =h , CNN=cnn ,iterations=5, batchSize=batch_size)
       # model = ADMM_U(h=h)
    else:
        model = LeADMM(h =h , iterations=5, batchSize=batch_size)


    model.double()
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]

    assert len(params) > 0
    
    

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
    running_loss = 0.0
    model.train()
    x = autograd.set_detect_anomaly(True)
    model2 = ADMM_Net(h=h, batchSize=1, ADMM=True, iterations=5)
    writer = SummaryWriter()
    # train the model
    for epoch in tqdm(range(epochs)):
        for i, data in tqdm(enumerate(train_dataLoad)):
            input, target = data["image"], data["Target"]
            id = data["Id"]
            input = (input/torch.max(input)).double()
            target = (target/torch.max(target)).double()
            if len(input.shape) > 3:
                target = target.view(batch_size, 1, 270, 480)
            input, target = input.to(device), target.to(device)
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad() 
                output = model(input)


                print("output: ", output.shape)
                assert output.shape == target.shape
                loss = loss_fn(output, target)
                print("loss gradF ", loss.grad_fn)
                print("loss: {}".format(id), loss)
                print("back prop")
                if loss.grad_fn is not None:
                    loss.backward(retain_graph=True)
                optimizer.step()
                running_loss += loss.item()
                writer.add_scalar('Loss/train/{}'.format(id), loss.item(), epoch)
                
                plt.figure()
                plt.title("LEADMM output")
                plt.imshow(output[0,...].detach().cpu().numpy(), cmap='gray')

                if os.path.exists("images/{}/".format(id)): 
                    plt.savefig("images_1/{}/{}.png".format(id,epoch))
                else:
                    os.mkdir("images/{}/".format(id))
                    plt.savefig("images_1/{}/{}.png".format(id,epoch))
                
            
                # plt.figure()
                # plt.title("Target")
                # plt.imshow(target[0,...].detach().cpu().numpy(), cmap='gray')
                # plt.savefig("target.png")
                # plt.close('all')
                    

        if epoch % 2 == 0:
            print("Epoch: ", epoch, " Loss: ", running_loss)
            saveCheckPoint(epoch, model.state_dict(), optimizer, running_loss, "checkpoints/leAdmm_epoch" + str(epoch) + ".pt")
            running_loss = 0.0
            
            if len(test_dataLoad) > 0:
                test_loss = evaluate(model, loss_fn, test_dataLoad)
                print("test loss: ", test_loss)

    print("Finished Training")
    saveCheckPoint(epoch, model.state_dict(), optimizer, running_loss, "checkpoints/leAdmm_epoch" + str(epochs) + ".pt")
