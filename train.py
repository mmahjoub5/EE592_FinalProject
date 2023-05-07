import torch
from torch.utils.data import Dataset, DataLoader
from model.UNet import UNet270480
from model.leadmm import LeADMM
from utils.dataloader import ImageDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np 
import argparse
from tqdm import tqdm
import os 
from unet_train import trainUnet
from leAddm_train import train_leAdmm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=0, required=True)
    parser.add_argument("--json", type=str, default="", required=True)
    parser.add_argument("--batch_size", type=int, default=-1, required=True)
    parser.add_argument("--model", type=int, default=-1, required=True, help="0 for Unet, 1 for LeADMM, 2 for LeADMM with U")
    parser.add_argument("--psf", type=str, default="", required=True)
    args = parser.parse_args()
    print(args)
    if args.model == 0:
        trainUnet(args.epochs, args.json, batch_size=args.batch_size)
    elif args.model == 1:
        print("Training LeADMM")
        train_leAdmm(args.epochs, args.json, args.batch_size, psfFile=args.psf, U=False)
    elif args.model == 2:
        train_leAdmm(args.epochs, args.json, args.batch_size, psfFile=args.psf, U=True)
    # save the model    
if __name__ == "__main__":
    main()