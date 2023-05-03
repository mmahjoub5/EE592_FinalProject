import torch
from torch.utils.data import Dataset, DataLoader
from model.UNet import UNet270480
from utils.dataloader import ImageDataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np 
import argparse

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

def saveCheckPoint(epoch,model_state_dict, optimizer, loss, path):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)

def getOptimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
    return optimizer, lr_scheduler


def train_test_split(train_dataset, test_size=0.2):
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    np.random.seed(0)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    return train_sampler, valid_sampler

def train(epochs, json,  batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("THIS IS THE DEVICE")
    # load the data
    dataSet = ImageDataset(json, rgb=True)
    train, test = train_test_split(dataSet)

    train_data = DataLoader(dataset=dataSet, batch_size=batch_size, sampler=train)
    test_data = DataLoader(dataset=dataSet, batch_size=batch_size, sampler=test)

    assert len(train_data) > 0
    assert len(test_data) > 0
    assert len(train_data) + len(test_data) == len(dataSet)
    # create the model
    model = UNet270480(in_shape=dataSet[0]["image"].shape)
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer, lr_scheduler = getOptimizer(model)
    running_loss = 0.0
    model.train()
    # train the model
    for epoch in range(epochs):
        for i, batch in enumerate(train_data):
            input, target = batch["image"], batch["Target"]
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        if epoch % 10 == 0:
            print("Epoch: ", epoch, " Loss: ", running_loss)
            saveCheckPoint(epoch, model.state_dict(), optimizer, running_loss, "checkpoints/epoch" + str(epoch) + ".pt")
            running_loss = 0.0
    print("Finished Training")

    # save the model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=0, required=True)
    parser.add_argument("--json", type=str, default="", required=True)
    parser.add_argument("--batch_size", type=int, default=-1, required=True)
    args = parser.parse_args()
    print(args)
    train(args.epochs, args.json, batch_size=args.batch_size)