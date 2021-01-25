import os
import zipfile
import gdown
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def download_data():
    
    dir_name = 'tiny-imagenet-200'
    zip_file = 'tiny-imagenet-200.zip'
    
    if not os.path.isdir(dir_name): # if directory not present then download and unzip
        if not os.path.exists(zip_file):
            url = 'https://drive.google.com/uc?id=1n-jwJulLoPraTe7KImctFjhsvufi_6yq'
            gdown.download(url, output=zip_file, quiet=False)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")

def get_dataloaders(batch_size = 256):
    transform = transforms.Compose(
        [transforms.RandomCrop((56,56)),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),])
    
    train_dir = 'tiny-imagenet-200/train'
    test_dir = 'tiny-imagenet-200/val'
    training = torchvision.datasets.ImageFolder(train_dir, transform = transform)
    test_ds = torchvision.datasets.ImageFolder(test_dir, transform = transform)
    
    tot_len = len(training)
    test_len = len(test_ds)
    val_len = tot_len //10
    train_len = tot_len - val_len
    
    
    train_ds, val_ds= torch.utils.data.random_split(training, [train_len, val_len])
    print("Length of train, valid, test set : ",(len(train_ds), len(val_ds), len(test_ds)))
    train_loader = DataLoader(train_ds, shuffle=True, batch_size= batch_size)
    val_loader = DataLoader(val_ds, shuffle=True, batch_size= batch_size)
    test_loader = DataLoader(test_ds, shuffle=True, batch_size= batch_size)
    return train_loader, val_loader, test_loader

