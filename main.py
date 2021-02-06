import torch
from SKNET import SKNet
import utils
from train_test import train,test
import argparse

'''
Define parser to get groups as cmd input
'''
parser = argparse.ArgumentParser()
parser.add_argument("-G", default = 1, help="number of conv groups in model")
args = parser.parse_args()

'''
Downloading & importing dataset
'''
utils.download_data()
train_loader, val_loader, test_loader = utils.get_dataloaders(batch_size = 256)

'''
Define SKNet Model
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

num_classes = 200
net = SKNet(200, [2,2,2,2], [1,2,2,2], G = args.G)
net.to(device)
print("# of Parameters : ",sum([p.numel() for p in net.parameters()]))

'''
Start Training
'''

net = train(net, device, train_loader, val_loader)
test(net, device, test_loader)