import torch
from SKNET import SKNet
import utils
from train_test import train,test


'''
Downloading & importing dataset
'''
utils.download_data()
train_loader, val_loader, test_loader = utils.get_dataloaders(64)

'''
Define SKNet Model
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

num_classes = 200
net = net = SKNet(200, [2,2,2,2], [1,2,2,2])
net.to(device)


'''
Start Training
'''

net = train(net, device, train_loader, val_loader)
test(net, device, test_loader)