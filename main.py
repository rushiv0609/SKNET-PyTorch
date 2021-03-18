import torch
from SKNET import SKNet
from Resnet import resnet18
import utils
from train_test import train,test
import argparse
from datetime import datetime 

'''
Define parser to get groups as cmd input
'''
parser = argparse.ArgumentParser()
parser.add_argument("-G", default = 1, help="number of conv groups in model")
parser.add_argument("-batchsize", default = 256, help="batch size")
parser.add_argument("-lr_schedule", 
                    default = "cyclic", 
                    help="lr scheduler => write `cyclic` if want to use CLR ")
parser.add_argument("-epochs", default = 30, help="epochs")
parser.add_argument("-model", default = 2, help="1 -> SKNET, 2-> ResNet, 3-> ResNeXt")
args = parser.parse_args()

'''
Downloading & importing dataset
'''
print('Process Started at %s'%(str(datetime.now())))
utils.download_data()
train_loader, val_loader = utils.get_dataloaders(batch_size = args.batchsize)
print('Data downloaded and loaded sucessfully')

'''
Define SKNet Model
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(device)

num_classes = 200
if args.model == 2:
    print("ResNet")
    net = resnet18(200)
else:
    net = SKNet(200, [2,2,2,2], [1,2,2,2], G = args.G)
net.to(device)
print("# of Parameters : ",sum([p.numel() for p in net.parameters()]))
print("Model loaded sucessfully")

'''
Start Training
'''

net = train(net, device, train_loader, val_loader, lr = 5e-04, lr_scheduler = args.lr_schedule)
# test(net, device, test_loader)