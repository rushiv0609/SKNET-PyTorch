import torch
from SKNET import SKNet
from Resnet import resnet18, resnet34
import utils
from train_test import train,test
import argparse
from datetime import datetime 

'''
Define parser to get groups as cmd input
'''
parser = argparse.ArgumentParser()
parser.add_argument("-G", default = 1, type=int, help="number of conv groups in model")
parser.add_argument("-batchsize", default = 256, type=int, help="batch size")
parser.add_argument("-lr_schedule", 
                    default = "plateu", 
                    help="lr scheduler => write `cyclic` if want to use CLR ")
parser.add_argument("-epochs", default = 30, type=int, help="epochs")
parser.add_argument("-model", default = 2, type=int, help="1 -> SKNET, 2-> ResNet18, 3-> ResNet34")
parser.add_argument("-skconv", action="store_true")
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
if int(args.model) == 2:
    print("ResNet18, skconv = ", args.skconv)
    net = resnet18(200, args.skconv)
elif int(args.model) == 3:
    print("ResNet34, skconv = ", args.skconv)
    net = resnet34(200, args.skconv)
else:
    net = SKNet(200, [2,2,2,2], [1,2,2,2], G = args.G)
net.to(device)
print("# of Parameters : ",sum([p.numel() for p in net.parameters()]))
print("Model loaded sucessfully")

'''
Start Training
'''

net = train(net, 
            device, 
            train_loader, 
            val_loader, 
            lr = 9.32e-05, 
            lr_scheduler = args.lr_schedule, 
            epochs = int(args.epochs))
# test(net, device, test_loader)