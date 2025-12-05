"""
Convert a pretrained model into an autoencoder by removing the final layers
and training it to reconstruct the input images
"""


project_name = 'vlad-lab'
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys
#add git_dir to path
sys.path.append(git_dir)
import argparse, shutil

import torch

import torch.nn as nn
import torchvision
from torchvision import transforms

import numpy as np
import timm

import pdb

from glob import glob as glob
#from model_loader import load_model as load_model
#from torch.utils.tensorboard import SummaryWriter

print('libs loaded')

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('--data', required=False,
                    help='path to folder that contains train and val folders', 
                    default=None)
parser.add_argument('-o', '--output_path', default=f'{git_dir}/weights',
                    help='path for storing ')
parser.add_argument('--arch', default='resnet50',
                    help='which model to train')
parser.add_argument('--epochs', default=30, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--workers', default=20, type=int,
                    help='how many data loading workers to use')
parser.add_argument('--rand_seed', default=1, type=int,
                    help='Seed to use for reproducible results')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

suf=''
global args, best_prec1
args = parser.parse_args()


out_dir = args.output_path
#set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")



#These are all the default learning parameters from the run_CorNet script
start_epoch = 1
lr = .01 #Starting learning rate
step_size = 10 #How often (epochs)the learning rate should decrease by a factor of 10
weight_decay = 1e-4
momentum = .9
n_epochs = args.epochs
n_save = 5 #save model every X epochs

def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    torch.save(state, f'{out_dir}/{filename}_checkpoint_{args.rand_seed}.pth.tar')
    if (epoch) == 1 or (epoch) % n_save  == 0:
        shutil.copyfile(f'{out_dir}/{filename}_checkpoint_{args.rand_seed}.pth.tar', f'{out_dir}/{filename}_{epoch}_{args.rand_seed}.pth.tar')
    if is_best:
        shutil.copyfile(f'{out_dir}/{filename}_checkpoint_{args.rand_seed}.pth.tar', f'{out_dir}/{filename}_best_{args.rand_seed}.pth.tar')


image_type = args.data
image_type=image_type.split('/')[-2]
model_type = f'{args.arch}_{image_type}{suf}'
print(model_type)

best_prec1 = 0


#Image directory

n_classes = len(glob(f'{args.data}/train/*'))


'''
load model
'''
model = timm.create_model(args.arch, pretrained=True)
#model = timm.create_model('vgg11', pretrained=True)

model = nn.Sequential(*list(model.children())[:-1])

# move model to device
model = model.to(device)
model.eval()

#reconstruction layer 
decoder = nn.Sequential(nn.ReLU(),  #initialize decoder with ReLU activation
    nn.ConvTranspose2d(2048, 3, 224)) #create 2D transposed convolutional layer



print('post decode')

#move decoder to device
decoder = decoder.to(device)
decoder.train()


print(model)

#create transform
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

#inverse normalization for reconstruction
inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])

'''
optimizer = torch.optim.SGD(decoder.parameters(),
                                         lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
'''
optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)


#lr updated given some rule
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

# The loss function to use for classification
#criterion = nn.CrossEntropyLoss()

# the loss function to use for reconstruction
criterion = nn.MSELoss()


criterion = criterion.to(device)



# specify loss function



#pdb.set_trace()

# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))



train_dir = args.data + 'train'
val_dir = args.data + 'val'

val_dir = train_dir #CHANGE THIS BACK TO VAL LATER


train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.workers, pin_memory=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers =  args.workers, pin_memory=True)




print('starting training...')
valid_loss_min = np.inf # track change in validation loss
nTrain = 1
nVal = 1
for epoch in range(start_epoch, n_epochs+1):


    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    
    ###################
    # train the model #
    ###################
    model.train()
    
    for data, target in trainloader:
        # move tensors to GPU if CUDA is available


        data, target = data.to(device), target.to(device)
            #print('moved to cuda')
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        
        encoder_out = model(data)

        
        #convert output to 4D tensor for decoder
        encoder_out = encoder_out.view(encoder_out.size(0), encoder_out.size(1), 1, 1)

        #pdb.set_trace()

        decode_out = decoder(encoder_out)
        
        #print(output.shape)
        # calculate the batch loss

        loss = criterion(decode_out, data)
        nTrain = nTrain + 1
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        
        # update training loss
        train_loss += loss.item()*data.size(0)
        #print(train_loss)

        
    
    #track the lr for the epoch
    #scheduler.step()

    print(f'Epoch: {epoch} \tTraining Loss: {train_loss}')
    # save model if validation loss has decreased
    save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': decoder.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, False,epoch,filename=f'{model_type}')
    