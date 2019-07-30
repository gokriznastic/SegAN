from __future__ import print_function
import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
from net import NetS, NetC
from data_loader import LITS, loader

import warnings
warnings.filterwarnings("ignore")

# Training settings
batchSize = 128 # training batch size
size = 128 # square image size
niter = 100 #number of epochs to train for
lr = 0.0002 #Learning Rate. Default=0.0002
ngpu = 1 #number of GPUs to use, for now it only supports one GPU
beta1 = 0.5 #beta1 for adam
decay = 0.5 #Learning rate decay
cuda = True #using GPU or not
seed = 666 #random seed to use
outpath = './outputs' #folder to output images and model checkpoint
alpha = 0.1 #weight given to dice loss while generator training

try:
    os.makedirs(outpath)
except OSError:
    pass

# custom weights initialization called on NetS and NetC
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dice_loss(input,target):
    assert input.size() == target.size(), "Input sizes must be equal."

    assert input.dim() == 4, "Input must be a 4D Tensor."

    num = input*target
    num = torch.sum(num,dim=3)
    num = torch.sum(num,dim=2)

    den1 = input*input
    den1 = torch.sum(den1,dim=3)
    den1 = torch.sum(den1,dim=2)

    den2 = target*target
    den2 = torch.sum(den2,dim=3)
    den2 = torch.sum(den2,dim=2)

    dice = 2*(num/(den1+den2))

    dice_total = 1 - torch.sum(dice)/dice.size(0) #divide by batchsize

    return dice_total

def mergeChannels(array, size):
    c0 = array[:,0,:,:].reshape(-1, 1, size, size)
    c1 = array[:,1,:,:].reshape(-1, 1, size, size)

    c0[c0>=0.5] = 1
    c0[c0<0.5] = 0

    c1[c1>=0.5] = 2
    c1[c1<0.5] = 0

    array = np.hstack((c0, c1))

    array = np.amax(array, axis=1)

    return array.reshape(-1, 1, size, size)

if cuda and not torch.cuda.is_available():
    raise Exception(' [!] No GPU found, please run without cuda.')

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

cudnn.benchmark = True
print('===> Building model')

NetS = NetS(ngpu = ngpu)
# NetS.apply(weights_init)
print('\n########## SEGMENTOR ##########\n')
print(NetS)
print()

NetC = NetC(ngpu = ngpu)
# NetC.apply(weights_init)
print('\n########## CRITIC ##########\n')
print(NetC)
print()

if cuda:
    NetS = NetS.cuda()
    NetC = NetC.cuda()
    # criterion = criterion.cuda()

# setup optimizer
optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(beta1, 0.999))
# load training data
dataloader = loader(LITS('preprocessed', (size, size), train=True), batchSize)
# load testing data
dataloader_val = loader(LITS('preprocessed', (size, size), train=False), batchSize)

print('===> Starting training\n')
max_iou = 0
NetS.train()
for epoch in range(1, niter+1):
    for i, data in tqdm(enumerate(dataloader, 1)):
        ##################################
        ### train Discriminator/Critic ###
        ##################################
        NetC.zero_grad()

        image, target, gt = Variable(data[0]), Variable(data[1]), Variable(data[2])

        if cuda:
            image = image.float().cuda()
            target = target.float().cuda()

        output = NetS(image)
        output = F.sigmoid(output)
        output = output.detach() ### detach G from the network

        input_mask = image.clone()
        output_masked = image.clone()
        output_masked = input_mask * output
        if cuda:
            output_masked = output_masked.cuda()

        target_masked = image.clone()
        target_masked = input_mask * target
        if cuda:
            target_masked = target_masked.cuda()

        output_D = NetC(output_masked)
        target_D = NetC(target_masked)
        loss_D = 1 - torch.mean(torch.abs(output_D - target_D))
        loss_D.backward()
        optimizerD.step()

        ### clip parameters in D
        for p in NetC.parameters():
            p.data.clamp_(-0.05, 0.05)

        #################################
        ### train Generator/Segmentor ###
        #################################
        NetS.zero_grad()

        output = NetS(image)
        output = F.sigmoid(output)

        loss_dice = dice_loss(output,target)

        output_masked = input_mask * output
        if cuda:
            output_masked = output_masked.cuda()

        target_masked = input_mask * target
        if cuda:
            target_masked = target_masked.cuda()

        output_G = NetC(output_masked)
        target_G = NetC(target_masked)
        loss_G = torch.mean(torch.abs(output_G - target_G))
        loss_G_joint = loss_G + alpha * loss_dice
        loss_G_joint.backward()
        optimizerG.step()

        if(i % 10 == 0):
            print("\nEpoch[{}/{}]\tBatch({}/{}):\tBatch Dice_Loss: {:.4f}\tG_Loss: {:.4f}\tD_Loss: {:.4f} \n".format(
                            epoch, niter, i, len(dataloader), loss_dice.item(), loss_G.item(), loss_D.item()))

    # saving visualizations after each epoch to monitor model's progress
    outputC0 = output[:,0,:,:].view(-1, 1, size, size)
    vutils.save_image(outputC0,
            '{}/epoch-{}-liver-output.png'.format(outpath, epoch),
            normalize=True)
    outputC1 = output[:,1,:,:].view(-1, 1, size, size)
    vutils.save_image(outputC1,
            '{}/epoch-{}-tumor-output.png'.format(outpath, epoch),
            normalize=True)
    targetC0 = target[:,0,:,:].view(-1, 1, size, size)
    vutils.save_image(targetC0,
            '{}/epoch-{}-liver-target.png'.format(outpath, epoch),
            normalize=True)
    targetC1 = target[:,1,:,:].view(-1, 1, size, size)
    vutils.save_image(targetC1,
            '{}/epoch-{}-tumor-target.png'.format(outpath, epoch),
            normalize=True)

    output = torch.from_numpy(mergeChannels(output.detach().cpu().numpy(), size)).cuda()

    vutils.save_image(data[0],
            '{}/image.png'.format(outpath, epoch),
            normalize=True)
    vutils.save_image(data[2],
            '{}/target.png'.format(outpath,epoch),
            normalize=True)
    vutils.save_image(output.data,
            '{}/prediction.png'.format(outpath, epoch),
            normalize=True)


    ##################################
    ## validate Generator/Segmentor ##
    ##################################
    NetS.eval()
    IoUs, dices = [], []
    for i, data in enumerate(dataloader_val, 1):
        img, target, gt = Variable(data[0]), Variable(data[1]), Variable(data[2])

        if cuda:
            img = img.cuda()
            target = target.cuda()
            gt = gt.cuda()

        pred = NetS(img)
        pred = F.sigmoid(pred)

        pred = torch.from_numpy(mergeChannels(pred.detach().cpu().numpy(), size)).cuda()
        # pred = pred.type(torch.LongTensor)
        pred_np = pred.data.cpu().numpy()

        gt = gt.data.cpu().numpy()

        for x in range(img.size()[0]):
            IoU = (np.sum(pred_np[x][gt[x]==1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x]==1]))) \
                + (np.sum(pred_np[x][gt[x]==2]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x]==2])))
            dice = (np.sum(pred_np[x][gt[x]==1])*2 / float(np.sum(pred_np[x]) + np.sum(gt[x]))) \
                    + (np.sum(pred_np[x][gt[x]==2])*2 / float(np.sum(pred_np[x]) + np.sum(gt[x])))
            IoUs.append(IoU)
            dices.append(dice)
    NetS.train()

    print('-------------------------------------------------------------------------------------------------------------------\n')

    IoUs = np.array(IoUs, dtype=np.float64)
    dices = np.array(dices, dtype=np.float64)
    mIoU = np.nanmean(IoUs, axis=0)
    mdice = np.nanmean(dices, axis=0)
    print('mIoU: {:.4f}'.format(mIoU))
    print('Dice: {:.4f}'.format(mdice))

    if mIoU > max_iou:
        max_iou = mIoU
        torch.save(NetS.state_dict(), '%s/checkpoint_epoch_%d.pth' % (outpath, epoch))
    vutils.save_image(data[0],
            '%s/val_image.png' % outpath,
            normalize=True)
    vutils.save_image(data[2],
            '%s/val_target.png' % outpath,
            normalize=True)
    pred = pred.type(torch.FloatTensor)
    vutils.save_image(pred.data,
            '%s/val_prediction.png' % outpath,
            normalize=True)

    if epoch % 25 == 0:
        lr = lr*decay
        if lr <= 0.00000001:
            lr = 0.00000001
        print('Learning Rate: {:.6f}'.format(lr))
        # print('K: {:.4f}'.format(k))
        print('Max mIoU: {:.4f}'.format(max_iou))
        optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(beta1, 0.999))

    print('-------------------------------------------------------------------------------------------------------------------')
    print()