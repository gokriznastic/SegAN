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

from net import NetS, NetC
from sampledata_loader import LITS, loader

# Training settings
batchSize = 2 #training batch size
niter = 10000 #number of epochs to train for
lr = 0.0002 #Learning Rate. Default=0.02
ngpu = 1 #number of GPUs to use, for now it only supports one GPU
beta1 = 0.5 #beta1 for adam
decay = 0.5 #Learning rate decay
cuda = True #using GPU or not
seed = 666 #random seed to use
outpath = './outputs' #folder to output images and model checkpoint

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
    num=input*target
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=2)

    den1=input*input
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=2)

    den2=target*target
    den2=torch.sum(den2,dim=2)
    den2=torch.sum(den2,dim=2)

    dice=2*(num/(den1+den2))

    dice_total=1-1*torch.sum(dice)/dice.size(0) #divide by batchsize

    return dice_total


if cuda and not torch.cuda.is_available():
    raise Exception(' [!] No GPU found, please run without cuda.')

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

cudnn.benchmark = True
print('===> Building model')
NetS = NetS(ngpu = ngpu)
# NetS.apply(weights_init)
print(NetS)
NetC = NetC(ngpu = ngpu)
# NetC.apply(weights_init)
print(NetC)

if cuda:
    NetS = NetS.cuda()
    NetC = NetC.cuda()
    # criterion = criterion.cuda()

# setup optimizer
optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(beta1, 0.999))
# load training data
dataloader = loader(LITS('preprocessed', train=True), batchSize)
# load testing data
dataloader_val = loader(LITS('preprocessed', train=False), batchSize)


max_iou = 0
NetS.train()
for epoch in range(niter):
    for i, data in enumerate(dataloader, 1):
        ##################################
        ### train Discriminator/Critic ###
        ##################################
        NetC.zero_grad()

        image, target = Variable(data[0]), Variable(data[1])
        if cuda:
            image = image.float().cuda()
            target = target.float().cuda()

        output = NetS(image)
        output = F.sigmoid(output)
        output = output.detach()

        input_mask = image.clone()
        output_masked = image.clone()
        output_masked = input_mask * output
        ### detach G from the network
        # for d in range(3):
        #     output_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * output
        if cuda:
            output_masked = output_masked.cuda()

        target_masked = image.clone()
        target_masked = input_mask * target
        # for d in range(3):
        #     target_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * target
        if cuda:
            target_masked = target_masked.cuda()

        output_D = NetC(output_masked)
        target_D = NetC(target_masked)
        loss_D = - torch.mean(torch.abs(output_D - target_D)) # minus sign because discriminator wants to amximize the abs. diff.
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

        # for d in range(3):
        #     output_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * output
        output_masked = input_mask * output
        if cuda:
            output_masked = output_masked.cuda()

        # for d in range(3):
        #     target_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * target
        target_masked = input_mask * target
        if cuda:
            target_masked = target_masked.cuda()

        output_G = NetC(output_masked)
        target_G = NetC(target_masked)
        loss_G = torch.mean(torch.abs(output_G - target_G))
        loss_G_joint = loss_G + loss_dice
        loss_G_joint.backward()
        optimizerG.step()

    print("===> Epoch[{}]({}/{}): Batch Dice: {:.4f}".format(epoch, i, len(dataloader), 1 - loss_dice.data[0]))
    print("===> Epoch[{}]({}/{}): G_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_G.data[0]))
    print("===> Epoch[{}]({}/{}): D_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_D.data[0]))
    print()

    vutils.save_image(data[0],
            '%s/image.png' % outpath,
            normalize=True)
    vutils.save_image(data[1],
            '%s/target.png' % outpath,
            normalize=True)
    vutils.save_image(output.data,
            '%s/prediction.png' % outpath,
            normalize=True)

    if epoch % 10 == 0:
        NetS.eval()
        IoUs, dices = [], []
        for i, data in enumerate(dataloader_val, 1):
            img, gt = Variable(data[0]), Variable(data[1])
            if cuda:
                img = img.cuda()
                gt = gt.cuda()

            pred = NetS(img)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.type(torch.LongTensor)
            pred_np = pred.data.cpu().numpy()
            gt = gt.data.cpu().numpy()

            for x in range(img.size()[0]):
                IoU = np.sum(pred_np[x][gt[x]==1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x]==1]))
                dice = np.sum(pred_np[x][gt[x]==1])*2 / float(np.sum(pred_np[x]) + np.sum(gt[x]))
                IoUs.append(IoU)
                dices.append(dice)

        NetS.train()
        IoUs = np.array(IoUs, dtype=np.float64)
        dices = np.array(dices, dtype=np.float64)
        mIoU = np.mean(IoUs, axis=0)
        mdice = np.mean(dices, axis=0)
        print('mIoU: {:.4f}'.format(mIoU))
        print('Dice: {:.4f}'.format(mdice))
        if mIoU > max_iou:
            max_iou = mIoU
            torch.save(NetS.state_dict(), '%s/NetS_epoch_%d.pth' % (outpath, epoch))
        vutils.save_image(data[0],
                '%s/image_val.png' % outpath,
                normalize=True)
        vutils.save_image(data[1],
                '%s/target_val.png' % outpath,
                normalize=True)
        pred = pred.type(torch.FloatTensor)
        vutils.save_image(pred.data,
                '%s/prediction_val.png' % outpath,
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