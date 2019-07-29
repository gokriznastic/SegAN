import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
from sampledata_loader import LITS, loader
import torch

# def splitChannels(gt):
#     # c0 = np.copy(gt)
#     # c1 = np.copy(gt)

#     c0 = gt[:,0,:,:].reshape(-1, 1, size, size)
#     c1 = gt[:,1,:,:].reshape(-1, 1, size, size)

#     # c0[c0 == 2] = 0 # np.unique(mask) = [0,1]
#     # c1[c1 == 1] = 0 # np.unique(c0) = [0, 2]
#     # c1[c1 == 2] = 1 # np.unique = [0, 1]

#     mask = np.hstack((c0, c1))

#     return mask

def mergeChannels(array, size):
    c0 = array[:,0,:,:].reshape(-1, 1, size, size)
    c1 = array[:,1,:,:].reshape(-1, 1, size, size)

    c0[c0>=0.5] = 1
    c0[c0<0.5] = 0

    c1[c1>=0.5] = 2
    c1[c1<0.5] = 0

    array = np.hstack((c0, c1))

    array = np.amax(array, axis=1)

    # c0 = c0.flatten()
    # c1 = c1.flatten()
    # array = array.flatten()

    # for i in range(array.shape[0]):
    #     if (array[i] == 0):
    #         array[i] = c0[i]
    #     else:
    #         array[i] = c1[i]

    return array.reshape(-1, 1, size, size)


def visualizeMasks(gt, imgLr, size):
    imgM1r, imgM2r = imgLr[:,0,:,:], imgLr[:,1,:,:]
    # print(imgM1r.shape)
    # print(imgM2r.shape)
    imgRr = mergeChannels(imgLr, size)
    # print(imgRr.shape)
    f = plt.figure()
    # f.suptitle('GT:{} LC:{} TC:{} CC:{}'.format([np.min(gt), np.max(gt)], [np.min(imgM1r), np.max(imgM1r)],
    #                                                 [np.min(imgM2r), np.max(imgM2r)], [np.min(imgRr), np.max(imgRr)]), fontsize=20)
    f.add_subplot(1, 4, 1)
    plt.imshow(gt.reshape(size, size))
    f.add_subplot(1, 4, 2)
    plt.imshow(imgM1r.reshape(size, size))
    f.add_subplot(1, 4, 3)
    plt.imshow(imgM2r.reshape(size, size))
    f.add_subplot(1, 4, 4)
    plt.imshow(imgRr.reshape(size, size))
    plt.show(block=True)

# masks = os.listdir('preprocessed/train/samplemasks/')
# # for i in tqdm(range(len(masks))):
# for i in range(len(masks)):
#     imgLr = imread('preprocessed/train/samplemasks/'+masks[i]).reshape(-1, 1, 512, 512)
#     # imgLr = Image.fromarray(imgLr.reshape(512,512))
#     # imgLr = np.asarray(imgLr).reshape(-1, 1, 512,512)
#     # print(imgLr.shape)
#     # imgLr[imgLr==1] = 0.5
#     # imgLr[imgLr==2] = 1
#     if (len(np.unique(imgLr))>2):
#         visualizeMasks(imgLr, 512)
size = 128
dataloader = loader(LITS('preprocessed', (size, size), train=True), 64)

for image, target, gt in dataloader:
    # print(target.size())
    target = target.numpy()
    gt = gt.numpy()
    for i in range(target.shape[0]):
        target_np = target[i].reshape(-1, 2, size, size)
        gt_np = gt[i].reshape(-1, 1, size, size)
        # print(target_np.shape)
        visualizeMasks(gt_np, target_np, size)