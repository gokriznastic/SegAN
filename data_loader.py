import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor, Resize
from tifffile import imread

def readData(root, train=True):
    data = []

    if train:
        dir = os.path.join(root, 'train')
    else:
        dir = os.path.join(root, 'test')

    imagefNames = os.listdir(os.path.join(dir,'sampleimages'))

    for imgfName in imagefNames:
        data.append([os.path.join(os.path.join(dir, 'sampleimages'),imgfName),
                    os.path.join(os.path.join(dir, 'samplemasks'), imgfName.replace('volume', 'segmentation'))])

    return data


MIN_BOUND = -100.0
MAX_BOUND = 400.0

PIXEL_MEAN = 0.1021
PIXEL_STD  = 0.19177

def setBounds(image,MIN_BOUND,MAX_BOUND):
    """
    Clip image to lower bound MIN_BOUND, upper bound MAX_BOUND.
    """
    return np.clip(image,MIN_BOUND,MAX_BOUND)

def normalize(image):
    """
    Perform standardization/normalization, i.e. zero_centering and setting
    the data to unit variance.
    """
    image = setBounds(image,MIN_BOUND,MAX_BOUND)
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    # image = np.clip(image, 0., 1.)
    # image = image - PIXEL_MEAN
    # image = image/PIXEL_STD
    return image

def splitChannels(gt):
    c0 = np.copy(gt)
    c1 = np.copy(gt)

    c0[c0 == 2] = 0 # np.unique(c0) = [0, 1]
    c1[c1 == 1] = 0 # np.unique(c1) = [0, 2]
    c1[c1 == 2] = 1 # np.unique(c1) = [0, 1]

    return c0, c1


class LITS(torch.utils.data.Dataset):

    def __init__(self, root, size=(512, 512), transform=None, train=True):
        self.train = train
        self.root = root
        self.size = size

        if not os.path.exists(self.root):
            raise Exception("[!] The directory {} does not exist.".format(root))

        if (transform==None):
            self.transform = Compose([
                            Resize(self.size),
                            ToTensor(),
                            ])
        else:
            self.transform = transform

        self.paths = readData(root, self.train)

    def __getitem__(self, index):
        self.imagePath, self.maskPath = self.paths[index]
        image = imread(self.imagePath)
        gt = imread(self.maskPath)

        image = normalize(image)
        targetL, targetT = splitChannels(gt)

        try:
            image = self.transform(Image.fromarray(image))
            gt = self.transform(Image.fromarray(gt))

            targetL = self.transform(Image.fromarray(targetL))
            targetT = self.transform(Image.fromarray(targetT))

        except:
            print("[!] Transform is invalid.")

        image = np.asanyarray(image)
        mask = np.vstack((np.asarray(targetL), np.asarray(targetT)))
        gt = np.asarray(gt)

        return image, mask, gt

    def __len__(self):
        return len(self.paths)


def loader(dataset, batch_size, num_workers=8, shuffle=True):

    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)

    return input_loader