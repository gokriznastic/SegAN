import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor
from tifffile import imread

def readData(root, train=True):
    data = []

    if train:
        dir = os.path.join(root, 'train')
    else:
        dir = os.path.join(root, 'test')

    imagefNames = os.listdir(os.path.join(dir,'images'))

    for imgfName in imagefNames:
        data.append([os.path.join(os.path.join(dir, 'images'),imgfName),
                    os.path.join(os.path.join(dir, 'masks'), imgfName.replace('volume', 'segmentation'))])

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
    image = np.clip(image,0.,1.)
    image = image - PIXEL_MEAN
    image = image/PIXEL_STD
    return image


class LITS(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, train=True):
        self.train = train
        self.root = root
        self.size = (512, 512)

        if not os.path.exists(self.root):
            raise Exception("[!] The directory {} does not exist.".format(root))

        if (transform==None):
            self.transform = Compose([
                            ToTensor()])
        else:
            self.transform = transform

        self.paths = readData(root, self.train)

    def __getitem__(self, index):
        self.imagePath, self.maskPath = self.paths[index]
        image = imread(self.imagePath)
        gtMask = imread(self.maskPath)

        image = image.reshape(self.size[0], self.size[1], -1)
        gtMask = gtMask.reshape(self.size[0], self.size[1], -1)

        image = normalize(image)

        try:
            image = self.transform(image)
            gtMask = self.transform(gtMask)
        except:
            print("[!] Transform is invalid.")

        return image, gtMask

    def __len__(self):
        return len(self.paths)


def loader(dataset, batch_size, num_workers=8, shuffle=True):

    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)

    return input_loader