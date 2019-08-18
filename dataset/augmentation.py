from imgaug import augmenters as iaa
from scipy import ndimage
import numpy as np
import skimage
import random


def get_augmenter(args):
    """Build imgaug augmenters"""

    augments = []
    if args.vflip:
        augments.append(iaa.Flipud(0.5))
    if args.hflip:
        augments.append(iaa.Fliplr(0.5))
    if args.zoom:
        augments.append(iaa.Sometimes(0.5, iaa.Affine(scale=(1.0, 1.5))))
    if args.rotate:
        augments.append(iaa.Sometimes(0.5, iaa.Affine(scale=(1.0, 1.5))))

    augmenter = iaa.Sequential(augments)

    return augmenter

def augmentation(x, y):
    """augment input and mask
    """

    # vertical flip
    if random.random() < 0.5:
        x = np.flip(x, axis=-2).copy()
        y = np.flip(y, axis=-2).copy()

    # horizontal flip
    if random.random() < 0.5:
        x = np.flip(x, axis=-1).copy()
        y = np.flip(y, axis=-1).copy()

    # random rotation
    if random.random() < 0.5:
        angle = random.randint(-30, 30)
        x = ndimage.rotate(x, angle, (-2, -1), reshape=False).copy()
        y = ndimage.rotate(y, angle, (-2, -1), reshape=False).copy()

    # random cropping
    '''
    if random.random() < 0.5:
        proportion = random.uniform(0.8,1.0)

        x, y = _random_crop(x, y, proportion)
    '''

    # gaussian noise
    if random.random() < 0.5:

        #mean = 0
        #var = 0.001
        #sigma = var**0.5
        #gauss = np.random.normal(mean, sigma, x.shape)
        #gauss = gauss.reshape(x.shape)
        #x += gauss
        x = skimage.util.random_noise(x, mode="gaussian")

    x = x.astype(np.double)
    y = y.astype(np.double)

    return x,y


def _random_crop(x, y, proportion):

    ori_height = x.shape[-2]
    ori_width = x.shape[-1]
    ori_dim = x.shape[0]

    crop_height = int(ori_height * proportion)
    crop_width = int(ori_width * proportion)

    hori_start = random.randint(0, ori_width - crop_width -1)
    verti_start = random.randint(0, ori_height - crop_height -1)

    hori_end = hori_start + crop_width
    verti_end = verti_start + crop_height

    x = x[:,verti_start:verti_end, hori_start:hori_end]
    y = y[:,verti_start:verti_end, hori_start:hori_end]

    x = skimage.transform.resize(x, (ori_dim, ori_height, ori_width))
    y = skimage.transform.resize(y, (ori_dim, ori_height, ori_width), preserve_range=True)

    return x, y                                                                           

