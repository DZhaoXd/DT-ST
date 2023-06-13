import random
import math
import numpy as np
import numbers
import collections
from PIL import Image

import torch
import torchvision
from torchvision.transforms import functional as F
import cv2

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        params = {}
        for t in self.transforms:
            image, label, params = t(image, label, params)
        return image, label, params

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    #img = torch.from_numpy (pic.transpose ((2, 0, 1)))
    #float/int
    #divide 255: img.float().div(255)
    def __call__(self, image, label, params):
        return F.to_tensor(image), F.to_tensor(label).squeeze(), params


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, label, params):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        params['mean'] = self.mean
        params['std'] = self.std
        return image, label, params


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size, resize_label=True):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.resize_label = resize_label

    def __call__(self, image, label, params):
        image = F.resize(image, self.size, Image.BICUBIC)
        if self.resize_label:
            if isinstance(label, np.ndarray): 
                # assert the shape of label is in the order of (h, w, c)
                label = cv2.resize(label,(self.size[1], self.size[0]), cv2.INTER_LINEAR)
            else:
                label = F.resize(label, self.size, Image.NEAREST)
        params['resize'] = self.size
        return image, label, params 

class RandomScale(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, scale, size=None, resize_label=True):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        self.scale = scale
        self.size = size
        self.resize_label = resize_label

    def __call__(self, image, label, params):
        w, h = image.size
        if self.size:
            h, w = self.size
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        size = (int(h*temp_scale), int(w*temp_scale))
        image = F.resize(image, size, Image.BICUBIC)
        if self.resize_label:
            if isinstance(label, np.ndarray):
                # assert the shape of label is in the order of (h, w, c)
                label = cv2.resize(label,(self.size[1], self.size[0]), cv2.INTER_LINEAR)
            else:
                label = F.resize(label, size, Image.NEAREST)
        params['rescale_size'] = size
        return image, label, params

class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, label_fill=255, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        
        if isinstance(size, numbers.Number):
            self.padding = (padding, padding, padding, padding)
        elif isinstance(size, tuple):
            if padding is not None and len(padding)==2:
                self.padding = (padding[0], padding[1], padding[0], padding[1])
            else:
                self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.label_fill = label_fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lab, param):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            if isinstance(lab, np.ndarray):
                lab = np.pad(lab,((self.padding[1], self.padding[3]), (self.padding[0], self.padding[2]), (0,0)), mode='constant')
            else:
                lab = F.pad(lab, self.padding, self.label_fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            if isinstance(lab, np.ndarray):
                lab = np.pad(lab,((0, 0), (self.size[1]-img.size[0], self.size[1]-img.size[0]), (0,0)), mode='constant')
            else:
                lab = F.pad(lab, (self.size[1] - lab.size[0], 0), self.label_fill, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            if isinstance(lab, np.ndarray):
                lab = np.pad(lab,((self.size[0]-img.size[1], self.size[0]-img.size[1]), (0, 0), (0,0)), mode='constant')
            else:
                lab = F.pad(lab, (0, self.size[0] - lab.size[1]), self.label_fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)
        if isinstance(lab, np.ndarray):
            # assert the shape of label is in the order of (h, w, c)
            lab = lab[i:i+h, j:j+w, :]
        else:
            lab = F.crop(lab, i, j, h, w)

        param['random_crop_axis'] = (i, i+h, j, j+w) 
        return img, lab, param

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, param):
        param['RandomHorizontalFlip'] = False
        if random.random() < self.p:
            image = F.hflip(image)
            if isinstance(label, np.ndarray): 
                # assert the shape of label is in the order of (h, w, c)
                label = label[:,::-1,:]
            else:
                label = F.hflip(label)
            param['RandomHorizontalFlip'] = True
        return image, label, param

class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, label, param):
        if random.random() < 0.9:
            image = self.color_jitter(image)
        else:
            image = torchvision.transforms.functional.to_grayscale(image,num_output_channels=3)
        return image, label, param


class GaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
        self.GaussianBlur = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

    def __call__(self, image, label, param):
        if random.random() < self.p:
            image = self.GaussianBlur(image)
        return image, label, param


class CutOut(object):
    def __init__(self, p=0.25, ratio=0.25, cut_size=32):
        self.p = p
        self.cut_ratio = ratio
        self.cut_size = cut_size

    def __call__(self, image, label, param):
        h, w = image.shape[-2:]
        if random.random() < self.p:
            cut_mask = torch.rand(h//self.cut_size, w//self.cut_size)  > self.cut_ratio
            cut_mask = torch.nn.functional.interpolate(cut_mask.unsqueeze(0).unsqueeze(0).float(), size=(h, w)).squeeze(0)
            image = image * cut_mask

        return image, label, param

