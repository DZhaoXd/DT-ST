import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from os.path import splitext
import os
import os.path as osp
from os import listdir
from collections import namedtuple
import pickle
from .transform import Compose

class RandomCrop_city(object):  # used for results in the CVPR-19 submission
    def __init__(self, size, padding=0):
        #if isinstance(size, numbers.Number):
            #self.size = (int(size), int(size))
        #else:
            #self.size = size
        self.size = tuple(size)

    def __call__(self, img, mask, param):
    
        assert img.size == mask.size
        w, h = img.size
        tw, th = self.size
        size_scale = random.random() * 0.25  + 1
        n_w, n_h = int(size_scale * w), int(size_scale * h)
        #n_w, n_h = int(size_scale * tw), int(size_scale * th)
        img = img.resize((n_w, n_h), Image.BICUBIC)
        mask = mask.resize((n_w, n_h), Image.NEAREST)
        
        x1 = random.randint(0, n_w - tw)
        y1 = random.randint(0, n_h - th)

        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
            param
        )
        
def denormalizeimage(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *=255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)
    
class rand_mixer():
    def __init__(self, root='./datasets/BDD/', dataset='BDD', class_num=19):

        if dataset == "BDD":
            #file_list = os.path.join(root, 'corda_save.p')
            #file_list = os.path.join(root, 'src_ur_soft.p')
            #file_list = os.path.join(root, 'param_color.p')
            #file_list = os.path.join(root, 'param.p')
            file_list = os.path.join(root, 'bdd_ur_19.p')
        else:
            print('rand_mixer {} unsupported'.format(dataset))
            return
        self.root = root
        #with open(data_list, "r") as handle:
        #    content = handle.readlines()
        #for fname in content:
        #    name = fname.strip()
        self.class_num = class_num
        if class_num==19:
            input_size = (1536, 768)
            input_size = (1280, 640)
        else:
            input_size = (1280, 640)
            #input_size = (1536, 768)
            
        self.label_to_file, _ = pickle.load(open(file_list, "rb"))
        self.data_aug = Compose([RandomCrop_city(input_size)])
        
    def oneMix(self, mask, data = None, target = None):
        #Mix
        if not (data is None):
            data = (mask*data[0]+(1-mask)*data[1]).unsqueeze(0)
        if not (target is None):
            target = (mask*target[0]+(1-mask)*target[1]).unsqueeze(0)
        return data, target
    
    def generate_class_mask(self, pred, classes):
        pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
        N = pred.eq(classes).sum(0)
        return N

    def mix(self, in_imgs, in_lbls, classes, PIXEL_MEAN=[0.485, 0.456, 0.406], PIXEL_STD=[0.229, 0.224, 0.225]):
        out_imgs = torch.ones_like(in_imgs)
        out_lbls = torch.ones_like(in_lbls)
        bs_idx = 0
        for (in_img, in_lbl) in zip(in_imgs, in_lbls):
            class_idx = random.sample(classes, 1)[0]
            while True:
                name = random.sample(self.label_to_file[class_idx], 1)
                img_path = os.path.join(self.root, "bdd-100k/bdd100k/images/10k/train/%s" % name[0].split('.')[0] + '.jpg')
                #label_path = os.path.join(self.root, "corda_save/%s" % name[0].split('/')[-1].replace('leftImg8bit', 'gtFine_labelIds') )
                #label_path = os.path.join(self.root, "src_ur_soft/%s" % name[0].split('/')[-1])
                #label_path = os.path.join(self.root, "param_color/%s" % name[0].split('/')[-1])
                # label_path = os.path.join(self.root, "param/%s" % name[0].split('/')[-1])
                label_path = os.path.join(self.root, "bdd_ur_19/%s" % name[0].split('.')[0] + '.png')
                img = Image.open(img_path)
                lbl = Image.open(label_path)
                img, lbl, _ = self.data_aug(img, lbl) # random crop to input_size
                img = np.asarray(img, np.float32)
                lbl = np.asarray(lbl, np.float32)
                if class_idx in lbl:
                    break
                    
            img = img[:, :, ::-1].copy() / 255 # change to BGR
            img -= PIXEL_MEAN
            img /= PIXEL_STD
            img = img.transpose((2, 0, 1))
            img = torch.Tensor(img)
            lbl = torch.Tensor(lbl)

            class_i = torch.Tensor([class_idx]).type(torch.int64)
            MixMask = self.generate_class_mask(lbl, class_i)
            
            if self.class_num==19:
                if class_idx == 12:
                    if 17 in lbl:
                        MixMask += self.generate_class_mask(lbl, torch.Tensor([17]).type(torch.int64))
                    if 18 in lbl:
                        MixMask += self.generate_class_mask(lbl, torch.Tensor([18]).type(torch.int64))

                if class_idx == 17 or class_idx == 18:
                    MixMask += self.generate_class_mask(lbl, torch.Tensor([12]).type(torch.int64))
            else:
                if class_idx == 11:
                    if 14 in lbl:
                        MixMask += self.generate_class_mask(lbl, torch.Tensor([14]).type(torch.int64))
                    if 15 in lbl:
                        MixMask += self.generate_class_mask(lbl, torch.Tensor([15]).type(torch.int64))

                if class_idx == 14 or class_idx == 15:
                    MixMask += self.generate_class_mask(lbl, torch.Tensor([11]).type(torch.int64))                
            
            #from skimage import io
            #io.imsave('mask.png', (MixMask.float().cpu().numpy()*255).astype(np.uint8))

            mixdata = torch.cat((img.unsqueeze(0), in_img.unsqueeze(0)))
            mixtarget = torch.cat((lbl.unsqueeze(0), in_lbl.unsqueeze(0)))
            data, target = self.oneMix(MixMask, data=mixdata, target=mixtarget)
            out_imgs[bs_idx] = data
            out_lbls[bs_idx] = target
            bs_idx += 1
        return out_imgs, out_lbls
        
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',  # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',  # Whether this label distinguishes between single instances or not

    'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 154)),  # (153,153,153)
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 143)),  # (  0,  0,142)
]

trainId2trainId = {label.trainId: label.trainId for label in labels}


class BddDataSet(data.Dataset):
    def __init__(
            self,
            imgs_dir,
            masks_dir,
            max_iters=None,
            num_classes=19,
            split="train",
            transform=None,
            ignore_label=255,
            debug=False,
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_list = []
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

        #if max_iters is not None and False:
        data_root = 'datasets/BDD/'
        file_name = 'bdd_ur_19.p'
        if max_iters is not None:
             self.label_to_file, self.file_to_label = pickle.load(open(osp.join(data_root, file_name), "rb"))
             self.ids = []
             SUB_EPOCH_SIZE = 1000
             tmp_list = []
             ind = dict()
             for i in range(self.NUM_CLASS):
                 ind[i] = 0
             for e in range(int(max_iters / SUB_EPOCH_SIZE) + 1):
                 cur_class_dist = np.zeros(self.NUM_CLASS)
                 for i in range(SUB_EPOCH_SIZE):
                     if cur_class_dist.sum() == 0:
                         dist1 = cur_class_dist.copy()
                     else:
                         dist1 = cur_class_dist / cur_class_dist.sum()
                     w = 1 / np.log(1 + 1e-2 + dist1)
                     w = w / w.sum()
                     c = np.random.choice(self.NUM_CLASS, p=w)
        
                     if ind[c] > (len(self.label_to_file[c]) - 1):
                         np.random.shuffle(self.label_to_file[c])
                         ind[c] = ind[c] % (len(self.label_to_file[c]) - 1)
        
                     c_file = self.label_to_file[c][ind[c]]
                     tmp_list.append(c_file)
                     ind[c] = ind[c] + 1
                     cur_class_dist[self.file_to_label[c_file.split('.')[0] + '.jpg']] += 1
             print("------------------------city balance sample-----------------------------")
             self.ids = tmp_list

        for fname in self.ids:
            name = fname.strip()
            self.data_list.append(
                {
                    "img": os.path.join(
                        self.imgs_dir, name.split('.')[0] + '.jpg'
                    ),
                    "label": os.path.join(
                        self.masks_dir, name.split('.')[0] + '.png'
                    ),
                    "name": name,
                }
            )

        self.id_to_trainid = trainId2trainId

        self.mixer = rand_mixer(class_num = self.NUM_CLASS)
        if self.NUM_CLASS == 19:
            self.mix_classes = [5, 6, 7, 12, 17, 18]
        else:
            self.mix_classes = [5, 6, 7, 11, 14, 15]
            

        self.transform = transform

        self.ignore_label = ignore_label

        self.debug = debug

        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle"
        }

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]
        
       # self.split = 'val_train'
        if 'val' in self.split:
            image = Image.open(datafiles["img"]).convert('RGB')
            label = np.array(Image.open(datafiles["label"]), dtype=np.float32)
            name = datafiles["name"] + '.png'

            # re-assign labels to match the format of Cityscapes
            label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            # for k in self.trainid2name.keys():
            #     label_copy[label == k] = k
            label = Image.fromarray(label_copy)

            if self.transform is not None:
                image, label, _ = self.transform(image, label)
            return image, label, name
        else:
            if self.NUM_CLASS == 19:
                self.img_size = (1536, 768)
                self.img_size = (1280, 640)
            if self.NUM_CLASS == 16:
                self.img_size = (1280, 640)
                #self.img_size = (1536, 768)
                
            image = Image.open(datafiles["img"]).convert('RGB')

            label = np.array(Image.open(datafiles["label"]),  dtype=np.float32)
            name = datafiles["name"]

            # re-assign labels to match the format of Cityscapes
            label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
            # for k in self.trainid2name.keys():
            #     label_copy[label == k] = k
            label = Image.fromarray(label_copy)

            ## full image
            img = image.resize(self.img_size, Image.BILINEAR)
            # full label
            label = label.resize(self.img_size, Image.NEAREST)     
            
            ## norm full image
            img_full = np.array(img, dtype=np.float64) / 255
            img_full -= [0.485, 0.456, 0.406]
            img_full /= [0.229, 0.224, 0.225]
            img_full_ori = torch.from_numpy(img_full.transpose(2, 0, 1)).float()
            
            ## mix up
            mix_label = torch.ones_like(torch.from_numpy(np.array(label, np.float32))) * 255
            img_full, mix_label = self.mixer.mix(img_full_ori.clone().unsqueeze(0), mix_label.unsqueeze(0), self.mix_classes)
            #img_full = img_full_ori.clone().unsqueeze(0)
            denormalized_image = denormalizeimage(img_full.clone())
            denormalized_image = np.asarray(denormalized_image.cpu().numpy()[0], dtype=np.uint8)
            image = Image.fromarray(denormalized_image.transpose((1,2,0)))
            
            if self.transform is not None:
                trans_image, trans_label, trans_param = self.transform(image, label)
            
            #return trans_image, trans_label, name, trans_param, img_full.squeeze(), mix_label.squeeze()
            return trans_image, [img_full_ori, torch.from_numpy(np.array(label, dtype=np.uint8)).float()], name, trans_param, img_full.squeeze(), mix_label.squeeze()
        
