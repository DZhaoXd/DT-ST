import os
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import pickle
import os.path as osp
import cv2
#from core.datasets.transform import Compose
from .transform import Compose
from skimage.measure import label as sklabel
from copy import deepcopy
        
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

class rand_mixer_v2():
    def __init__(self, cfg, root='./datasets/cityscapes/'):
        self.image_root = root
        self.cdd_root = os.path.join(cfg.OUTPUT_DIR)
        self.class_num = cfg.MODEL.NUM_CLASSES
        candidate_name = 'CTR'
        self.candidate_name = candidate_name
        self.file_list = os.path.join(cfg.OUTPUT_DIR, candidate_name + '.p')
        self.label_to_file, _ = pickle.load(open(self.file_list, "rb"))
        self.join_mode = 'direct'
        
    def mix(self, in_imgs, classes, choice_p):
    # in_imgs :  PIL or numpy   (w, h, 3)
    # classes :  candidate category for copy-paste  
    # choice_p :  sample prob for candidate category
    # return : 
    #       out_imgs:    (w, h, 3)
    #       out_labels:  (w, h, 3)
        in_imgs = np.array(in_imgs)
        out_imgs = deepcopy(in_imgs)
        out_lbls = np.ones(in_imgs.shape[:2]) * 255 
        in_w, in_h = in_imgs.shape[:2]
        class_idx = np.random.choice(classes, size=1, p=choice_p)[0]
        ins_pix_num = 50
        while True:
            name = random.sample(self.label_to_file[class_idx], 1)
            cdd_img_path = os.path.join(self.image_root, "leftImg8bit/train/%s" % name[0])
            cdd_label_path = os.path.join(self.cdd_root, "{}/{}".format(self.candidate_name, name[0].split('/')[-1]))
            cdd_img = np.asarray(Image.open(cdd_img_path).convert('RGB'), np.uint8)
            cdd_lbl = np.asarray(Image.open(cdd_label_path))
            mask = cdd_lbl==class_idx           
            if np.sum(mask)>=ins_pix_num:
                break
        if self.join_mode == 'direct':
            if self.class_num==19:
                # 12: rider  17: bike   18: moto-
                if class_idx == 12:
                    if 17 in cdd_lbl:
                        mask += cdd_lbl==17
                    if 18 in cdd_lbl:
                        mask += cdd_lbl==18
                if class_idx == 17 or class_idx == 18:
                    mask += cdd_lbl==12
                # 5: pole  6: light   7: sign
                if class_idx == 5:
                    if 6 in cdd_lbl:
                        mask += cdd_lbl==6
                    if 7 in cdd_lbl:
                        mask += cdd_lbl==7
                if class_idx == 6 or class_idx == 7:
                    mask += cdd_lbl==5            
            else:
                # 11: rider  14: bike 15: moto-
                if class_idx == 11:
                    if 14 in cdd_lbl:
                        mask += cdd_lbl==14
                    if 15 in cdd_lbl:
                        mask += cdd_lbl==15
                if class_idx == 14 or class_idx == 15:
                    mask += cdd_lbl==11
                # 5: pole  6: light   7: sign
                if class_idx == 5:
                    if 6 in cdd_lbl:
                        mask += cdd_lbl==6
                    if 7 in cdd_lbl:
                        mask += cdd_lbl==7
                if class_idx == 6 or class_idx == 7:
                    mask += cdd_lbl==5           
        
        masknp = mask.astype(int) 
        seg, forenum = sklabel(masknp, background=0, return_num=True, connectivity=2)
        filled_mask = np.zeros(in_imgs.shape[:2])
        filled_boxes = []
        for i in range(forenum):
            instance_id = i+1
            if np.sum(seg==instance_id) < 20:
                continue
            ins_mask = (seg==instance_id).astype(np.uint8)
            cont, hierarchy = cv2.findContours(ins_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
            cont.sort(key=lambda c: cv2.contourArea(c), reverse=True)                    
            x, y, w, h = cv2.boundingRect(cont[0])   
            #### rescale instance 
            #randscale = 0.5 + np.random.rand() * 1.5
            randscale = 1.0 + np.random.rand() * 1.5
            resized_crop_ins = cv2.resize(cdd_img[y:y+h, x:x+w], None, fx=randscale, fy=randscale, interpolation = cv2.INTER_NEAREST)
            resized_crop_mask = cv2.resize(ins_mask[y:y+h, x:x+w], None, fx=randscale, fy=randscale, interpolation = cv2.INTER_NEAREST)
            resized_crop_mask_cdd = cv2.resize(cdd_lbl[y:y+h, x:x+w], None, fx=randscale, fy=randscale, interpolation = cv2.INTER_NEAREST)
            new_w, new_h = resized_crop_ins.shape[:2]
            if in_w <= new_w or in_h <= new_h:
                continue
            ##### cal new axis
            cnt = 100
            while cnt>1:
                ##### 判断共现类，生成paste位置，穿插到单实例中
                #if class_idx not in resized_crop_mask_cdd and len(filled_boxes) > 0 and random.random() < 0.5:
                if class_idx not in resized_crop_mask_cdd[resized_crop_mask>0] and len(filled_boxes) > 0:
                    rand_box = random.sample(filled_boxes, 1)[0]
                    x1 = random.randint(rand_box[0], rand_box[1] - 1)
                    y1 = random.randint(rand_box[2], rand_box[3] - 1)
                    if x1+new_w<in_w-1 and y1+new_h<in_h-1:
                        break
                x1 = random.randint(0, in_w - new_w - 1)
                y1 = random.randint(0, in_h - new_h - 1)
                if filled_mask[x1, y1] == 0 and filled_mask[x1+new_w, y1+new_h] == 0:
                    break
                cnt-=1
            ##### paste
            if cnt>1:
                out_imgs[x1:x1+new_w, y1:y1+new_h][resized_crop_mask>0] = resized_crop_ins[resized_crop_mask>0]
                out_lbls[x1:x1+new_w, y1:y1+new_h][resized_crop_mask>0] = resized_crop_mask_cdd[resized_crop_mask>0]
                filled_mask[x1:x1+new_w, y1:y1+new_h] = 1 
                ##### 将单实例的存入filled_boxes
                if len(np.unique(resized_crop_mask_cdd[resized_crop_mask>0])) == 1 and class_idx in resized_crop_mask_cdd[resized_crop_mask>0]:
                    filled_boxes.append([x1, x1+new_w, y1,y1+new_h])
                    
        return out_imgs, out_lbls


class cityscapesDataSetTrain(data.Dataset):
    def __init__(
        self,
        data_root,
        data_list,
        max_iters=None,
        num_classes=19, 
        split="train",
        transform=None,
        cfg=None,
        ignore_label=255,
        debug=False,
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        with open(data_list, "r") as handle:
            content = handle.readlines()
        self.img_ids = [i_id.strip() for i_id in content]

        rsc_freq_file = 'CTR_O.p'
        #if max_iters is not None and False:
        if max_iters is not None:
             self.label_to_file, self.file_to_label = pickle.load(open(osp.join(cfg.OUTPUT_DIR, rsc_freq_file), "rb"))
             self.img_ids = []
             SUB_EPOCH_SIZE = 500
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

                     if len(self.label_to_file[c]) == 0 or len(self.label_to_file[c]) == 1:
                         continue
                             
                     if ind[c] > (len(self.label_to_file[c]) - 1):
                         np.random.shuffle(self.label_to_file[c])
                         ind[c] = ind[c] % (len(self.label_to_file[c]) - 1)
        
                     if len(self.label_to_file[c]) == 0:
                         continue
                     c_file = self.label_to_file[c][ind[c]]
                     tmp_list.append(c_file)
                     ind[c] = ind[c] + 1
                     cur_class_dist[self.file_to_label[c_file]] += 1
             print("------------------------city balance sample-----------------------------")
             self.img_ids = tmp_list
        
        self.mixer = rand_mixer_v2(cfg)
        if self.NUM_CLASS == 19:
            self.mix_classes = [4, 5, 6, 7, 12, 16, 17, 18]
            self.mix_p = np.ones(len(self.mix_classes)) / len(self.mix_classes)
        else:
            self.mix_classes = [3, 5, 6, 7, 11, 14, 15]
            self.mix_p = np.ones(len(self.mix_classes)) / len(self.mix_classes)
            
        for name in self.img_ids:
            self.data_list.append(
                {
                    "img": os.path.join(
                        self.data_root, "leftImg8bit/%s/%s" % (self.split, name)
                    ),
                    "label": os.path.join(
                        self.data_root,
                        "gtFine/%s/%s"
                        % (
                            self.split,
                            name.split("_leftImg8bit")[0]
                            + "_gtFine_labelIds.png",
                        ),
                    ),
                    "name": name,
                }
            )

        """ 
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
        """

        self.id_to_trainid = {
            7: 0,
            8: 1,
            11: 2,
            12: 3,
            13: 4,
            17: 5,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18,
        }
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
            18: "bicycle",
        }
        if self.NUM_CLASS==16:# SYNTHIA 
            self.id_to_trainid = {
                7: 0,
                8: 1,
                11: 2,
                12: 3,
                13: 4,
                17: 5,
                19: 6,
                20: 7,
                21: 8,
                23: 9,
                24: 10,
                25: 11,
                26: 12,
                28: 13,
                32: 14,
                33: 15,
            }
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
                9: "sky",
                10: "person",
                11: "rider",
                12: "car",
                13: "bus",
                14: "motocycle",
                15: "bicycle",
            }
        self.transform = transform

        self.ignore_label = ignore_label

        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]
        
        if self.NUM_CLASS == 19:
            self.img_size = (1536, 768)
        if self.NUM_CLASS == 16:
            self.img_size = (1280, 640)
            
        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.array(Image.open(datafiles["label"]),  dtype=np.float32)
        name = datafiles["name"]

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy)
        ## full image
        image = image.resize(self.img_size, Image.BILINEAR)
        img_full = deepcopy(image)
        # full label
        label = label.resize(self.img_size, Image.NEAREST)     

        if self.transform is not None:
            trans_image, _, trans_param = self.transform(image, label)
            
        # mixup 
        trans_image = denormalizeimage(trans_image.unsqueeze(0))
        trans_image = np.asarray(trans_image[0].numpy(), dtype=np.uint8).transpose(1, 2, 0)
        trans_image, mix_label = self.mixer.mix(trans_image, self.mix_classes, self.mix_p)
        
        # norm
        img_full = np.array(img_full, dtype=np.float64) / 255
        img_full -= [0.485, 0.456, 0.406]
        img_full /= [0.229, 0.224, 0.225]
        img_full = torch.from_numpy(img_full.transpose(2, 0, 1)).float()

        # norm
        trans_image = np.array(trans_image, dtype=np.float64) / 255
        trans_image -= [0.485, 0.456, 0.406]
        trans_image /= [0.229, 0.224, 0.225]
        trans_image = torch.from_numpy(trans_image.transpose(2, 0, 1)).float()
        
        mix_label = torch.from_numpy(mix_label)
        
        return trans_image, \
            [torch.from_numpy(np.array(image).transpose(2, 0, 1)), torch.from_numpy(np.array(label))], \
            name, \
            trans_param, \
            img_full, \
            mix_label

