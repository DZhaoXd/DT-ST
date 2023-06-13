import argparse
import os
import math
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

def generate_label_info():
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = {e.strip():[] for e in imglist}

    for labfile in tqdm(labfiles):
        label = np.unique(np.array(Image.open(os.path.join(labdir, labfile)), dtype=np.uint8))
        # print(label)
        
        for lab in label:
            if 255 == lab: continue
            label_to_file[lab].append(os.path.join(labfile.split('_')[0], labfile.replace('gtFine_labelIds', 'leftImg8bit')))
            file_to_label[os.path.join(labfile.split('_')[0], labfile.replace('gtFine_labelIds', 'leftImg8bit'))].append(lab)
    
    return label_to_file, file_to_label

def _foo(i):
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = dict()
    labfile = labfiles[i]
    file_to_label[labfile] = []
    label = np.unique(np.array(Image.open(os.path.join(labdir, labfile)), dtype=np.float32))
    for lab in label:
        label_to_file[int(lab)].append(labfile)
        file_to_label[labfile].append(lab)
    return label_to_file, file_to_label


def main():
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = {e:[] for e in imglist}
    
    if nprocs==1:
        label_to_file, file_to_label = generate_label_info()
    else:
        with Pool(nprocs) as p:
            r = list(tqdm(p.imap(_foo, range(len(labfiles))), total=len(labfiles)))
        for l2f, f2l in r:
            for lab in range(len(l2f)):
                label_to_file[lab].extend(l2f[lab])
            for fname in f2l.keys():
                print(fname)
                if fname in file_to_label:
                    print(fname)
                    #file_to_label[fname.replace('gtFine_labelIds', 'leftImg8bit')].extend(f2l[fname])
                    file_to_label[fname].extend(f2l[fname])

    if NUM_CLASSES == 19:
        with open(os.path.join(savedir, dir_name + '.p'), 'wb') as f:
            pickle.dump((label_to_file, file_to_label), f)
    else:
        with open(os.path.join(savedir, dir_name + '.p'), 'wb') as f:
            pickle.dump((label_to_file, file_to_label), f)        

            
def gen_lb_info(cfg, dir_name = 'CTR', nprocs=1):
    clsses_num=cfg.MODEL.NUM_CLASSES
    savedir=os.path.join(cfg.OUTPUT_DIR)
    data_list = "datasets/cityscapes_train_list.txt"
    with open(data_list, "r") as handle:
        imglist = handle.readlines()
    labdir = os.path.join(savedir, dir_name)
    labfiles = os.listdir(labdir)
    
    if clsses_num == 19:
        id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                                      19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                                      26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
    else:
        id_to_trainid = {
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
                
    label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
    file_to_label = {e:[] for e in imglist}

    def generate_label_info():
        label_to_file = [[] for _ in range(len(id_to_trainid.keys()))]
        file_to_label = {e.strip():[] for e in imglist}

        for labfile in tqdm(labfiles):
            label_ = np.array(Image.open(os.path.join(labdir, labfile)), dtype=np.uint8)
            label = np.unique(label_)
            # print(label)
            
            for lab in label:
                if 255 == lab: continue
                if np.sum(label_==lab) < 50: continue
                label_to_file[lab].append(os.path.join(labfile.split('_')[0], labfile.replace('gtFine_labelIds', 'leftImg8bit')))
                file_to_label[os.path.join(labfile.split('_')[0], labfile.replace('gtFine_labelIds', 'leftImg8bit'))].append(lab)
        
        return label_to_file, file_to_label
    
    if nprocs==1:
        label_to_file, file_to_label = generate_label_info()
    else:
        with Pool(nprocs) as p:
            r = list(tqdm(p.imap(_foo, range(len(labfiles))), total=len(labfiles)))
        for l2f, f2l in r:
            for lab in range(len(l2f)):
                label_to_file[lab].extend(l2f[lab])
            for fname in f2l.keys():
                print(fname)
                if fname in file_to_label:
                    print(fname)
                    #file_to_label[fname.replace('gtFine_labelIds', 'leftImg8bit')].extend(f2l[fname])
                    file_to_label[fname].extend(f2l[fname])

    if clsses_num == 19:
        with open(os.path.join(savedir, dir_name + '.p'), 'wb') as f:
            pickle.dump((label_to_file, file_to_label), f)
    else:
        with open(os.path.join(savedir, dir_name + '.p'), 'wb') as f:
            pickle.dump((label_to_file, file_to_label), f)    
            


#if __name__ == "__main__":
#    main()

