import os
from .cityscapes_b import cityscapesDataSetTrain 
from .cityscapes import cityscapesDataSet
from .cityscapes_self_distill import cityscapesSelfDistillDataSet
from .synthia import synthiaDataSet
from .gta5 import GTA5DataSet
from .bdd import BddDataSet

class DatasetCatalog(object):
    DATASET_DIR = "datasets"
    DATASETS = {
        "gta5_train": {
            "data_dir": "gta5/GTAV",
            "data_list": "gta5_train_list.txt"
        },
        "synthia_train": {
            "data_dir": "synthia",
            "data_list": "synthia_train_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_TT_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_II_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_entropy_src": {
            "data_dir": "cityscapes",
            "data_list": "cs_top20_train_list.txt",
            "label_dir": "cityscapes/src_pret36_hard"
        },
        "cityscapes_entropy_tgt": {
            "data_dir": "cityscapes",
            "data_list": "cs_rest80_train_list.txt",
        },
        "cityscapes_self_distill_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt",
            #"label_dir": "cityscapes/src_pret36_soft"
            "label_dir": "cityscapes/syn_ur_soft"
        },
        "cityscapes_val": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_val_list.txt"
        },
        "bdd_val": {
            "data_dir": "BDD/bdd-100k/bdd100k/images/10k/val/",
            "masks_dir": "BDD/bdd-100k/bdd100k/labels/sem_seg/masks/val/"
        },
        "bdd_train": {
            "data_dir": "BDD/bdd-100k/bdd100k/images/10k/train/",
            "masks_dir": "BDD/bdd-100k/bdd100k/labels/sem_seg/masks/train/"
        },
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None, cfg=None):
        if "gta5" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return GTA5DataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "synthia" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return synthiaDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'distill' in name:
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return cityscapesSelfDistillDataSet(args["root"], args["data_list"], args['label_dir'], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            if 'entropy_src' in name:
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return cityscapesSelfDistillDataSet(args["root"], args["data_list"], args['label_dir'], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)                
            if 'entropy_tgt' in name:
                return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            if 'TT' in name: # only for generate psd
                return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, pseudo=True, cfg=cfg)
            if 'II' in name: 
                return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, pseudo=False)
            if 'train' in mode:
                return cityscapesDataSetTrain(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform, cfg=cfg)
            else:
                return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "bdd" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                mask_path=os.path.join(data_dir, attrs["masks_dir"]),
            )
            if 'distill' in name:
                RuntimeError("Dataset not available: {}".format(name))
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                #return cityscapesSelfDistillDataSet(args["root"], args["data_list"], args['label_dir'], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            if 'train' in mode:
                return BddDataSet(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            else:
                return BddDataSet(args["root"], args["mask_path"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
 
        raise RuntimeError("Dataset not available: {}".format(name))



