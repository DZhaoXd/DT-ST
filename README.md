# DT-ST
# Towards Better Stability and Adaptability: Improve Online Self-Training for Model Adaptation in Semantic Segmentation(CVPR-2023)

This is a [pytorch](http://pytorch.org/) implementation of [DT-ST](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Towards_Better_Stability_and_Adaptability_Improve_Online_Self-Training_for_Model_CVPR_2023_paper.html).

### Prerequisites
- Python 3.6
- Pytorch 1.2.0
- torchvision from master
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.0

### Step-by-step installation

```bash
conda create --name fada -y python=3.6
conda activate fada

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

pip install ninja yacs cython matplotlib tqdm opencv-python imageio mmcv

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```

### Getting started
Data:
- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

The data folder should be structured as follows:
```
├── datasets/
│   ├── cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/		
...
```
Pretrain models:
- Download pretrained model on GTA5: ([GTA5](https://drive.google.com/file/d/1dwWoEKTnPBHr9-_-uVyqPH_HfBcLcfns/view?usp=drive_link))
- Download pretrained model on SYNTHIA: ([SYNTHIA](https://drive.google.com/file/d/1UUTfaKBmXmp4mMAAzysOCZf3Of7mN088/view?usp=drive_link)) 

Then, put these *.pth into the pretrain folder.

### Train
G2C model adaptation
```
python train_TCR_DTU.py -cfg configs/deeplabv2_r101_dtst.yaml OUTPUT_DIR results/dtst/ resume pretrain/G2C_model_iter020000.pth
```
S2C model adaptation

```
python train_TCR_DTU.py -cfg configs/deeplabv2_r101_dtst_synthia.yaml OUTPUT_DIR results/synthia_dtst/ resume ./pretrain/S2C_model_iter020000.pth
```


### Evaluate
```
python test.py -cfg configs/deeplabv2_r101_dtst.yaml resume results/dtst_g2c/model_iter020000.pth
```
Our trained model and the training logs are available via [DTST](https://drive.google.com/drive/folders/1MINE4So10eXdXA2mDJtxMagspExDtvB8?usp=drive_link).


### Acknowledge
Some codes are adapted from [FADA](https://github.com/JDAI-CV/FADA#classes-matter-a-fine-grained-adversarial-approach-to-cross-domain-semantic-segmentation-eccv-2020),  [SAC](https://github.com/visinf/da-sac) and [DSU](https://github.com/lixiaotong97/DSU). We thank them for their excellent projects.


### Citation
If you find this code useful please consider citing
```
@inproceedings{zhao2023towards,
  title={Towards Better Stability and Adaptability: Improve Online Self-Training for Model Adaptation in Semantic Segmentation},
  author={Zhao, Dong and Wang, Shuang and Zang, Qi and Quan, Dou and Ye, Xiutiao and Jiao, Licheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11733--11743},
  year={2023}
}
```


