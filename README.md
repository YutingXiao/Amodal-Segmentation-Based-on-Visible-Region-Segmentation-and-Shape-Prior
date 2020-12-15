# Amodal Segmentation Based on Visible Region Segmentation and Shape Prior
<img src="./teaser2.jpg" width="800" height="250" >

This is the PyTorch preview version implementation of our AAAI 2021 paper:

"Amodal Segmentation Based on Visible Region Segmentation and Shape Prior".

It is based on the Detectron2.
Detectron2 is Facebook AI Research's next generation software system.
The details of this frame work can be found at https://github.com/facebookresearch/detectron2


## Installation
```
cd Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior
```
We test this implementation on Python 3.6+, Pytorch 1.4, torchvision 0.5, cuda10.1, fvcore 0.1. Download these 
from [PyTorch](https://pytorch.org). You can install other common packages by:
```
pip install -r requirements.txt
```

Setup Detectron2:
```
python -m pip install -e .
```
You can see [INSTALL.md](INSTALL.md) to setup Detectron2. 
This is for the previous version of Detectron2. If you want to use the latest version of Detectron2,
you should go to [Detectron2](https://github.com/facebookresearch/detectron2) and use their setup files.

Install COCO API:
```
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
In our paper, we provide the mAP(Occluded) to evaluate the performance on occluded object (occlusion
rate > 0.15). If you want to compute this metrics, you should use the cocoeval.py in
detectron2/data/amodal_datasets/pycocotools. Besides, this metrics can only be appiled on methods
predicting both amodal and visible mask.

The evaluator outputs tasks: amodal_segm(coarse), amodal2_segm(refined), visible_segm(coarse), 
visible_segm(refined), bbox. The amodal2_segm corresponds to the results in the paper.

## About Detectron2
If you want to know more information about how to use Detectron2 framework, 
see [GETTING_STARTED.md](GETTING_STARTED.md),
or the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).

Learn more at [documentation](https://detectron2.readthedocs.org).

## Download Resource
### D2SA dataset
The information of D2S Amodal could be found at [mvtec-d2sa](https://www.mvtec.com/company/research/datasets/mvtec-d2s/).

Images: ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_d2s/amodal/d2s_amodal_images_v1.tar.xz

Amodal Annotations: ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_d2s/amodal/d2s_amodal_annotations_v1.tar.xz

### KINS dataset
Download the [Images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip)
from [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d). 

The [Amodal Annotations](https://drive.google.com/drive/folders/1FuXz1Rrv5rrGG4n7KcQHVWKvSyr3Tkyo?usp=sharing)
could be found at [KINS dataset](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset)

### COCOA dataset
The COCOA dataset annotation: ftp://guest:GU.205dldo@ftp.softronics.ch/cocoa/COCOA_annotations_detectron.tar.xz.

The images of COCOA dataset is the train2014 and val2014 of [COCO dataset](http://cocodataset.org/).
The [COCO API (pycocotools)](https://github.com/cocodataset/cocoapi) is used for COCO format data.

## Preparation
Set the image path and annotation path in the ./detectron2/data/datasets/builtin.py. Besides, the 
items whose key end with "visible" are used for visible mask evaluation.

Set the output path in the ./detectron2/config/defaults.py:
```
_C.OUTPUT_DIR = 'your_output_path'
```
or in the respective .yaml file

## Train
```
cd Amodal-Segmentation-Based-on-Visible-Region-Segmentation-and-Shape-Prior
```
For example, on the D2SA dataset, we show how to train our method and other baselines.

1.Train our model (ResNet50 backbone) on D2SA dataset with pretrained shape prior auto-encoder 
and codebook:
```
python tools/train_net.py --config-file configs/D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRef_SPRet_FM.yaml
```
2.We have provided the pretrained auto-encoder (d2sa_recon_net.pth) and codebook (d2sa_codebook.npy).
If you want to get your own pretrained shape prior codebook and auto-encoder, you can set
this in the config .yaml file:
```
MODEL.ROI_MASK_HEAD.RECON_NET.LOAD_CODEBOOK: False
```
If you do so. The model will be trained on amodal mask segmentation and amodal mask reconstruction simultaneously.
After the training, the codebook will be generated. 

The input of amodal mask reconstrction has two parts:
ground truth amodal mask and predicted amodal mask, and you can use the parameter 
MODEL.ROI_MASK_HEAD.RECON_NET.MASK_THS to filter the predicted amodal mask based on the mask IOU.
If you only want to use ground truth mask as shape prior, you can set:
```
MODEL.ROI_MASK_HEAD.RECON_NET.MASK_THS: 1.0
```
Hint: 

Since the .yaml files with "SPRef" (Shape Prior Refinement) in the name need to 
use shape prior in the training, you should have the shape prior codebook and auto-encoder 
before training. So, the operations for generating shape prior are not applicable for the .yaml files with "SPRef".
 
We recommend to use our pretrained shape prior codebook and auto-encoder directly.

2.Train the Mask-RCNN (ResNet50 backbone) on D2SA dataset:
```
python tools/train_net.py --config-file configs/D2SA-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_amodal.yaml
```

3.Train the ORCNN (ResNet50 backbone) on D2SA dataset:
```
python tools/train_net.py --config-file configs/D2SA-AmodalSegmentation/mask_orcnn_R_50_FPN_1x.yaml
```
## Test
If you want to eval your saved checkpoints:
```
python tools/train_net.py --config-file configs/{your_yaml_file} 
--eval-only MODEL.WEIGHTS {your_OUTPUT_DIR}/model_final.pth'
```