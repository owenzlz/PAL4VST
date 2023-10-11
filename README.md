# PAL4VST

[Project Page](https://owenzlz.github.io/PAL4VST/) |  [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Perceptual_Artifacts_Localization_for_Image_Synthesis_Tasks_ICCV_2023_paper.pdf) | [Bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:SCG89JWD2TYJ:scholar.google.com/&output=citation&scisdr=ClFw-pD-EOWtv4IsW4M:AFWwaeYAAAAAZR0qQ4NG4ZJLvyJzv8fZpndl5xA&scisig=AFWwaeYAAAAAZR0qQ33AJMGShkF_7tLplELgMr8&scisf=4&ct=citation&cd=-1&hl=en)

<img src="https://github.com/owenzlz/PAL4VST/blob/project_page/resources/pal4vst.png" style="width:800px;">

[**Perceptual Artifacts Localization for Image Synthesis Tasks**](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Perceptual_Artifacts_Localization_for_Image_Synthesis_Tasks_ICCV_2023_paper.pdf)\
*International Conference on Computer Vision (ICCV), 2023*\
Lingzhi Zhang, Zhengjie Xu, Connelly Barnes, Yuqian Zhou, Qing Liu, He Zhang, Sohrab Amirghodsi, Zhe Lin, Eli Shechtman, Jianbo Shi

This paper presents a comprehensive study of Perceptual Artifacts Localization on multiple synthesis tasks. 

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN

**Table of Contents:**<br>
1. [Setup](#setup) - download pretrained models and resources
2. [Test Images](#test_images) - quick usage with torchscript
2. [Quick Usage](#quick_usage) - quick usage with torchscript
3. [Datasets](#datasets) - download our train/val/test artifacts datasets (coming soon)
4. [Checkpoints](#checkpoints) - download the checkpoints for all our models (coming soon)
5. [Inference](#inference) - inference with models/data (coming soon)
6. [Training](#training) - training scripts (coming soon)

<a name="setup"/>

## Setup
- Clone this repo:
```bash
git clone https://github.com/owenzlz/PAL4VST
```

- Install dependencies:
```bash
conda create --name pal4vst python=3.8 -y
conda activate pal4vst
pip install torch torchvision
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install timm==0.6.5
pip install scikit-image
pip install -U openmim && mim install "mmpretrain>=1.0.0rc8"
pip install mmdeploy==1.3.0
cd mmsegmentation
pip install -v -e .
```
For more information, please feel free to refer to MMSegmentation: https://mmsegmentation.readthedocs.io/en/latest/

<a name="test_images"/>

## Test images

We randomly sampled 10 test images per synthesis tasks, in case you want to quickly try it out. Please check "./demo_test_data".

```bash
- demo_test_data
    |- anyresegan
    |- cvton
    |- edge2image
    |- stylegan2_ffqh
    |- ...
```

<a name="quick_usage"/>

## Quick Usage

Download torchscript checkpoint
([swin-large_upernet_unified_512x512](https://drive.google.com/file/d/1alICAkY8sjr-gwWknAEQjIKDkdPUPyKd/view?usp=sharing)), and place it under ./deployment/pal4vst/swin-large_upernet_unified_512x512/. 

- Quick Inference with torchscript
```bash
python test_torchscript.py \
       --img_file ./demo_test_data/stylegan2_ffhq/images/seed0417.jpg \
       --torchscript_file ./deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt \
       --out_pal_file pal.png \
       --out_vis_file img_with_pal.jpg
```

- Alternatively, a Snippet of Inference Code
```bash
from utils import *
from PIL import Image
import numpy as np 
import torch

device = 0
torchscript_file = './deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt'
img_file = './demo_test_data/stylegan2_ffhq/images/seed0417.jpg'

model = torch.load(torchscript_file).to(device)
img_tensor = prepare_input(np.array(Image.open(img_file).resize((512, 512))), device)
pal = model(img_tensor).cpu().data.numpy()[0][0] # prediction: Perceptual Artifacts Localization (PAL)
```

- Test over All Test Images
```bash
python test_torchscript_all.py
```

**Note**: This checkpoint is trained on the unified data, and produces reasonable predictions to all synthesis tasks, except for shadow removal which requires a specialist model as reported in the paper. You will need a Shadow specialist checkpoint. Specialist checkpoints can be downloaded below. 


## Checkpoints
Coming soon!

| Model Name | Torchscript | Pytorch |
|:--------:|:--------:|:--------:|
| anyresgan | Value2a | Value3a |
| cvton | Value2b | Value3b |
| unified | Value2b | Value3b |


## Datasets
Coming soon!


## Application 1 - Artifacts Refinement
Coming soon!


## Application 2 - Image Curation
Coming soon!



## Training

An examplar training command line (multi-GPU training)
```bash
./tools/dist_train.sh configs/pal4vst/swin-large_upernet_unified_512x512.py 8 --work-dir work_dirs/pal4vst/swin-large_upernet_unified_512x512
```

**Note**: All checkpoints are trained in MMSeg 0x codebase. We haven't retrained/reproduced the models after the codebase update (MMSeg 0x &rarr; MMSeg 1x).

## Citation
If you use this code for your research, please cite our paper:
```
@InProceedings{Zhang_2023_ICCV,
    author    = {Zhang, Lingzhi and Xu, Zhengjie and Barnes, Connelly and Zhou, Yuqian and Liu, Qing and Zhang, He and Amirghodsi, Sohrab and Lin, Zhe and Shechtman, Eli and Shi, Jianbo},
    title     = {Perceptual Artifacts Localization for Image Synthesis Tasks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {7579-7590}
}
```

This work is inspired by our previous work: 
```bash
@inproceedings{zhang2022perceptual,
  title={Perceptual artifacts localization for inpainting},
  author={Zhang, Lingzhi and Zhou, Yuqian and Barnes, Connelly and Amirghodsi, Sohrab and Lin, Zhe and Shechtman, Eli and Shi, Jianbo},
  booktitle={European Conference on Computer Vision},
  pages={146--164},
  year={2022},
  organization={Springer}
}
```

## Acknowledgement

We thank all the authors who their open-sourced research codes, which we used to generate the training data for this paper. 

StyleGAN: 
StyleGAN-Human:
Latent Diffusion Models: 
