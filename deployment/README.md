# Model Conversion and Deployment

## Packages

```bash
pip install mmdeploy==1.3.0
```

## Torchscript 

```bash
python mmdeploy/tools/deploy.py \
       mmdeploy/configs/mmseg/segmentation_torchscript.py \
       mmsegmentation/work_dirs/pal4vst/convnext-tiny_upernet_unified1_512x512/convnext-tiny_upernet_unified1_512x512.py \
       mmsegmentation/work_dirs/pal4vst/convnext-tiny_upernet_unified1_512x512/best_mIoU_iter_19500.pth \
       data/pal4vst/unified/images/val/ffhq_seed0524.jpg \
       --work-dir deployment/pal4vst/convnext-tiny_upernet_unified1_512x512 \
       --device cuda \
       --dump-info
```

## ONNX 

```bash

```

## CoreML 

```bash

```
