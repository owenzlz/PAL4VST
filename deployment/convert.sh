# python mmdeploy/tools/deploy.py \
#     mmdeploy/configs/mmseg/segmentation_torchscript.py \
#     mmsegmentation/work_dirs/pal4vst/convnext-tiny_upernet_unified1_512x512/convnext-tiny_upernet_unified1_512x512.py \
#     mmsegmentation/work_dirs/pal4vst/convnext-tiny_upernet_unified1_512x512/best_mIoU_iter_19500.pth \
#     data/pal4vst/unified/images/val/ffhq_seed0524.jpg \
#     --work-dir deployment/pal4vst/convnext-tiny_upernet_unified1_512x512 \
#     --device cuda \
#     --dump-info

python ../mmdeploy/tools/deploy.py \
    ../mmdeploy/configs/mmseg/segmentation_torchscript.py \
    ../mmsegmentation/work_dirs/pal4vst/swin-large_upernet_unified_512x512/swin-large_upernet_unified_512x512.py \
    /sensei-fs/users/lingzzha/projects/PAL4VST2/mmsegmentation/work_dirs/unified/best_mIoU_iter_7200.pth \
    ../data/pal4vst/unified/images/val/ffhq_seed0524.jpg \
    --work-dir ./pal4vst/swin-large_upernet_unified_512x512 \
    --device cuda \
    --dump-info


