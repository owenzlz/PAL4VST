# # convert unified swin-l + upernet checkpoint (trained with mmseg 0.x) to torchscript
# python ../mmdeploy/tools/deploy.py \
#     ../mmdeploy/configs/mmseg/segmentation_torchscript.py \
#     ../mmsegmentation/work_dirs/pal4vst/swin-large_upernet_unified_512x512/swin-large_upernet_unified_512x512.py \
#     /sensei-fs/users/lingzzha/projects/PAL4VST2/mmsegmentation/work_dirs/unified/best_mIoU_iter_7200.pth \
#     ../data/pal4vst/unified/images/val/ffhq_seed0524.jpg \
#     --work-dir ./pal4vst/swin-large_upernet_unified_512x512 \
#     --device cuda \
#     --dump-info

# # convert shadow swin-l + upernet checkpoint (trained with mmseg 0.x) to torchscript
# python ../mmdeploy/tools/deploy.py \
#     ../mmdeploy/configs/mmseg/segmentation_torchscript.py \
#     ../mmsegmentation/work_dirs/pal4vst/swin-large_upernet_portraitshadowremoval_512x512/swin-large_upernet_portraitshadowremoval_512x512.py \
#     /sensei-fs/users/lingzzha/projects/PAL4VST2/mmsegmentation/work_dirs/portrait_shadow_removal/best_mIoU_iter_4800.pth \
#     /sensei-fs/users/lingzzha/projects/PAL4VST/demo_test_data/portrait_shadow_removal/images/9166-066.jpg \
#     --work-dir ./pal4vst/swin-large_upernet_portraitshadowremoval_512x512 \
#     --device cuda \
#     --dump-info

# convert stylegan2 swin-l + upernet checkpoint (trained with mmseg 0.x) to torchscript
python ../mmdeploy/tools/deploy.py \
    ../mmdeploy/configs/mmseg/segmentation_torchscript.py \
    ../mmsegmentation/work_dirs/pal4vst/swin-large_upernet_stylegan2_512x512/swin-large_upernet_stylegan2_512x512.py \
    /sensei-fs/users/lingzzha/projects/PAL4VST2/mmsegmentation/work_dirs/stylegan2/stylegan2_merge/best_mIoU_iter_10200.pth \
    /sensei-fs/users/lingzzha/projects/PAL4VST/demo_test_data/stylegan2_ffhq/images/seed0128.jpg \
    --work-dir ./pal4vst/swin-large_upernet_stylegan2_512x512 \
    --device cuda \
    --dump-info


