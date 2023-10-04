# training exp 1

python tools/test.py \
       work_dirs/pal4vst/convnext-tiny_upernet_unified1_512x512/convnext-tiny_upernet_unified1_512x512.py \
       work_dirs/pal4vst/convnext-tiny_upernet_unified1_512x512/best_mIoU_iter_19500.pth
echo "====== convnext-tiny_upernet_unified1_512x512 -> unified ====="

python tools/test.py \
       work_dirs/pal4vst/convnext-large_upernet_unified1_512x512/convnext-large_upernet_unified1_512x512.py \
       work_dirs/pal4vst/convnext-large_upernet_unified1_512x512/best_mIoU_iter_6500.pth
echo "====== convnext-large_upernet_unified1_512x512 -> unified ====="



