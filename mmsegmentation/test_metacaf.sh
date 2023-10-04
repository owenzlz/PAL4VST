# training exp 1
python test.py \
       --config_file work_dirs/metacaf/efficientformer_l3_segformer_data0a_512x512/efficientformer_l3_segformer_data0a_512x512.py \
       --checkpoint_file work_dirs/metacaf/efficientformer_l3_segformer_data0a_512x512/best_mIoU_iter_10000.pth \
       --img_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/images \
       --msk_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/masks \
       --seg_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/pal/efficientformer_l3_segformer_data0a_512x512

# training exp 2
python test.py \
       --config_file work_dirs/metacaf/efficientformer_l3_segformer_data0b_1024x1024/efficientformer_l3_segformer_data0b_1024x1024.py \
       --checkpoint_file work_dirs/metacaf/efficientformer_l3_segformer_data0b_1024x1024/best_mIoU_iter_12000.pth \
       --img_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/images \
       --msk_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/masks \
       --seg_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/pal/efficientformer_l3_segformer_data0b_1024x1024

# training exp 3
python test.py \
       --config_file work_dirs/metacaf/efficientformer_l3_segformer_data1_1024x1024/efficientformer_l3_segformer_data1_1024x1024.py \
       --checkpoint_file work_dirs/metacaf/efficientformer_l3_segformer_data1_1024x1024/best_mIoU_iter_18000.pth \
       --img_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/images \
       --msk_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/masks \
       --seg_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/pal/efficientformer_l3_segformer_data1_1024x1024

# training exp 4
python test.py \
       --config_file work_dirs/metacaf/efficientformer_l3_segformer_data2_1024x1024/efficientformer_l3_segformer_data2_1024x1024.py \
       --checkpoint_file work_dirs/metacaf/efficientformer_l3_segformer_data2_1024x1024/best_mIoU_iter_20500.pth \
       --img_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/images \
       --msk_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/masks \
       --seg_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/pal/efficientformer_l3_segformer_data2_1024x1024

# training exp 5
python test.py \
       --config_file work_dirs/metacaf/efficientformer_l3_segformer_data2_512x512/efficientformer_l3_segformer_data2_512x512.py \
       --checkpoint_file work_dirs/metacaf/efficientformer_l3_segformer_data2_512x512/best_mIoU_iter_14500.pth \
       --img_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/images \
       --msk_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/masks \
       --seg_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/pal/efficientformer_l3_segformer_data2_512x512

# training exp 6
python test.py \
       --config_file work_dirs/metacaf/efficientformer_l3_segformer_data2_768x768/efficientformer_l3_segformer_data2_768x768.py \
       --checkpoint_file work_dirs/metacaf/efficientformer_l3_segformer_data2_768x768/best_mIoU_iter_14000.pth \
       --img_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/images \
       --msk_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/masks \
       --seg_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/pal/efficientformer_l3_segformer_data2_768x768

# training exp 7
python test.py \
       --config_file work_dirs/metacaf/efficientformer_l1_segformer_data2_1024x1024/efficientformer_l1_segformer_data2_1024x1024.py \
       --checkpoint_file work_dirs/metacaf/efficientformer_l1_segformer_data2_1024x1024/best_mIoU_iter_71500.pth \
       --img_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/images \
       --msk_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/masks \
       --seg_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/pal/efficientformer_l1_segformer_data2_1024x1024

# training exp 10
python test.py \
       --config_file work_dirs/metacaf/convnext-tiny_upernet_data2_1024x1024/convnext-tiny_upernet_data2_1024x1024.py \
       --checkpoint_file work_dirs/metacaf/convnext-tiny_upernet_data2_1024x1024/best_mIoU_iter_24000.pth \
       --img_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/images \
       --msk_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/masks \
       --seg_dir /sensei-fs/users/lingzzha/projects/AdobePAL/data/metacaf/adobepal_datasets/evaluation/curated/pal/convnext-tiny_upernet_data2_1024x1024





