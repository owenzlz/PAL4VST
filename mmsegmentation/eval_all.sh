# python eval_all.py \
#        --config_file work_dirs/metacaf/efficientformer_l3_segformer_data2_1024x1024/efficientformer_l3_segformer_data2_1024x1024.py \
#        --checkpoint_file work_dirs/metacaf/efficientformer_l3_segformer_data2_1024x1024/best_mIoU_iter_20500.pth \
#        --eval_modes cmgan supercaf pd2k  unified1 unified2

python eval_all.py \
       --config_file work_dirs/metacaf/efficientformer_l3_segformer_data1_1024x1024/efficientformer_l3_segformer_data1_1024x1024.py \
       --checkpoint_file work_dirs/metacaf/efficientformer_l3_segformer_data1_1024x1024/best_mIoU_iter_18000.pth \
       --eval_modes cmgan supercaf pd2k unified1 unified2





