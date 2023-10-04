python test_torchscript.py \
       --img_file ./demo_test_data/stylegan2_ffhq/images/seed0417.jpg \
       --torchscript_file ./deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt \
       --out_pal_file pal.png \
       --out_vis_file img_with_pal.jpg