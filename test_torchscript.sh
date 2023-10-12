# python test_torchscript.py \
#        --img_file ./demo_test_data/stylegan2_ffhq/images/seed0417.jpg \
#        --torchscript_file ./deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt \
#        --out_pal_file ./demo_results/seed0417_pal.png \
#        --out_vis_file ./demo_results/seed0417_pal_vis.jpg

python test_torchscript.py \
       --img_file ./demo_test_data/mask2image/images/000000483531.jpg \
       --torchscript_file ./deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt \
       --out_pal_file ./demo_results/000000483531_pal.png \
       --out_vis_file ./demo_results/000000483531_pal_vis.jpg