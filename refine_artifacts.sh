python refine_artifacts.py \
       --img_file ./demo_test_data/mask2image/images/000000483531.jpg \
       --torchscript_file ./deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt \
       --out_refine_file refine.jpg \
       --num_inference_steps 75 \
       --high_noise_frac 0.5