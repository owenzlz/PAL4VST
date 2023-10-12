from diffusers import StableDiffusionXLInpaintPipeline
from skimage.io import imsave
from PIL import Image
import numpy as np 
from utils import *
import argparse
import torch
import cv2
import pdb
import os 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_file', type=str, default='./demo_test_data/mask2image/images/000000483531.jpg')
    parser.add_argument('--torchscript_file', type=str, default='./deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt')
    parser.add_argument('--out_refine_file', type=str, default='refine.jpg')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_inference_steps', type=int, default=75)
    parser.add_argument('--high_noise_frac', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.3)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out_refine_file), exist_ok = True)
    
    # load SD-XL inpainter (base + refiner)
    base = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    
    # load PAl model
    model = torch.load(args.torchscript_file).to(args.device)

    # compute pal on the generated image
    img = np.array(Image.open(args.img_file).resize((512, 512)))    
    img_tensor = prepare_input(img, args.device)
    pal = model(img_tensor)
    pal_np = pal.cpu().data.numpy()[0][0]    
    pink = np.zeros((img.shape)); pink[:,:,0] = 255; pink[:,:,2] = 255
    img_with_pal = img * (1 - pal_np[:,:,None]) + args.alpha * pink * pal_np[:,:,None] + (1 - args.alpha) * img * pal_np[:,:,None]
    
    # refine artifacts with SD-XL inpainter
    prompt = 'a photo of clean bedroom'
    kernel = np.ones((5,5), np.uint8)
    pal_np = cv2.dilate(pal_np.astype(np.uint8), kernel, iterations = 5) * 255.0
    
    image = base(
        prompt=prompt,
        image=Image.fromarray(img.astype(np.uint8)),
        mask_image=Image.fromarray(pal_np),
        num_inference_steps=args.num_inference_steps,
        denoising_end=args.high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        image=image,
        mask_image=Image.fromarray(pal_np),
        num_inference_steps=args.num_inference_steps,
        denoising_start=args.high_noise_frac,
    ).images[0]
    
    image.save(args.out_refine_file)
    