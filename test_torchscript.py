import torch.nn.functional as F
from skimage.io import imsave
from PIL import Image
import numpy as np 
from utils import *
import argparse
import torch
import cv2
import pdb



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--img_file', type=str, default='./demo_test_data/stylegan2_ffhq/images/seed0417.jpg')
    parser.add_argument('--torchscript_file', type=str, default='./deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt')
    parser.add_argument('--out_pal_file', type=str, default='pal.png')
    parser.add_argument('--out_vis_file', type=str, default='img_with_pal.jpg')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--half_precision', action="store_true", help="whether to use half precision")
    args = parser.parse_args()
    
    model = torch.load(args.torchscript_file).to(args.device)
    img = np.array(Image.open(args.img_file).resize((512, 512)))
    img_tensor = prepare_input(img, args.device)
    pal = model(img_tensor)
    pal_np = pal.cpu().data.numpy()[0][0]
    pink = np.zeros((img.shape)); pink[:,:,0] = 255; pink[:,:,2] = 255
    img_with_pal = img * (1 - pal_np[:,:,None]) + args.alpha * pink * pal_np[:,:,None] + (1 - args.alpha) * img * pal_np[:,:,None]
    
    if args.half_precision:
        model = model.half()
        img_tensor = img_tensor.half()
    
    imsave(args.out_pal_file, (pal_np * 255.0).astype(np.uint8))
    imsave(args.out_vis_file, img_with_pal.astype(np.uint8))

    
    