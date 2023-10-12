from skimage.io import imsave
from tqdm import tqdm
from PIL import Image
import numpy as np 
from utils import *
import argparse
import torch
import glob
import cv2
import pdb
import os



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--torchscript_file', type=str, default='./deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt')
    parser.add_argument('--input_img_dir', type=str, default='./demo_test_data/stylegan2_ffhq/images')
    parser.add_argument('--rank_img_dir', type=str, default='./demo_results/stylegan2_ffhq_rank')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(args.rank_img_dir, exist_ok = True)
    
    # load PAl model
    model = torch.load(args.torchscript_file).to(args.device)

    # compute pal on the generated image
    img_par_dict = {}
    for img_file in tqdm(sorted(glob.glob(args.input_img_dir + '/*'))):
        fname = os.path.basename(img_file)
        img = np.array(Image.open(img_file).resize((512, 512)))    
        img_tensor = prepare_input(img, args.device)
        pal = model(img_tensor)
        pal_np = pal.cpu().data.numpy()[0][0]
        par = pal_np.sum() / (pal_np.shape[0] * pal_np.shape[1])
        img_par_dict[fname] = par
    
    # sort the image by par - perceptual artifacts ratio
    sorted_img_par_dict = dict(sorted(img_par_dict.items(), key=lambda item: item[1], reverse=False))
    
    # visualize and save the ranked images (from best to worst)
    n_digit = len(str(len(sorted_img_par_dict)))
    rank_idx = 1
    for fname, par in sorted_img_par_dict.items():
        src_img_file = os.path.join(args.input_img_dir, fname)
        rank_info = 'rank: ' + str(rank_idx) + ' ; par: ' + str(round(par, 3))
        img = draw_text_on_image(src_img_file, rank_info, 0.05)
        save_img_file = os.path.join(args.rank_img_dir, str(rank_idx).zfill(n_digit) + '_' + fname)
        img.save(save_img_file)
        rank_idx += 1
        
        
        