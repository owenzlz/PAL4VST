from mmseg.apis import inference_model, init_model, show_result_pyplot
from skimage.io import imsave
from tqdm import tqdm 
from PIL import Image 
import numpy as np
import argparse
import torch
import glob
import mmcv
import pdb
import os

parser = argparse.ArgumentParser(description="inference with mmsegmentation")
parser.add_argument("--alpha", default=0.3, type=float)
parser.add_argument("--config_file", default='', type=str)
parser.add_argument("--checkpoint_file", default='', type=str)
parser.add_argument("--img_dir", default='', type=str)
parser.add_argument("--msk_dir", default='', type=str)
parser.add_argument("--seg_dir", default='', type=str)
args = parser.parse_args()

os.makedirs(args.seg_dir, exist_ok = True)

# build the model from a config file and a checkpoint file
model = init_model(args.config_file, args.checkpoint_file, device='cuda:0')

for img_file in tqdm(sorted(glob.glob(args.img_dir + '/*'))):

    fname = os.path.basename(img_file)

    img = Image.open(img_file); H, W = img.size
    img_resize_np = np.array(img.resize((512, 512)))

    result = inference_model(model, img_resize_np)
    pred_seg = result.pred_sem_seg.data[0].cpu().numpy()
    pred_seg = np.array(Image.fromarray(pred_seg.astype(np.uint8)).convert('RGB').resize((H, W), resample = Image.NEAREST))
    
    if args.msk_dir != "":
        msk = np.array(Image.open(os.path.join(args.msk_dir, fname.replace('.jpg', '.png'))).convert('RGB'))
        msk[msk > 0] = 1
        pred_seg = pred_seg * msk
    
    img = np.array(img)
    pink = np.zeros((img.shape)); pink[:,:,0] = 255; pink[:,:,2] = 255

    img_seg_vis = img * (1 - pred_seg) + pink * args.alpha * pred_seg + img * (1 - args.alpha) * pred_seg
    imsave(os.path.join(args.seg_dir, fname), img_seg_vis.astype(np.uint8))

   
    