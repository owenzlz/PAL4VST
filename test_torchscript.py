import torch.nn.functional as F
from skimage.io import imsave
from PIL import Image
import numpy as np 
import argparse
import torch
import cv2
import pdb



def get_mean_stdinv(img):
    """
    Compute the mean and std for input image (make sure it's aligned with training)
    """

    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]

    mean_img = np.zeros((img.shape))
    mean_img[:,:,0] = mean[0]
    mean_img[:,:,1] = mean[1]
    mean_img[:,:,2] = mean[2]
    mean_img = np.float32(mean_img)

    std_img = np.zeros((img.shape))
    std_img[:,:,0] = std[0]
    std_img[:,:,1] = std[1]
    std_img[:,:,2] = std[2]
    std_img = np.float64(std_img)

    stdinv_img = 1 / np.float32(std_img)

    return mean_img, stdinv_img

def numpy2tensor(img):
    """
    Convert numpy to tensor
    """
    img = torch.from_numpy(img).transpose(0,2).transpose(1,2).unsqueeze(0).float()
    return img

def prepare_input(img, device):
    """
    Convert numpy image into a normalized tensor (ready to do segmentation)
    """
    mean_img, stdinv_img = get_mean_stdinv(img)
    img_tensor = numpy2tensor(img).to(device)
    mean_img_tensor = numpy2tensor(mean_img).to(device)
    stdinv_img_tensor = numpy2tensor(stdinv_img).to(device)
    img_tensor = img_tensor - mean_img_tensor
    img_tensor = img_tensor * stdinv_img_tensor
    return img_tensor



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

    
    