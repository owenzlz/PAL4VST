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

def prepare_input(img):
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

    device = 0
    half_precision = True
    img_file = '../data/pal4vst/demo_test_data/stylegan2_ffhq/images/seed0958.jpg'
    model_file = './pal4vst/swin-large_upernet_unified_512x512/end2end.pt'

    model = torch.load(model_file).to(device)
    img = np.array(Image.open(img_file).resize((512, 512)))
    img_tensor = prepare_input(img).to(device)
    output = model(img_tensor)

    if half_precision:
        model = model.half()
        img_tensor = img_tensor.half()
    
    imsave('img.jpg', img.astype(np.uint8))
    imsave('output.png', (output.cpu().data.numpy()[0][0] * 255.0).astype(np.uint8))


    