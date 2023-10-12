from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from PIL import Image
import numpy as np 
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

def draw_text_on_image(image_path, text, scale):
    # Open an image file
    with Image.open(image_path) as img:
        width, height = img.size

        # Determine the size of the text based on the image size and scale
        text_size = int(min(width, height) * scale)
        # Use a truetype font from PIL, adjust the path to the font file as needed
        # Here we're using the DejaVuSans which is typically installed with matplotlib
        font = ImageFont.truetype("DejaVuSans.ttf", text_size)
        # Create a draw object
        draw = ImageDraw.Draw(img)
        # Determine text position
        text_position = (20, 0) # horizontal, vertical
        # Add text to image
        draw.text(text_position, text, font=font, fill=(255, 105, 180))

    return img
