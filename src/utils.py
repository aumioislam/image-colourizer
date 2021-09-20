#utility functions were adapted from Richard Zhang's Colorful Image Colorization Repo

from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F

def read_image(path):
    img = np.asarray(Image.open(path))
    if (img.ndim == 2):
        img = np.tile(img[:,:,None], 3)
    return img, img.shape[:2]

def resize(img, dims=(256,256), resample=3):
    return np.asarray(Image.fromarray(img).resize((dims[1],dims[0]), resample=resample))

def preprocess_img(orig, dims=(256,256), resample=3):

    rs = resize(orig, dims=dims, resample=resample)

    orig_l = color.rgb2lab(orig)[:,:,0]
    orig_l = orig_l[None,None,:,:]
    orig_l_tens = torch.from_numpy(orig_l).float()

    rs_lab = color.rgb2lab(rs)

    rs_l = rs_lab[:,:,0]
    rs_l = rs_l[None,:,:]
    rs_l_tens = torch.from_numpy(rs_l).float()

    rs_ab = rs_lab[:,:,1:]
    rs_ab = rs_ab.transpose((2,0,1))
    rs_ab_tens = torch.from_numpy(rs_ab).float()

    return (orig_l_tens, rs_l_tens, rs_ab_tens)

def postprocess_img(orig_l_tens, pred_ab, mode='bilinear'):
    dims_orig = orig_l_tens.shape[2:]
    dims = pred_ab.shape[2:]

    if (dims_orig[0] != dims[0] or dims_orig[1] != dims[1]):
        out_ab = F.interpolate(pred_ab, size=dims_orig, mode=mode, align_corners=False)
    else:
        out_ab = pred_ab

    out_lab = torch.cat((orig_l_tens, out_ab), dim=1)
    return color.lab2rgb(torch.squeeze(out_lab).data.cpu().numpy().transpose((1,2,0)))
