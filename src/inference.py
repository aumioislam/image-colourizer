import os
import sys
import torch
import argparse
from colorizers import CNN, CNN_AE, load
from utils, import read_image, preprocess_img, postprocess_img
import torch.nn.functional as F

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CNN', help='choose model for inference')
    parser.add_argument('--img-path', type=str, default='None', help='Path to image')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        sys.exit(0)

    if args.model == 'CNN':
        model = CNN().to(device)
        dims = (256,256)
    elif args.model == 'CNN_AE':
        model = CNN_AE().to(device)
        dims = (128,128)
    else:
        sys.exit(0)

    if os.path.isfile(args.img_path):
        path = args.img_path
    else:
        sys.exit(0)

    model = load(model, training=False)

    img, img_shape = read_image(path)
    inputs = preprocess_img(img) 
    pred = model(inputs[1])
    img = postprocess_img(inputs[0], pred)
