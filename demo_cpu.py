import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from torch.autograd import Variable
import torch.nn as nn



parser = argparse.ArgumentParser()
parser.add_argument('--model', default='models/raft-sintel.pth', help="restore checkpoint")
parser.add_argument('--path', default='/work/eexna/Creative/results/ESPRITlandscape', help="dataset for evaluation")
parser.add_argument('--result', default='/work/eexna/Creative/raft_results_cur1p', help="save result")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
parser.add_argument('--gpu', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()

if args.gpu:
    DEVICE = 'cuda'# 'cpu'
else:
    DEVICE = 'cpu'

print('using ' + DEVICE)

def load_image(imfile):
    im = Image.open(imfile)
    (width, height) = (im.width // 4, im.height // 4)
    print(str(width) + 'x' + str(height))
    img = np.array(im.resize((width, height))).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()
        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo
        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0
        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size()))#.cuda()
        if x.is_cuda:
            mask = mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        return output*mask

model = torch.nn.DataParallel(RAFT(args))
model.load_state_dict(torch.load(args.model , map_location=torch.device('cpu')))

model = model.module
model.to(DEVICE)
model.eval()

with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            warpimg1 = warp(image2,flow_up)
            wrapimg2 = warp(image1,flow_up)
            image1 = padder.unpad(image1[0]).permute(1, 2, 0).cpu().numpy()
            image2 = padder.unpad(image2[0]).permute(1, 2, 0).cpu().numpy()
            warpimg1 = padder.unpad(warpimg1[0]).permute(1, 2, 0).cpu().numpy()
            wrapimg2 = padder.unpad(wrapimg2[0]).permute(1, 2, 0).cpu().numpy()
            # save result
            subname = imfile1.split("/")
            savename = os.path.join(args.result, subname[-1])
            img_flo1 = np.concatenate([image1, np.abs(image1 - warpimg1)], axis=0)
            img_flo2 = np.concatenate([image2, np.abs(image2 - wrapimg2)], axis=0)
            img_flo = np.concatenate([img_flo1, img_flo2], axis=1)
            imgout = np.array(img_flo[:, :, [2,1,0]], dtype='uint8')
            cv2.imwrite(savename , imgout)

