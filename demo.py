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

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(savename, img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    imgout = np.array(img_flo[:, :, [2,1,0]], dtype='uint8')
    cv2.imwrite(savename , imgout)
    # cv2.waitKey()


def vizproject(savename, img1, img2, flo):
    u = flo[:,:,0]
    v = flo[:,:,1]
    refimg = img1
    curimg = img2
    imgproj = np.zeros(img1.shape)
    reverseflowu = np.zeros(u.shape)
    reverseflowv = np.zeros(v.shape)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            ii = int(i + u[i,j])
            jj = int(j + v[i,j])
            if (ii>=0) and (jj>=0) and (ii<img1.shape[0]) and (jj<img1.shape[1]):
                if (u[i,j]**2 + v[i,j]**2) > (reverseflowu[i,j]**2 + reverseflowv[i,j]**2):
                    imgproj[i,j,:] = curimg[ii,jj,:]
                    reverseflowu[i,j] = u[i,j]
                    reverseflowv[i,j] = v[i,j]
    # map flow to rgb image
    errormap = refimg  - imgproj + 128
    flo = flow_viz.flow_to_image(flo)
    uimage = np.repeat(u[:,:,np.newaxis]*10, 3, axis=2)
    vimage = np.repeat(v[:,:,np.newaxis]*10, 3, axis=2)
    # img_flo1 = np.concatenate([refimg, 0.5*(refimg + imgproj)], axis=0)
    # img_flo2 = np.concatenate([curimg, 0.5*(refimg +curimg)], axis=0)
    img_flo1 = np.concatenate([refimg, uimage], axis=0)
    img_flo2 = np.concatenate([curimg, vimage], axis=0)
    img_flo = np.concatenate([img_flo1, img_flo2], axis=1)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    imgout = np.array(img_flo[:, :, [2,1,0]], dtype='uint8')
    cv2.imwrite(savename , imgout)    # cv2.waitKey()

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model)) # , map_location=torch.device('cpu')))

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
            flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
            image1 = padder.unpad(image1[0]).permute(1, 2, 0).cpu().numpy()
            image2 = padder.unpad(image2[0]).permute(1, 2, 0).cpu().numpy()
            subname = imfile1.split("/")
            savename = os.path.join(args.result, subname[-1])
            vizproject(savename, image1, image2, flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-sintel.pth', help="restore checkpoint")
    parser.add_argument('--path', default='demo-frames', help="dataset for evaluation")
    parser.add_argument('--result', default='/work/eexna/Creative/raft_results_cur1p', help="save result")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)


