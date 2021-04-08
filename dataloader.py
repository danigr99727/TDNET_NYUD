import os
import torch
import numpy as np
import imageio
import cv2
import pdb

def color_map( N=40, normalized=False):
    """
    Return Color Map in PASCAL VOC format
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255.0 if normalized else cmap
    return cmap


def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)]


class cityscapesLoader():

  
    colors = color_map(N=256)
    label_colours = dict(zip(range(256), colors))


    def __init__(self,img_path,in_size):
        self.img_path = img_path
        self.n_classes = 40
        self.files = recursive_glob(rootdir=self.img_path, suffix=".png")
        self.files.sort()
        self.files_num = len(self.files)
        self.data = []
        self.size = (in_size[1],in_size[0])
        self.mean = np.array([.485, .456, .406])
        self.std = np.array([.229, .224, .225])

    def load_frames(self):

        for idx in range(self.files_num):
            img_path = self.files[idx].rstrip()
            img_name = img_path.split('/')[-1]
            folder = img_path.split('/')[-2]

            #img = cv2.imread(img_path).astype(np.float32)
            img = imageio.imread(img_path)
            ori_size = img.shape[:-1]

            img = cv2.resize(img,self.size)/255.0
            img = (img-self.mean)/self.std

            img = img.transpose(2, 0, 1)
            img = img[np.newaxis,:]
            img = torch.from_numpy(img).float()

            self.data.append([img,img_name,folder,self.size])

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r #/ 255.0
        rgb[:, :, 1] = g #/ 255.0
        rgb[:, :, 2] = b #/ 255.0
        return rgb



















	
        
