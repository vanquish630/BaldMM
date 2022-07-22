import os, sys
import torch
import numpy as np
import cv2

from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from PIL import Image

from ..utils.segmenter import Segmenter
from . fan import FAN


class TestDataset(Dataset):
    def __init__(self,testpath , image_size = 224):


        self.image_size = image_size
        self.segmenter = Segmenter()
        self.fan = FAN(self.image_size)


        if isinstance(testpath, list):
            self.data_lines = testpath
        elif os.path.isdir(testpath):
            self.data_lines = glob(testpath + '/*.jpg') + glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.data_lines = [testpath]
        else:
            print(f'please check the test path: {testpath}')
            exit()


    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):

        image_path = self.data_lines[idx]
        name = os.path.basename(image_path).split('.')[0]

        image = np.array(Image.open(image_path).convert("RGB"))
        #
        # face, bbox_coordinates = self.segmenter.returnfacebbox(np.array(image), margin=0.05, msk_type='full', getbbox=True)
        # cropped_image = cv2.resize(np.array(face), (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)

        cropped_image, lmks71, mask = self.fan.runTest(image , margin = 0.08)

        # normalized

        lmks71_norm = lmks71[:, :, :2] / self.image_size * 2 - 1
        mask_norm = mask / 255
        cropped_image = np.array(cropped_image) / 255
        cropped_image = cropped_image[0].transpose(2, 0, 1)

        data_dict = {
            'image':  torch.tensor(cropped_image).float(),
            'imagename': name,
            'mask': mask_norm,
        }

        return data_dict
