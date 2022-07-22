import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import face_alignment
import torchfile
import random
from PIL import Image

from .. utils.landmarkDetector import LandmarkDetector71
from .. utils.segment import returnfacebbox, return_skin_mask




class FAN(object):
    def __init__(self , image_size, scale_range = [-2,4] ):
        self.landmarkDetector71 = LandmarkDetector71()
        self.landmarkDetector68 = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        self.image_size = image_size
        self.scale_min = np.random.uniform(scale_range[0],-1)
        self.scale_max = np.random.uniform(1,scale_range[1]) 
        self.scale_random = np.random.uniform(scale_range[0],scale_range[1]) 
    
    def get_landmarks_71(self,image , ignore = False):
        
        lmks71 = self.landmarkDetector71.detect_landmarks(image , ignore)
        
        if lmks71 is None:
            return None
        
        return np.array(lmks71)
    
    def get_landmarks_68(self, image):

        landmark = self.landmarkDetector68.get_landmarks(np.array(image))

        if landmark is None:
            return None

        landmark = np.array(landmark)
        landmark.resize((68,2))
        
        return landmark
    
    def get_small_crop(self,image):
        
        image = np.array(image)
        
        lmk68 = self.get_landmarks_68(np.array(image))

        if lmk68 is None:
            return image , None

        left = np.min(lmk68[:,0]); right = np.max(lmk68[:,0]); 
        top = np.min(lmk68[:,1]); bottom = np.max(lmk68[:,1])

        old_size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        size = int(old_size*1.25)
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])

        # crop image
        dst_pts = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, dst_pts)
        
        image = image/255.
        resized_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        
        resized_image = np.array(resized_image*255, dtype = np.uint8)
                
        lmks71 = self.get_landmarks_71(Image.fromarray(resized_image))
        lmk68 = self.get_landmarks_68(resized_image)
        
        if lmk68 is None or lmks71 is None:
            return resized_image , None
        else:
            lmks71[:68] =lmk68
            
        
        return resized_image , lmks71
    
    def flip(self,image,lmk,mask):
        ### WxHXC
        image = np.fliplr(image)
        mask = np.fliplr(mask)
        lmk[:, 0] = image.shape[1] - lmk[:, 0]
        return image,lmk,mask
        

        
    def get_mask(self,image):
        input_mask = return_skin_mask(np.array(image))  ##512,512
        mask = cv2.resize(input_mask, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        return mask
        
    def get_crop_lmks_from_file(self,image,filepath ,margin,  ):
        
        cropped_image, bbox_coordinates = returnfacebbox(np.array(image),margin = margin, msk_type='full', getbbox=True , padding  = True)
        resized_image = cv2.resize(np.array(cropped_image), (self.image_size, self.image_size),interpolation=cv2.INTER_CUBIC)

        lmks71 = self.get_landmarks_71(Image.fromarray(resized_image) , ignore = True)
        #force_8bytes_long=True
        kpt68 = torchfile.load(filepath)[0]

        ###crop align
        kpt68[:, 0] = kpt68[:, 0] - bbox_coordinates[2]
        kpt68[:, 1] = kpt68[:, 1] - bbox_coordinates[0]

        ###resize align
        kpt68[:, 0] = kpt68[:, 0] * self.image_size / cropped_image.shape[1]
        kpt68[:, 1] = kpt68[:, 1] * self.image_size / cropped_image.shape[0]
        
        lmks71[:68] = kpt68
        
#         lmks71 = np.concatenate([lmks68, np.array(lmks71)[-3:]], )
        return resized_image , lmks71
    
    def get_crop(self,image,margin):
        
        cropped_image, bbox_coordinates = returnfacebbox(np.array(image),margin = margin, msk_type='full', getbbox=True , padding = True)
        resized_image = cv2.resize(np.array(cropped_image), (self.image_size, self.image_size),interpolation=cv2.INTER_CUBIC)

        lmks71 = self.get_landmarks_71(Image.fromarray(resized_image))
        if lmks71 is None:
          return resized_image,None

        lmk68 = self.get_landmarks_68(resized_image)
        
        if lmk68 is None:
            return resized_image , None
        else:
            lmks71[:68] =lmk68
            
        
        return resized_image , lmks71
        
        
        
    def run(self,image, landmark_path = None, crop_type = 'single'):
                    
        
        if landmark_path is None:
            if crop_type =='single':
                
                resized_image , lmks71 = self.get_crop(image,margin = self.scale_random*0.05)
                if lmks71 is None:
                    return None,None,None
                    
                mask = self.get_mask(resized_image)
                
                if random.random() <= 0.5:
                    resized_image,lmks71,mask = self.flip(resized_image,lmks71,mask)
                    
                return resized_image[None,...] , lmks71[None,...] , mask[None,...]
                
            if crop_type =='multi':
                
                resized_image_small , lmks71_small = self.get_small_crop(image)
                mask_small = self.get_mask(resized_image_small)
                
                resized_image_med , lmks71_med = self.get_crop(image,margin = 0.05)
                mask_med = self.get_mask(resized_image_med)

                resized_image_large , lmks71_large = self.get_crop(image,margin = self.scale_max*0.05)
                mask_large = self.get_mask(resized_image_large)
                
                if lmks71_small is None or lmks71_med is None or lmks71_large is None:
                    return None,None,None
                
                if random.random() <= 0.4:
                    resized_image_small,lmks71_small,mask_small = self.flip(resized_image_small,lmks71_small,mask_small)

                if random.random() <= 0.4:
                    resized_image_med,lmks71_med,mask_med = self.flip(resized_image_med,lmks71_med,mask_med)
                    
                if random.random() <= 0.4:
                    resized_image_large,lmks71_large,mask_large = self.flip(resized_image_large,lmks71_large,mask_large)
                    
                resized_image = np.stack([resized_image_small, resized_image_med,resized_image_large], axis = 0)
                lmks71 = np.stack([lmks71_small, lmks71_med,lmks71_large], axis = 0)
                mask = np.stack([mask_small,mask_med,mask_large],axis = 0)

                return resized_image , lmks71 ,mask
            #[(resized_image_small , lmks71_small), (resized_image_med , lmks71_med) , (resized_image_large , lmks71_large) ]
                
                
        else:
            if crop_type =='single':
                resized_image , lmks71 = self.get_crop_lmks_from_file(image,landmark_path,margin =self.scale_min*0.05,)
                mask = self.get_mask(resized_image)

                if random.random() <= 0.4:
                    resized_image,lmks71,mask = self.flip(resized_image,lmks71,mask)
                
                return resized_image[None,...] , lmks71[None,...], mask[None,...]
                
            if crop_type =='multi':
                
                resized_image_small , lmks71_small = self.get_crop_lmks_from_file(image,landmark_path,margin = self.scale_min*0.05)
                mask_small = self.get_mask(resized_image_small)

                resized_image_med , lmks71_med = self.get_crop_lmks_from_file(image,landmark_path,margin = 0.05)
                mask_med = self.get_mask(resized_image_med)

                resized_image_large , lmks71_large = self.get_crop_lmks_from_file(image,landmark_path,margin = self.scale_max*0.05)
                mask_large = self.get_mask(resized_image_large)
                
                if random.random() <= 0.4:
                    resized_image_small,lmks71_small,mask_small = self.flip(resized_image_small,lmks71_small,mask_small)

                if random.random() <= 0.4:
                    resized_image_med,lmks71_med,mask_med = self.flip(resized_image_med,lmks71_med,mask_med)
                    
                if random.random() <= 0.4:
                    resized_image_large,lmks71_large,mask_large = self.flip(resized_image_large,lmks71_large,mask_large)

                
                resized_image = np.stack([resized_image_small, resized_image_med,resized_image_large], axis = 0)
                lmks71 = np.stack([lmks71_small, lmks71_med,lmks71_large], axis = 0)
                mask = np.stack([mask_small,mask_med,mask_large],axis = 0)


                
                return resized_image , lmks71 ,mask


    def runTest(self, image , margin = 0.05):

        resized_image, lmks71 = self.get_crop(image, margin= margin)

        mask = self.get_mask(resized_image)


        return resized_image[None, ...], lmks71[None, ...], mask[None, ...]

