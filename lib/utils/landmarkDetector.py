import sys
sys.path.append("..")

import os
import torch
from PIL import Image
import numpy as np
import math
import cv2
import face_alignment

from . import util
from . import transforms
from .. models.fan_model import FAN
from . segmenter import Segmenter

class LandmarkDetector71:

    def __init__(self, num_lmks = 71, checkpoint_dir = "./data/"):

        self.checkpoint_dir = os.path.join(checkpoint_dir , "71lmk_checkpoint.pth.tar")

        self.num_lmks = num_lmks
        self.fan_model = FAN(2, 73)
        self.fan_model = torch.nn.DataParallel(self.fan_model).cuda()
        print(f"=> Loading checkpoint {self.checkpoint_dir}")
        self.checkpoint = torch.load(self.checkpoint_dir)
        self.fan_model.load_state_dict(self.checkpoint['state_dict'])
        print(f"=> Loaded checkpoint {self.checkpoint_dir} (epoch {self.checkpoint['epoch']})")
        self.landmarkDetector68 = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        self.segmenter = Segmenter()


    def get_preds(self, scores):
        ''' get predictions from score maps in torch Tensor
            return type: torch.LongTensor
        '''
        assert scores.dim() == 4, 'Score maps should be 4-dim'
        # batch, chn, height, width ===> batch, chn, height*width
        # chn = 68
        # height*width = score_map
        maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

        maxval = maxval.view(scores.size(0), scores.size(1), 1)
        idx = idx.view(scores.size(0), scores.size(1), 1) + 1

        preds = idx.repeat(1, 1, 2).float()

        # batchsize * numPoints * 2
        # 0 is x coord
        # 1 is y coord
        # shape = batchsize, numPoints, 2
        preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(2)) + 1

        pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
        preds *= pred_mask
        return preds

    def final_preds(self, output, center, scale, res,):
        if output.size(1) == 136:
            coords = output.view((output.szie(0), self.num_lmks, 2))
        else:
            coords = self.get_preds(output[:,:self.num_lmks,:,:])  # float type


        # output shape is batch, num_lmks, 64, 64
        # coords shape is batch, num_lmks, 2
        # pose-processing
        for n in range(coords.size(0)):
            for p in range(coords.size(1)):
                hm = output[n][p]
                px = int(math.floor(coords[n][p][0]))
                py = int(math.floor(coords[n][p][1]))
                if px > 1 and px < res[0] and py > 1 and py < res[1]:
                    diff = torch.Tensor(
                        [hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1] - hm[py - 2][px - 1]])
                    coords[n][p] += diff.sign() * .25
        coords += 0.5
        preds = coords.clone()


        # Transform back
        for i in range(coords.size(0)):
            preds[i] = transforms.transform_preds(coords[i], center, scale, res)

        if preds.dim() < 3:
            preds = preds.view(1, preds.size())
        return preds


    def validate(self,inputs, model,center,scale):


        model.eval()
        input_var = torch.autograd.Variable(inputs.cuda())
        output = model(input_var)
        score_map = output[-1].data.cpu()

        preds = self.final_preds(score_map, center,scale, [64, 64])

        return preds

    def get_landmarks_68(self, image):
        image= np.array(image)
        landmarks = self.landmarkDetector68.get_landmarks(image)

        if landmarks is None:
            return None

        landmarks = np.array(landmarks)
        landmarks.resize((68, 2))

        return landmarks


    def get_landmarks_71(self, image , ignore):

        img = np.array(image)
        # cropped_image, bbox_coordinates = self.segmenter.returnfacebbox(image, msk_type='full', getbbox=True)
        # img = cv2.resize(np.array(cropped_image), (256, 256), interpolation=cv2.INTER_CUBIC)


        self.c = torch.Tensor((img.shape[0] / 2, img.shape[1] / 2))
        self.s = 1.8
        inp = transforms.crop(util.im_to_torch(img), self.c, self.s, [256, 256], rot=0)


        landmarks71 = self.validate(inp[None, ...], self.fan_model, self.c , self.s)

        landmarks71 = np.array(landmarks71[0])


        # landmarks68 = self.get_landmarks_68(img)
        # if landmarks68 is not None:
        #     landmarks71[:68] =landmarks68
        # else:
        #     print("lmk68 did not work")
        #     if ignore == True:
        #       landmarks71 = landmarks71
        #     else:
        #       landmarks71 = None



        return landmarks71 ##num_lmks X 2



    def detect_landmarks(self , image , ignore = False):

        # try:
        #   cropped_image, bbox_coordinates = self.segmenter.returnfacebbox(np.array(image), msk_type='full', getbbox=True , margin = 0.2)
        # except:
        #   print("failed to detect face")
        #   return None
        
        # resized_image = cv2.resize(np.array(cropped_image), (256, 256), interpolation=cv2.INTER_CUBIC)
       
        landmarks71 = self.get_landmarks_71(np.array(image) , ignore)

        if landmarks71 is None:
            print("failed to detect landmarks")
            return None
        else:

            ###resize align
            # landmarks71[:, 0] = landmarks71[:, 0] * cropped_image.shape[1] / resized_image.shape[1]
            # landmarks71[:, 1] = landmarks71[:, 1] * cropped_image.shape[0] / resized_image.shape[0]
            
            landmarks71[68:,1] =  landmarks71[68:,1] + 6
            # if landmarks71[69,0] < landmarks71[68,0]:
            #     landmarks71[69, 0] = 2*landmarks71[68,0] - landmarks71[70,0]

            ###crop align
            # landmarks71[:, 0] = landmarks71[:, 0] + bbox_coordinates[2]
            # landmarks71[:, 1] = landmarks71[:, 1] + bbox_coordinates[0]

            return landmarks71
