import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image
import math
import os
import kornia
from kornia import morphology as morph

from .. models.segmentation_model import BiSeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize(input , device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
  #print(input.device)

  output = ((input - torch.Tensor([0.485, 0.456, 0.406]).to(device)) / torch.Tensor([0.229, 0.224, 0.225]).to(device))
  #print(output.device)

  return output

def inv_normalize(input , device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    output = ((input - torch.Tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]).to(device)) / torch.Tensor([1/0.229, 1/0.224, 1/0.225]).to(device))
    return output



def segment_image(image,modelpath = './data/79999_iter.pth', size = 256):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    # print(os.listdir("./"))
    net.load_state_dict(torch.load(modelpath) )
    net.float()
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
    image = cv2.resize(image, (512,512),interpolation = cv2.INTER_CUBIC)
    with torch.no_grad():
      
      img = to_tensor(image)
      img = torch.unsqueeze(img, 0)
      img = img
      out = net(img.float())[0]
      img = inv_normalize(img)
      output = (np.transpose(np.array(out.squeeze(0).cpu()),(1,2,0)).argmax(2).astype(np.uint8))

      #out size = 512,512

    return output

def return_hair_mask(image):
  ##in 0-1

  image = (image - np.min(image))/(np.max(image) - np.min(image))

  out = segment_image(image)
  hairmask =  np.where(out==17,1,0)  #index corresponding to hair segmentation
  return hairmask



def parsing(out):
  base = np.zeros((out.shape[0],out.shape[1]))
  base[out == 1] = 1
  base[out == 2] = 1
  base[out == 3] = 1
  base[out == 4] = 1
  base[out == 5] = 1
  base[out == 6] = 1
  base[out == 7] = 1
  base[out == 8] = 1
  base[out == 9] = 1
  base[out == 10] = 1
  base[out == 11] = 1
  base[out == 12] = 1
  base[out == 13] = 1
  base[out == 15] = 1
  base[out == 17] = 0

  return np.array(base , dtype=np.uint8)



def return_face_mask(image):
  # if np.max(image) > 1 :
  #   image = image/255

  image = (image - np.min(image))/(np.max(image) - np.min(image))

  out = segment_image(image)
  #print(np.unique(out))
  facemask = np.zeros(out.shape)
  facemask[(out==1)] = 1  #face
  facemask[(out==2)] = 1  #eyebrow r
  facemask[(out==3)] = 1  #eyebrow l

  facemask[(out==4)] = 1  #eye r
  facemask[(out==5)] = 1  #eye l
  facemask[(out==7)] = 1  #ear l
  facemask[(out==8)] = 1  #ear r
  facemask[(out==10)] = 1  #nose
  facemask[(out==11)] = 1  #lip
  facemask[(out==12)] = 1  #lip up
  facemask[(out==13)] = 1  #lip dwn

  im_th = np.array(facemask*255 , dtype=np.uint8)

  kernel = np.ones((7,7), np.uint8)

  im_th = cv2.dilate(im_th, kernel, iterations=6)
  im_th = cv2.erode(im_th, kernel, iterations=6)

  im_floodfill = im_th.copy()

  # Mask used to flood filling.
  # Notice the size needs to be 2 pixels than the image.
  h, w = im_th.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)

  # Floodfill from point (0, 0)
  cv2.floodFill(im_floodfill, mask, (0,0), 255);

  # Invert floodfilled image
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)

  # Combine the two images to get the foreground.
  im_out = im_th | im_floodfill_inv

  return im_out

  
def return_full_mask(image):
  # if np.max(image) > 1 :
  #   image = image/255

  image = (image - np.min(image))/(np.max(image) - np.min(image))


  out = segment_image(image)
  #print(np.unique(out))
  facemask = np.zeros(out.shape)
  facemask[(out==1)] = 1  #face
  facemask[(out==2)] = 1  #eyebrow r
  facemask[(out==3)] = 1  #eyebrow l

  facemask[(out==4)] = 1  #eye r
  facemask[(out==5)] = 1  #eye l
  facemask[(out==7)] = 1  #ear l
  facemask[(out==8)] = 1  #ear r
  facemask[(out==10)] = 1  #nose
  facemask[(out==11)] = 1  #lip
  facemask[(out==12)] = 1  #lip up
  facemask[(out==13)] = 1  #lip dwn

  facemask[(out==17)] = 1  #hair
  facemask[(out==18)] = 1  #apparel

  im_th = np.array(facemask*255 , dtype=np.uint8)

  kernel = np.ones((7,7), np.uint8)

  im_th = cv2.dilate(im_th, kernel, iterations=6)
  im_th = cv2.erode(im_th, kernel, iterations=6)

  im_floodfill = im_th.copy()

  # Mask used to flood filling.
  # Notice the size needs to be 2 pixels than the image.
  h, w = im_th.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)

  # Floodfill from point (0, 0)
  cv2.floodFill(im_floodfill, mask, (0,0), 255);

  # Invert floodfilled image
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)

  # Combine the two images to get the foreground.
  im_out = im_th | im_floodfill_inv

  return im_out


def return_skin_mask(image):
  # if np.max(image)> 10 :
  #   image = image/255

  image = (image - np.min(image))/(np.max(image) - np.min(image))

  out= segment_image(image)
  #print(np.unique(out))
  facemask = np.zeros(out.shape)
  facemask[(out==1)] = 1  #face
  # facemask[(out==2)] = 1  #eyebrow r
  # facemask[(out==3)] = 1  #eyebrow l

  # facemask[(out==4)] = 1  #eye r
  # facemask[(out==5)] = 1  #eye l
  # facemask[(out==7)] = 1  #ear l
  # facemask[(out==8)] = 1  #ear r
  facemask[(out==10)] = 1  #nose
  # facemask[(out==11)] = 1  #lip
  facemask[(out==12)] = 1  #lip up
  facemask[(out==13)] = 1  #lip dwn

  im_th = np.array(facemask*255 , dtype=np.uint8)

  # kernel = np.ones((7,7), np.uint8)

  # im_th = cv2.dilate(im_th, kernel, iterations=6)
  # im_th = cv2.erode(im_th, kernel, iterations=6)

  # im_floodfill = im_th.copy()

  # # Mask used to flood filling.
  # # Notice the size needs to be 2 pixels than the image.
  # h, w = im_th.shape[:2]
  # mask = np.zeros((h+2, w+2), np.uint8)

  # # Floodfill from point (0, 0)
  # cv2.floodFill(im_floodfill, mask, (0,0), 255);

  # # Invert floodfilled image
  # im_floodfill_inv = cv2.bitwise_not(im_floodfill)

  # # Combine the two images to get the foreground.
  # im_out = im_th | im_floodfill_inv

  return im_th



def return_coma_mask(image):
  # if np.max(image)> 10 :
  #   image = image/255

  image = (image - np.min(image))/(np.max(image) - np.min(image))

  out= segment_image(image)
  #print(np.unique(out))
  facemask = np.zeros(out.shape)
  facemask[(out==1)] = 1  #face
  facemask[(out==2)] = 1  #eyebrow r
  facemask[(out==3)] = 1  #eyebrow l

  facemask[(out==4)] = 1  #eye r
  facemask[(out==5)] = 1  #eye l
  facemask[(out==7)] = 1  #ear l
  facemask[(out==8)] = 1  #ear r
  facemask[(out==10)] = 1  #nose
  facemask[(out==11)] = 1  #lip
  facemask[(out==12)] = 1  #lip up
  facemask[(out==13)] = 1  #lip dwn
  facemask[(out==17)] = 1  #hair
  facemask[(out==18)] = 1  #apparel

#  facemask[(out==14)] = 1  #neck

  im_th = np.array(facemask*255 , dtype=np.uint8)

  # kernel = np.ones((7,7), np.uint8)

  # im_th = cv2.dilate(im_th, kernel, iterations=6)
  # im_th = cv2.erode(im_th, kernel, iterations=6)

  # im_floodfill = im_th.copy()

  # # Mask used to flood filling.
  # # Notice the size needs to be 2 pixels than the image.
  # h, w = im_th.shape[:2]
  # mask = np.zeros((h+2, w+2), np.uint8)

  # # Floodfill from point (0, 0)
  # cv2.floodFill(im_floodfill, mask, (0,0), 255);

  # # Invert floodfilled image
  # im_floodfill_inv = cv2.bitwise_not(im_floodfill)

  # # Combine the two images to get the foreground.
  # im_out = im_th | im_floodfill_inv

  return im_th


def returnfacebbox(image, margin=0.05, msk_type='face', getbbox=False, padding=False):
    ### input is numpy Image
    # image = Image.fromarray(np.array(image), mode="RGB")
    if msk_type == 'face':
        mask = return_face_mask(image)

    elif msk_type == 'full':
        mask = return_full_mask(image)

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # print(mask.max(), mask.dtype, mask.shape)
    # print(image.max(), image.dtype, image.shape )

    masked = cv2.bitwise_and(np.array(image), np.array(image), mask=mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)

    if len(contours) == 0:
        return None

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    start = [int(boundRect[i][0]), int(boundRect[i][1])]
    end = [int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])]

    # print(start)
    # print(end)

    m_y = margin
    m_x = (((end[1] - start[1]) / (end[0] - start[0])) * (1 + 2 * m_y) - 1) * 0.5

    m_x = math.floor(m_x * 100) / 100

    margin_x = m_x * (end[0] - start[0])
    margin_y = m_y * (end[1] - start[1])

    start_y = max(int(start[1] - margin_y), 0)
    start_x = max(int(start[0] - margin_x), 0)

    end_y = min(int(end[1] + margin_y), image.shape[0])
    end_x = min(int(end[0] + margin_x), image.shape[1])

    face = (np.array(image)[start_y: end_y, start_x: end_x])
    height, width, _ = face.shape

    if padding == True:

        if width > height and width - height > 0.02 * height:
            result = Image.new(Image.fromarray(face).mode, (width, width), (0, 0, 0))
            result.paste(Image.fromarray(face), (0, (width - height) // 2))
            result = np.array(result)

        elif width < height and height - width > 0.02 * width:
            result = Image.new(Image.fromarray(face).mode, (height, height), (0, 0, 0))
            result.paste(Image.fromarray(face), ((height - width) // 2, 0))
            result = np.array(result)

        else:
            result = face

    elif padding == False:
        result = face

    if getbbox is False:
        return face
    else:
        return face, [start_y, end_y, start_x, end_x]


def segment_image_grad(image,segmentation_network,modelpath = './data/79999_iter.pth'):
    
    ### input Tensor, 0-1 , bz,c,h,w
    
    # n_classes = 19
    # net = BiSeNet(n_classes=n_classes , device = device)      ####0-1
    # net.load_state_dict(torch.load(modelpath) )
    # net.to(device)
    # net.eval()



    # image = image.to(device)

    image = (image - torch.min(image))/(torch.max(image) - torch.min(image))

    image.retain_grad()

    
    img = torch.nn.functional.interpolate(image, scale_factor=(2, 2), mode= 'bilinear',recompute_scale_factor=True,align_corners=False)
   
    out = segmentation_network(img.float())[0]

    m = torch.nn.Softmax(dim=1)

    out = m(out)

    out1 = out.clone()

# facemask[(out==0)] = 1  #backgrnd
    
# facemask[(out==1)] = 1  #face
# facemask[(out==2)] = 1  #eyebrow r
# facemask[(out==3)] = 1  #eyebrow l

# facemask[(out==4)] = 1  #eye r
# facemask[(out==5)] = 1  #eye l
# facemask[(out==7)] = 1  #ear l
# facemask[(out==8)] = 1  #ear r
# facemask[(out==10)] = 1  #nose
# facemask[(out==11)] = 1  #lip
# facemask[(out==12)] = 1  #lip up
# facemask[(out==13)] = 1  #lip dwn

    out1[:,[0,2,3,4,5,7,8,11,14,16,17,18],:,:] = out1[:,[0,2,3,4,5,7,8,11,14,16,17,18],:,:]*0

    out1 = out1.sum(dim = 1)

    out1 = out1[:,None,...]



    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)

    out1 = morph.dilation(out1, kernel) 
    out1 = morph.dilation(out1, kernel) 
    out1 = morph.dilation(out1, kernel) 
    out1 = morph.erosion(out1, kernel)
    out1 = morph.erosion(out1, kernel)
    out1 = morph.erosion(out1, kernel)

    out1 = kornia.filters.box_blur(out1, (3, 3))

    out1 = kornia.filters.box_blur(out1, (3, 3))

    out1 = torch.round(out1)

    out1 = torch.nn.functional.interpolate(out1, scale_factor=(0.5, 0.5), mode= 'bilinear',recompute_scale_factor=True,align_corners=False)

    return out1.squeeze()


