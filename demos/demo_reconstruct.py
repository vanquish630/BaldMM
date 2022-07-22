

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.baldnet import BaldNet
from lib.utils import util
from lib.utils.config import cfg
from lib.datasets import test_dataset

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images
    testdata = test_dataset.TestDataset(args.inputpath)

    # run BaldNet
    cfg.model.use_tex = args.useTex
    baldnet = BaldNet(config=cfg, device=device)
    # for i in range(len(testdata)):
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        name = name.split("\\")[-1]
        images = testdata[i]['image'].to(device)[None, ...]
        os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        codedict = baldnet.encode(images)
        np.save(os.path.join(savefolder, name, name + '_parameters.npy'), codedict)
        plt.imsave(os.path.join(savefolder, name, name + '_input.png'), images.cpu().permute(0, 2, 3, 1).numpy()[0])

        opdict, visdict = baldnet.decode(codedict)  # tensor

        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        if args.saveDepth:
            depth_image = baldnet.render.render_depth(opdict['transformed_vertices']).repeat(1, 3, 1, 1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj:
            print(f"{name}")
            print(os.path.join(savefolder, name, name + '.obj'))
            baldnet.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        if args.saveVis:
            cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), baldnet.visualize(visdict))
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images' , 'landmarks2d']:
                if vis_name not in visdict.keys():
                    continue
                image = util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name + '.jpg'),
                            util.tensor2image(visdict[vis_name][0]))
    print(f'-- please check the results in {savefolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BaldNet')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save keypoints')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image')
    parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj. \
                            Note that saving objs could be slow')
    parser.add_argument('--saveImages', default=True, type=lambda x: x.lower() in ['False', '0'],
                        help='whether to save visualization output as seperate images')
    main(parser.parse_args())