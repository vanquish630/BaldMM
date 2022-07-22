'''
Default config for DECA
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.deca_dir = abs_deca_dir
cfg.device = 'cuda'
cfg.device_id = '0'

cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'model_lfpw_dice.tar')

# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.topology_path = os.path.join(cfg.deca_dir, 'data', 'head_template_uv.obj')
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.model.dense_template_path = os.path.join(cfg.deca_dir, 'data', 'texture_data_256.npy')
cfg.model.fixed_displacement_path = os.path.join(cfg.deca_dir, 'data', 'fixed_displacement_256.npy')
cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model_avg5.pkl')
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'data', 'landmark_embedding_71_upper.npy')
cfg.model.face_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_mask.png') 
cfg.model.face_eye_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_eye_mask.png') 
cfg.model.mean_tex_path = os.path.join(cfg.deca_dir, 'data', 'mean_texture_symmetric.png')
cfg.model.flame_tex_path = os.path.join(cfg.deca_dir, 'data', 'FLAME_texture.npz')
cfg.model.tex_type = 'FLAME' # BFM, FLAME, albedoMM
cfg.model.uv_size = 256
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_shape = 100
cfg.model.n_tex = 50
cfg.model.n_exp = 50
cfg.model.n_cam = 3
cfg.model.n_pose = 6
cfg.model.n_light = 27
cfg.model.use_tex = True
cfg.model.jaw_type = 'aa' # default use axis angle, another option: euler

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.batch_size = 24
cfg.dataset.num_workers = 2
cfg.dataset.image_size = 224

