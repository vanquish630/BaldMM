# Learning to regulate 3D head shape by removing occluding hair from in-the-wild images 


![samples](demos/Sample.png)


## Getting Started

### Cloning the repository

    https://github.com/vanquish630/BaldMM.git
    cd BaldMM

### Requirements

1. Python 3.7 (numpy, skimage, scipy, opencv)
2. PyTorch >= 1.6
3. Pytorch3d, installation instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

Alternatively you can run 

    pip install -r requirements.txt


## How to use

### Downloading pretrained models

Download the following trained models and place them in the **./data** folder.

1. [Trained 3DMM model](https://drive.google.com/file/d/1rdodi1D0YEGuu09G75YIA2NxhGeqwWZA/view?usp=sharing)
2. [Trained custom landmark model](https://drive.google.com/file/d/1UtSW4zx232qtIMwQyli5Uu9jfdsdfJuJ/view?usp=sharing)

Downloading the FLAME texture model

1. Download the FLAME texture space from the [FLAME texture space](https://flame.is.tue.mpg.de/download.php) and place it in **./data** folder.
   
2. Modify the *flame_tex_path* parameter in the **./lib/utils/config** file with the path of the downloaded texture space. 
    

### Run the demo

     python demos/demo.py -i input -s results --saveObj True --useTex True --device gpu

Where *i* is the input folder path and *s* the savefolder path. Please run `python demos/demo_reconstruct.py --help` for more details. 


## Acknowledgements
We have used code from the following repositories:  
- [DECA](https://github.com/YadiraF/DECA)
- [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch) and [TF_FLAME](https://github.com/TimoBolkart/TF_FLAME) for the FLAME model  
- [Pytorch3D](https://pytorch3d.org/), [neural_renderer](https://github.com/daniilidis-group/neural_renderer), [SoftRas](https://github.com/ShichenLiu/SoftRas) for rendering  
- [kornia](https://github.com/kornia/kornia) for diffrentiable image processing
- [face_segmentation](https://github.com/zllrunning/face-parsing.PyTorch) for skin mask
- [VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch) for identity loss  


