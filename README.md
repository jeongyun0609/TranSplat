## TranSplat: Surface Embedding-guided 3D Gaussian Splatting for Transparent Object Manipulation

Official Repository for "TranSplat: Surface Embedding-guided 3D Gaussian Splatting for Transparent Object Manipulation", Underreview

<div align="left">  
  <a href="https://scholar.google.com/citations?hl=ko&user=vW2JtFAAAAAJ">Jeongyun Kim</a>,
  <a href="https://rpm.snu.ac.kr">Jeongho Noh</a>,
  <a href="https://scholar.google.com/citations?user=u6VDnlgAAAAJ&hl=ko&oi=ao">Dong-Guw Lee</a>,  
  <a href="https://ayoungk.github.io/">Ayoung Kim</a>
</div>


### Overview of the TranSplat

<div align="center">
  
![put method picture here](./assets/pipline.png)
</div>
TLDR: We propose a new Gaussian Splatting-based depth completion framework specifically for transparent objects based on Surface Embedding features.

### Abstract

Transparent object manipulation remains a significant challenge in robotics due to the difficulty of acquiring accurate and dense depth measurements. Conventional depth sensors often fail with transparent objects, resulting in incomplete or erroneous depth data. Existing depth completion methods struggle with interframe consistency and incorrectly model transparent objects as Lambertian surfaces, leading to poor depth reconstruction. To address these challenges, we propose TranSplat, a surface embedding-guided 3D Gaussian Splatting method tailored for transparent objects. TranSplat uses a latent diffusion model to generate surface embeddings that provide consistent and continuous representations, making it robust to changes in viewpoint and lighting. By integrating these surface embeddings with input RGB images, TranSplat effectively captures the complexities of transparent surfaces, enhancing the splatting of 3D Gaussians and improving depth completion. Evaluations on synthetic and real-world transparent object benchmarks, as well as robot grasping tasks, show that TranSplat achieves accurate and dense depth completion, demonstrating its effectiveness in practical applications.



Our method is divided into two stages:

- Surface Embedding colorization through Latent Diffusion Model: Basically extracting surface embedding from transparent objects (i.e. colorizing transparent surfaces) 
- Joint Gaussian optimization: We use both surface embedding and input RGB images as part of the Gaussian optimization process in Guassian splatting pipeline. Using surface embedding mitigates the opacity values from collapsing to zero on transparent surfaces.
  

# Results
### Qualitative Results 

<details>
  <summary>Real world Transparent objects: Transpose Dataset</summary>
  
<div align="center">
  
![put real transpose image here](./assets/real_TRansPose.png)
</div>

</details>

<details>
  <summary>Synthetic Transparent objects with known object priors: Synthetic Transpose Dataset</summary>
  
<div align="center">
  
![put synthetic transpose here](./assets/syn_TRansPose.png)

</div>

</details>


<details>
  <summary> Synthetic Transparent objects with Unknown priors: Clearpose Dataset </summary>
  
<div align="center">
  
![put clearpose here](./assets/syn_ClearPose.png)


</div>

</details>


## Video demonstration


[![Video Label](http://img.youtube.com/vi/O_atdUlaF4I/maxresdefault.jpg)](https://youtu.be/O_atdUlaF4I)

Youtube Link: https://www.youtube.com/watch?v=O_atdUlaF4I

## Setup

TranSplat is tested with Python 3.11. 

### Dockerhub

```sh
docker pull jungyun0609/transplat:1.0

docker run --rm  --gpus all -it --env="DISPLAY" --net=host --ipc=host \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /:/mydata --volume /dev/:/dev/ --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all -v /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0:/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0:ro -v /usr/lib/x86_64-linux-gnu/libEGL.so.1:/usr/lib/x86_64-linux-gnu/libEGL.so.1:ro --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --name {docker name ex: transplat} jungyun0609/transplat:2.0

docker attach {docker name}
```

### Cloning the Repository

```sh
git clone https://github.com/jeongyun0609/TranSplat.git
```

### Download dataset, checkpoints, HF models

```sh
# in the docker container
pip install gdown pyrender
pip install PyOpenGL==3.1.4
bash Download_ckpt.sh
cd stage_0/SurfEmb/data/bop/TRansPose
bash Download_TRansPose_models.sh
cd sequences
bash Download_TRansPose.sh
bash Download_syn.sh
cd ../../../../../../stage_1/models
bash download_hf_models.bash
cd ../../
```

The structure of the downloaded contents should be
```text
TranSplat
├── stage_0/SurfEmb/data
│   ├── bop
│   │   ├── TRansPose
│   │   │   ├── models
│   │   │   │   ├── obj_*/
│   │   │   │   ├── models_info.json
│   │   │   │   ├── obj_key.json
│   │   │   │   ├── object_label.json
│   │   │   │   └── obj_*.ply
│   │   │   ├── sequences
│   │   │   │   ├── train
│   │   │   │   │   └── seq_*/
│   │   │   │   ├── test
│   │   │   │   │   └── seq_test_*/
│   │   │   │   └── Download_TRansPose.sh
│   │   │   └── etc (.sh, .json)
│   │   └── ClearPose_syn
│   │       ├── syn_test_*/
│   │       └── etc (.json)
│   └── model
│       └── TRansPose-SurfEmb.ckpt
├── stage_1/models
│   ├── control_v11f1e_sd15_tile.pth
│   ├── SurfEmb_sd.ckpt
│   ├── v1-5-pruned.ckpt
│   └── etc (.yaml, .bash, .py)
└── stage_2
```

### TRansPose Dataset Preprocessing

TRansPose Link : https://sites.google.com/view/transpose-dataset/download

```sh
# add scene_*.json in every folder of stage_0/SurfEmb/data/bop/TransPose/(train,test)
# if error occurs, change os.environ['PYOPENGL_PLATFORM'] = 'osmesa' to os.environ['PYOPENGL_PLATFORM'] = 'egl'
cd stage_0/SurfEmb
python convert_transpose_to_bop.py --is_train
python convert_transpose_to_bop.py

# train SurfEmb model
# the output checkpoint is already downloaded (stage_0/SurfEmb/data/model/TRansPose-SurfEmb.ckpt)
# if you want to train by yourself, use the command below
# and change the output ckpt name to TRansPose-SurfEmb.ckpt
# python -m surfemb.scripts.train TRansPose --gpus {GPU to use, ex: 0 1}

# make surfemb images
python -m surfemb.scripts.save_emb data/model/TRansPose-SurfEmb.ckpt
cd ../../
```

The structure of the preprocessing should be
```text
TranSplat
├── stage_0/SurfEmb/data
│   └── bop/TRansPose_surfemb
│       ├── train
│       │   └── seq_*/
│       └── etc (.json)
├── stage_1
└── stage_2
```

### Test ControlNet
```sh
cd stage_1
# test with seen object and make json
# will make generate SurfEmb directory and transforms_(train,val,test).json in TRansPose/test/seq_test_{args.index}/cam_R
CUDA_VISIBLE_DEVICES={GPU to use ex 0,2} python3 test_TRansPose.py --dataset TRansPose --index {seq number, ex 3 if syn_test_03} --batch 4 --ckpt SurfEmb_sd.ckpt
python3 json_make_with_bg.py --index {seq number, ex 3 if syn_test_03} --dataset TRansPose

# test with unseen object and make json
# will make generate SurfEmb directory and transforms_(train,val,test).json in ClearPose_syn/seq_test_{args.index}/cam_R
CUDA_VISIBLE_DEVICES={GPU to use ex 0,2} python3 test_ClearPose.py --index 1 --batch 4 --ckpt SurfEmb_sd.ckpt
python3 json_make_with_bg.py --index 1 --dataset ClearPose_syn
cd ../
```

## Gaussian Splatting

```sh
cd stage_2

# Train Gaussian Splatting
CUDA_VISIBLE_DEVICES={GPU to use ex 0,2} python3 train.py -s ../stage_0/SurfEmb/data/bop/TRansPose/test/seq_test_01/cam_R/ -m ./output/{name} --eval

# Render Gaussian Splatting
CUDA_VISIBLE_DEVICES={GPU to use ex 0,2} python3 render.py -m ./output/{name} --eval
```


