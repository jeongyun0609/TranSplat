## TranSplat: Surface Embedding-guided 3D Gaussian Splatting for Transparent Object Manipulation

Official Repository for "TranSplat: Surface Embedding-guided 3D Gaussian Splatting for Transparent Object Manipulation", Underreview


[insert img here]

TLDR: We propose a new Gaussian Splatting-based depth completion framework specifically for transparent objects based on Surface Embedding features.

### Abstract

Transparent object manipulation remains a significant challenge in robotics due to the difficulty of acquiring accurate and dense depth measurements. Conventional depth sensors often fail with transparent objects, resulting in incomplete or erroneous depth data. Existing depth completion methods struggle with interframe consistency and incorrectly model transparent objects as Lambertian surfaces, leading to poor depth reconstruction. To address these challenges, we propose TranSplat, a surface embedding-guided 3D Gaussian Splatting method tailored for transparent objects. TranSplat uses a latent diffusion model to generate surface embeddings that provide consistent and continuous representations, making it robust to changes in viewpoint and lighting. By integrating these surface embeddings with input RGB images, TranSplat effectively captures the complexities of transparent surfaces, enhancing the splatting of 3D Gaussians and improving depth completion. Evaluations on synthetic and real-world transparent object benchmarks, as well as robot grasping tasks, show that TranSplat achieves accurate and dense depth completion, demonstrating its effectiveness in practical applications.

### Overview of the Thermal Cameleon Network

<div align="center">
  
[put method picture here]

</div>

Our method is divided into two stages:

- Surface Embedding colorization through Latent Diffusion Model: Basically extracting surface embedding from transparent objects (i.e. colorizing transparent surfaces) 
- Joint Gaussian optimization: We use both surface embedding and input RGB images as part of the Gaussian optimization process in Guassian splatting pipeline. Using surface embedding mitigates the opacity values from collapsing to zero on transparent surfaces.
  

# Results
### Qualitative Results on real transparent objects

<details>
  <summary>Real Transpose Dataset</summary>
  
<div align="center">
  
[put real transpose image here]
</div>

</details>

### Qualitative Results on synthetic transparent objects

<details>
  <summary>Synthetic Transpose Dataset</summary>
  
<div align="center">
  
[put synthetic transpose here]

</div>

</details>


### Qualitative Results on unseen synthetic transparent objects


<details>
  <summary> Clearpose Dataset </summary>
  
<div align="center">
  
[put clearpose here]


</div>

</details>


## Video demonstration


[![Video Label](http://img.youtube.com/vi/DKPlcUHIcTM/maxresdefault.jpg)]([https://youtu.be/DKPlcUHIcTM](https://youtu.be/DKPlcUHIcTM)?t=0s)

Youtube Link: https://www.youtube.com/watch?v=DKPlcUHIcTM

## Usage

### Will be announced after review period

