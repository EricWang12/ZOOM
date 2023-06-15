# ZOOM: Zero-shot-model-diagnosis (CVPR 2023)

[paper](https://arxiv.org/abs/2303.15441) | [project website](http://zero-shot-model-diagnosis.github.io/)

<p align="center">
<img src=img/teaser_figure.png />
</p>



<!-- 
## References
1. [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
2. [CLIP](https://github.com/openai/CLIP.git)
3. [StyleCLIP](https://github.com/orpatashnik/StyleCLIP)
4. [StyleCLIP-pytorch](https://github.com/soushirou/StyleCLIP-pytorch) -->

> **Zero-shot Model Diagnosis**<br>
> Jinqi Luo*, Zhaoning Wang*, Chen Henry Wu,  Dong Huang,  Fernando De La Torre<br>
> *Equal contribution <br>
> https://arxiv.org/abs/2303.15441 <br>
>
>**Abstract:** When it comes to deploying deep vision models, the behavior of these systems must be explicable to ensure confidence in their reliability and fairness. A common approach to evaluate deep learning models is to build a labeled test set with attributes of interest and assess how well it performs. However, creating a balanced test set (i.e., one that is uniformly sampled over all the important traits) is often time-consuming, expensive, and prone to mistakes. The question we try to address is: can we evaluate the sensitivity of deep learning models to arbitrary visual attributes without an annotated test set?
>
>This paper argues the case that Zero-shot Model Diagnosis (ZOOM) is possible without the need for a test set nor labeling. To avoid the need for test sets, our system relies on a generative model and CLIP. The key idea is enabling the user to select a set of prompts (relevant to the problem) and our system will automatically search for semantic counterfactual images (i.e., synthesized images that flip the prediction in the case of a binary classifier) using the generative model. We evaluate several visual tasks (classification, key-point detection, and segmentation) in multiple visual domains to demonstrate the viability of our methodology. Extensive experiments demonstrate that our method is capable of producing counterfactual images and offering sensitivity analysis for model diagnosis without the need for a test set.



## Installation
```
conda env create -f environment.yml
conda activate zoom
```

## Pretrained weights

Please refer to [StyleCLIP-pytorch](https://github.com/soushirou/StyleCLIP-pytorch) for pre-trained weights and  or training your own weights.


save the pretrained weights in `pretrained/` directory. it would look like something like this:

```
├── pretrained
│ ├── utils
│ │ ├── shape_predictor_68_face_landmarks.dat
│ │ ├── imagenet_templates.npy
│ │ ├── e4e_ffhq_encode.pt
│ │ └── model_ir_se50.pth
│ ├── victim_models
│ │ ├── resnet50_Young_trainfull
│ │ └── ...
│ ├── ffhq.pkl
│ └── ...
```
### Extract W, S, S_mean, S_std

You can also following [StyleCLIP-pytorch](https://github.com/soushirou/StyleCLIP-pytorch) for the extractions. We provide some pre-extracted [Here](https://drive.google.com/drive/folders/1Uv1w6k4C5Gz1Jrcjv7eMt4wz_XFJf0at?usp=drive_link).

Save the fs3 in the  `tensor/` directory with experiment name. it would look like something like this:

```
├── tensor\
│ ├── fs3ffhq.npy
│ ├── fs3afhqcat.npy
│ └── ...
```


### Pretrained victim classifiers

For binary victim classifiers, we mainly used the resnet50 from torchvision. We provide some of our binary face classifiers on Celeba in [Here](https://drive.google.com/drive/folders/1Uv1w6k4C5Gz1Jrcjv7eMt4wz_XFJf0at?usp=drive_link)


## run ZOOM

```
bash run_zoom.sh
```
The hyperparameters can be adjusted in the file

### Adjust diagnosis attributes

You can adjust the attributes in [class_labels.py](./class_labels.py)
## Citation

If you use this code for your research, please consider cite our paper:

```
@InProceedings{Luo_2023_CVPR,
    author    = {Luo, Jinqi and Wang, Zhaoning and Wu, Chen Henry and Huang, Dong and De la Torre, Fernando},
    title     = {Zero-shot Model Diagnosis},
    booktitle = {CVPR},
    year      = {2023},
}
```