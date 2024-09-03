# DNA-Image-Analysis


Code to calculate length of DNA strand from each image


## Introduction
The advent of artificial intelligence (AI) has greatly enhanced the statistical analysis of DNA molecules imaged from various microscopy platforms, aiding our understanding of biochemical and molecular phenomena. Despite this progress, AI-driven analyses of DNA molecules in electron microscopy (EM) images remain unexplored, with most research focusing on fluorescence microscopy (FM) and atomic force microscopy (AFM). Additionally, AI's application in classifying DNA based on structural or conformational differences has been limited, compared to its widespread use for sequence-based classifications. This study addresses these gaps by introducing a pipeline that leverages AI to analyze DNA molecules in EM images, regardless of their size, shape, or conformation, extracting key dimensional features using custom algorithms.


## Citation
Artificial Intelligence Enhanced Analysis of Genomic DNA Visualized with Nanoparticle-Tagged Peptides under Electron Microscopy (Research Article, No. smll.202405065)


## Requirement
+ Python 3.10.9
+ TensorFlow 2


## Installation
```
git clone https://github.com/cecileshim/dna-image-analysis.git
cd dna-image-analysis
```

## Usage
The dataset used in both cal_len.py and cal_radius.py has been segmented from the original images through the U-Net training process. For instructions on how to use unet.ipynb, refer to this link : <https://github.com/CatchZeng/tensorflow-unet-labelme>


### 1. Calculate length of the linear DNA
The overall directory tree structure is as follows.
```
cal_len
├── cal_DNA_len.py
├── cal_DNA_len_each.py
└── dataset
    ├── 1
    │   ├── ori_imgs
    │   └── unet_imgs
    └── 2
        ├── ori_imgs
        └── unet_imgs
```
0. If you are using a custom dataset, assign a specific number to each type of DNA (e.g., original dataset: fish sperm (long) - 1, fish sperm (short) - 2). Create a folder named ori_imgs for the original images and a folder named unet_imgs for the segmented images, with each folder named according to the assigned number. If there are various types of DNA, the assigned numbers can exceed 2. Additionally, the filenames for the original and segmented images should follow the format outlined below (pixel/um ratio).
```
    │   ├── ori_imgs
    │   │   └── 1. 948p 5um.png
    │   └── unet_imgs
    │       └── 1.png
```
1. Run `python cal_DNA_len.py num`. The 'num' should be same with the dna directory number you want to know.
2. Run `python cal_DNA_len_each.py num filenum` for analyzing each image. You should put the file number in 'filenum'.
3. ```
   ├── 1_result
   │   ├── 1_dist.png
   │   └── total_dist.png
   ├── 2_result
   │   ├── 1_dist.png
   │   └── total_dist.png
   ```
   You can get the output like this. 'total_dist' is for cal_DNA_len.py, and 'num_dist' is for cal_DNA_len_each.py


### 2. Calculate length of the circular DNA
The overall directory tree structure is as follows.
```
cal_radius
├── cal_radius.py
├── dataset
└── result
```
0. If you are using a custom dataset, place the U-Net trained segmented images into the dataset folder. Additionally, ensure that the filenames reveal information in the format filenum_pix_um, such as 1_200_474.jpg.
1. Run `python cal_radius.py num`. The 'num' should be same with 'filenum'
2. You can get the output in the 'result' directory.


---


For more information, please refer to our paper.
