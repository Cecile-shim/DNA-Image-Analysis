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
'''python
git clone https://github.com/cecile-shim/dna-image-analysis # fix later
cd dna-image-analysis
'''


## Usage
The dataset used in both cal_len.py and cal_radius.py has been segmented from the original images through the U-Net training process. For instructions on how to use unet.ipynb, refer to this link : <https://github.com/CatchZeng/tensorflow-unet-labelme>


### 1. Calculate length of the linear DNA


### 2. Calculate length of the circular DNA


For more information, please refer to our paper.
