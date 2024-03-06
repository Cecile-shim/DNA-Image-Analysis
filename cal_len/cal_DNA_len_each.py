import sys
import os
import glob
import re
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import measure
from fil_finder import FilFinder2D
import astropy.units as u
import seaborn as sns
from scipy.stats import norm
from skimage.segmentation import clear_border


def main():
    if len(sys.argv) < 3:
        print("Need to specify img type and number to process as a parameter")
        print("     Exiting     ")
        exit()
    img_type = sys.argv[1]
    img_no = sys.argv[2]

    os.system("mkdir %s_result"%(img_type))

    pcv.params.debug = "plot"
    ratio = set_ratio(img_type,img_no)
    mask = imgProcess(img_type,img_no)
    regions = getSkelRegion(mask)
    total_dna_len_list = getDNAlen(regions,ratio)
    save_dist(total_dna_len_list,img_type,img_no)

def cal_node_len(skeleton):
    img1, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=50)
    labeled_img  = pcv.morphology.segment_path_length(segmented_img=seg_img, 
                                                      objects=edge_objects, label="default")
    node_len_list = pcv.outputs.observations['default']['segment_path_length']['value']
    node_len_list = [x for x in node_len_list if x > 0]
    return node_len_list 

def save_dist(len_list, img_type,img_no):
    dna_num = len(len_list)
    ax = sns.distplot(len_list, kde=True)
    ax = sns.distplot(len_list, kde=False, fit=norm)
    avg = round(sum(len_list)/len(len_list),2)
    max_len = round(np.max(len_list),2)
    plt.title('Total DNA : '+str(format(dna_num,','))+'\n'+
        'avg : '+str(avg)+' (μm)'+'\n'+
        'max_len : '+str(max_len)+' (μm)')
    plt.xlabel('Length')
    plt.tight_layout()
    plt.savefig('%s_result/%s_dist.png'%(img_type,img_no))
    plt.show()

def set_ratio(img_type,img_no):
    ratio = 1
    file_list = glob.glob('dataset/%s/ori_imgs/*.png'%(img_type))
    for file in file_list:
        base_dir, file_name = os.path.split(file)
        file_nums = re.findall('\d+',file_name)
        if file_nums[0] == img_no:
            ratio = int(file_nums[2]) / int(file_nums[1])
    return ratio

def imgProcess(img_type,img_no):
    img_dir = 'dataset/%s/unet_imgs/%s.png'%(img_type,img_no)
    img, path, filename = pcv.readimage(filename=img_dir, mode='gray')
    mask = pcv.gaussian_blur(img=img, ksize=(51,51), sigma_x=0, sigma_y=None)
    mask = clear_border(mask)
    return mask

def getSkelRegion(mask):
    skeleton = pcv.morphology.skeletonize(mask=mask)
    labels = measure.label(skeleton)
    labels = (labels * 255).astype(np.uint8)
    nlabels, markers, stats, centroids = cv2.connectedComponentsWithStats(labels, None, None, None, 8, cv2.CV_32S)
    min_area_threshold = 81
    filtered_labels = np.copy(markers)

    for label in range(1,nlabels):
        area = stats[label,cv2.CC_STAT_AREA]
        if area < min_area_threshold:
            filtered_labels[filtered_labels ==label] = 0

    regions = measure.regionprops(filtered_labels, intensity_image=skeleton)
    return regions

def getDNAlen(regions,ratio):
    total_node_len_list = []
    for i in range(len(regions)):
        skeleton = np.array(regions[i].image, dtype='uint8')
        node_len_list = cal_node_len(skeleton)
        for node in node_len_list:
            total_node_len_list.append(node*ratio)
    return total_node_len_list


if __name__ == "__main__":
    main()

