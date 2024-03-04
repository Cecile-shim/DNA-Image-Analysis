from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.measure import label, regionprops, regionprops_table

img_file = '2_predict.png'
pcv.params.debug = "plot"
img, path, filename = pcv.readimage(filename=img_file, mode='gray')
blur_img = pcv.gaussian_blur(img=img,ksize=(51,51),sigma_x=0,sigma_y=None)
cross_kernel = pcv.get_kernel(size=(15,15),shape="cross")
filtered_img = pcv.opening(gray_img = blur_img, kernel = cross_kernel)
#skel1 = pcv.morphology.skeletonize(img)
skel2 = pcv.morphology.skeletonize(blur_img)
#skel3 = pcv.morphology.skeletonize(filtered_img)

labels = label(skel2)
labels = (labels * 255).astype(np.uint8)
nlabels, markers, stats, centroids = cv2.connectedComponentsWithStats(labels, None, None, None, 8, cv2.CV_32S)
min_area_threshold = 25
filtered_labels = np.copy(markers)

# print(stats)

for label in range(1,nlabels):
	if stats[label,cv2.CC_STAT_AREA] < min_area_threshold:
		filtered_labels[markers == label] = 0

plt.imshow(filtered_labels)
plt.show()