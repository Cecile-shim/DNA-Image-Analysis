from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import cv2
import numpy as np

for img_no in range(1,13):
	img_file = str(img_no)+'_predict.png'
	img, path, filename = pcv.readimage(filename=img_file, mode='gray')
	blur_img = pcv.gaussian_blur(img=img,ksize=(51,51),sigma_x=0,sigma_y=None)
	skel = pcv.morphology.skeletonize(blur_img)
	skel = (skel * 255).astype(np.uint8)
	# plt.imsave(str(img_no)+'_skel.png',skel)
	nlabels, markers, stats, centroids = cv2.connectedComponentsWithStats(skel,None,None,None,8,cv2.CV_32S)
	min_area_threshold = 81
	filtered_labels = np.copy(markers)

	for label in range(1,nlabels):
		area = stats[label,cv2.CC_STAT_AREA]
		if area < min_area_threshold:
			filtered_labels[filtered_labels ==label] = 0

	filtered_labels[filtered_labels != 0] = 255
	plt.imsave(str(img_no)+'_skel.png',filtered_labels,cmap='gray')