import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
import os
import scipy.stats as stats

from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import sigmaclip
from scipy.spatial import distance as Distance

from skimage.util import img_as_ubyte
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from symfit import parameters, variables, sin, cos, Fit

def main():
    if len(sys.argv)==1:
        print("Need to specify file to process as a parameter")
        print("     Exiting     ")
        exit()
    file = sys.argv[1]

    os.system("mkdir result")

    fname, img, ratio = img_read(file)
    x_cordi_list, y_cordi_list, xy_cordi_list = getOrthoCordinate(img)
    x_center, y_center, center = getCenter(x_cordi_list, y_cordi_list)
    theta_list, radius_list = getPolarCordinate(xy_cordi_list, center)
    new_theta_list, new_radius_list = sigmaClipping(theta_list, radius_list)
    xdata, ydata = setRange(new_theta_list, new_radius_list)

    pred_model = findOptmodel(xdata, ydata)
    final_xy_cordi_list = predCordinate(pred_model,x_center,y_center)
    true_dna_len = dna_length(final_xy_cordi_list,ratio)
    showNsave(img,fname,final_xy_cordi_list,true_dna_len)



def fourier_series(x, f, n=0):
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

def cal_vec_len(vec):
    vec_len = np.sqrt(np.dot(vec,vec))
    return vec_len

def cicle_fit(xdata, ydata,no):
    x, y = variables('x, y')
    w, = parameters('w')
    model_dict = {y: fourier_series(x, f=w, n=no)}
    print(model_dict)
    fit = Fit(model_dict, x=xdata, y=ydata)
    fit_result = fit.execute()

    corr = stats.pearsonr(ydata, fit.model(x=xdata, **fit_result.params).y)
    print(corr[0])
    return corr[0]

def img_read(file):
    base_dir, file_name = os.path.split(file)
    fname, ext = os.path.splitext(file_name)
    ratio = int(fname.split('_')[1])/int(fname.split('_')[2])
    img = Image.open(file).convert('L')
    img_in = img_as_ubyte(img)    
    img = np.reshape(img_in, [img_in.shape[0], img_in.shape[1]])
    return fname, img, ratio

def getOrthoCordinate(image): #unet_image
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)

    x_cordi_list = []
    y_cordi_list = []
    xy_cordi_list = []
    for region in regionprops(labels):
        y0, x0 =region.centroid
        x_cordi_list.append(x0)
        y_cordi_list.append(y0)
        xy_cordi_list.append([x0,y0])
    return x_cordi_list, y_cordi_list, xy_cordi_list

def getCenter(x_cordi_list,y_cordi_list):
    x_center = sum(x_cordi_list)/len(x_cordi_list)
    y_center = sum(y_cordi_list)/len(y_cordi_list)

    center = np.array([x_center,y_center])
    return x_center, y_center, center

def getPolarCordinate(xy_cordi_list,center):
    theta_list = []
    radius_list = []
    for xy in xy_cordi_list:
        vec_xy = np.array(xy)-center
        radius = cal_vec_len(vec_xy)
        theta = np.arccos(np.dot(np.array([1,0]),vec_xy)/(cal_vec_len(vec_xy)*cal_vec_len(np.array([0,1]))))
        if vec_xy[1] < 0:
            theta = math.pi*2 - theta
        else:
            theta = theta
        theta_list.append(theta)
        radius_list.append(radius)
    return theta_list, radius_list

def sigmaClipping(theta_list,radius_list):
    fact = 2.5
    c, low, upp = sigmaclip(radius_list, fact, fact)

    new_theta_list = []
    new_radius_list = []
    radius_dict = {}
    for i in range(len(radius_list)):
        if low < radius_list[i] < upp:
            new_theta_list.append(theta_list[i])
            radius_dict[theta_list[i]] = radius_list[i]
    new_theta_list.sort()
    for theta in new_theta_list:
        radius = radius_dict[theta]
        new_radius_list.append(radius)
    return new_theta_list, new_radius_list

def setRange(new_theta_list,new_radius_list):
    pad_len = int(len(new_radius_list)/2)
    xdata = np.concatenate(((np.array(new_theta_list)-2*math.pi)[-pad_len:]
                                ,new_theta_list,(np.array(new_theta_list)+2*math.pi)[:pad_len]),axis=0)
    ydata = np.concatenate((new_radius_list[-pad_len:],new_radius_list,
                            new_radius_list[:pad_len]),axis=0)
    return xdata, ydata

def findOptmodel(xdata, ydata):
    corr_list = []
    no_list = []
    for no in range(10,20):
        corr = cicle_fit(xdata,ydata,no+1)
        corr_list.append(corr)
        no_list.append(no)
        
    opt_no = no_list[np.argmax(np.array(corr_list))]+1
    print('opt_no',opt_no)
    x, y = variables('x, y')
    w, = parameters('w')
    model_dict = {y: fourier_series(x, f=w, n=opt_no)}
    print(model_dict)
    fit1 = Fit(model_dict, x=xdata, y=ydata)
    return fit1

def predCordinate(fit1,x_center,y_center):
    theta_range = np.arange(0,2*math.pi, 0.01)
    final_xy_cordi_list = []
    fit_result = fit1.execute()
    for i in range(len(theta_range)):
        r = fit1.model(x=theta_range, **fit_result.params).y[i]
        theta = theta_range[i]
        x_poly = x_center+r*math.sin(theta)
        y_poly = y_center+r*math.cos(theta)
        final_xy_cordi_list.append([x_poly,y_poly])
    return final_xy_cordi_list

def dna_length(final_xy_cordi_list,ratio):
    DNA_len = 0
    for i in range(len(final_xy_cordi_list)-1):
        a = np.array(final_xy_cordi_list[i])
        b = np.array(final_xy_cordi_list[i+1])
        node_len = Distance.euclidean(a, b)
        DNA_len += node_len
    true_dna_len = DNA_len * ratio
    true_dna_len = round(true_dna_len,2)
    return true_dna_len

def showNsave(image,fname,final_xy_cordi_list,true_dna_len):
    plt.imshow(image)
    for x, y in final_xy_cordi_list:
        plt.plot(x,y,'bo')
    plt.title(str(true_dna_len)+' nm')
    plt.savefig('_result/'+fname+'.png', dpi=300,bbox_inches='tight')
    plt.show()





if __name__ == "__main__":
    main()
