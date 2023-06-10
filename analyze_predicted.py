
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import torch
import sys
from tifffile import imread
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sewar.full_ref import msssim
import scipy.signal
from skimage import exposure
from scipy.ndimage import shift


def compare_imgs_MS_SSIM(img1, img2):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    return ms_ssim(img1, img2)

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
def generate_and_compare_imgs_from_models(model1, model2, widefield_path, HR_path, labels):
    widefield_img = imread(widefield_path)
    widefield_img_normd = normalize(widefield_img)
    widefield_img_normd_f = widefield_img_normd.astype(np.float32)
    HR_img = imread(HR_path)
    hr_resized = cv2.resize(HR_img, [512, 512])
    widefield_img_normd_f_resized = cv2.resize(widefield_img_normd_f, [128, 128])

    data1_predicted = model1.predict(widefield_img_normd_f_resized, axes='YX', factor=4)
    data2_predicted = model2.predict(widefield_img_normd_f_resized, axes='YX', factor=4)

    data1_normed =normalize(data1_predicted)
    data2_normed =normalize(data2_predicted)

    hr_res_m = hr_resized - np.mean(hr_resized)
    data1_nrmd_m = data1_normed - np.mean(data1_normed)
    correlation = scipy.signal.fftconvolve(hr_res_m, data1_nrmd_m[::-1, ::-1], mode='same')

    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift_y, shift_x = hr_resized.shape[0] // 2 - y, hr_resized.shape[1] // 2 - x
    shifted_hr = shift(hr_resized, (shift_y, shift_x), cval=0)
    shifted_hr[shifted_hr<0]=0
    data1_normed[data1_normed<0]=0
    shifted_hr_g = exposure.adjust_gamma(shifted_hr, 0.5)
    data1_normed_g = exposure.adjust_gamma(data1_normed, 0.5)

    model1_3ch = np.zeros((hr_resized.shape[0], hr_resized.shape[1], 3))
    model2_3ch = np.zeros((hr_resized.shape[0], hr_resized.shape[1], 3))
    model1_3ch[:, :, 0] = data1_normed_g
    model1_3ch[:, :, 1] = shifted_hr_g
    model2_3ch[:, :, 0] = data2_normed
    model2_3ch[:, :, 1] = shifted_hr_g

    f1 = plt.figure(1)
    plt.imshow(model1_3ch)
    plt.title(labels[0])
    plt.show()
    f2 = plt.figure(2)
    plt.imshow(model2_3ch)
    plt.title(labels[1])
    plt.show()



if __name__ == '__main__':
    sys.path.append('/data/GAN_project/CARE/CSBDeep')
    from csbdeep.models import Config, UpsamplingCARE

    care_data_model = UpsamplingCARE(config=None, name='model_care', basedir='models')
    our_data_model = UpsamplingCARE(config=None, name='model_0806', basedir='models')
    imgs_folder = '/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/6'
    widefield_img_path = imgs_folder + '/widefield.tif'
    HR_img_path = imgs_folder + '/data.tiff'
    generate_and_compare_imgs_from_models(care_data_model, our_data_model, widefield_img_path, HR_img_path, labels = ['CARE data', 'Our data'])
    print('hey')
