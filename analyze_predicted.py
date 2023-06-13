
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
import os

def compare_imgs_MS_SSIM(img1, img2):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    return ms_ssim(img1, img2)

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
def generate_and_compare_imgs_from_models(model1, model2, widefield_path, HR_path, labels, resize_factor):
    widefield_img = imread(widefield_path)
    widefield_img_normd = normalize(widefield_img)
    widefield_img_normd_f = widefield_img_normd.astype(np.float32)
    HR_img = imread(HR_path)
    hr_new_size = [int(HR_img.shape[0]//resize_factor), int(HR_img.shape[1]//resize_factor)]
    widefield_new_size = [int(widefield_img_normd_f.shape[0]//resize_factor), int(widefield_img_normd_f.shape[1]//resize_factor)]

    hr_resized = cv2.resize(HR_img, hr_new_size)
    widefield_img_normd_f_resized = cv2.resize(widefield_img_normd_f, widefield_new_size)

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
    shifted_hr_g = shifted_hr #exposure.adjust_gamma(shifted_hr, 0.5)
    data1_normed_g = data1_normed #exposure.adjust_gamma(data1_normed, 0.5)

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

    return model1_3ch, model2_3ch



if __name__ == '__main__':
    Ours_FRC = [8.279, 7.105, 4.57, 3.579, 4.053, 9.670]
    CARE_FRC = [8.093, 6.918, 4.534, 4.578, 3.799, 7.022]
    diff_FRC = [Ours_FRC[i] - CARE_FRC[i] for i in range(len(Ours_FRC))]
    mean_diff = np.mean(diff_FRC)
    sys.path.append('/data/GAN_project/CARE/CSBDeep')
    from csbdeep.models import Config, UpsamplingCARE

    care_data_model = UpsamplingCARE(config=None, name='model_care', basedir='models') #'model_care', 'model_care_1000', 'model_care_1000_2'
    our_data_model = UpsamplingCARE(config=None, name='model_0806', basedir='models') #'model_0806', 'model_ours_1000', 'model_ours_1000_2'
    main_folder = '/data/GAN_project/test_imgs/'
    folder1 = 'shareloc_MT3D_160530_C1C2_758K/'
    folder2 = 'shareloc2'
    folder1_path = main_folder + folder1
    folder2_path = main_folder + folder2
    output_folder1 = folder1_path + '/output_orig_1000'
    output_folder2 = folder2_path + '/output_orig'
    # for i in range(1,8):
    #     widefield_img_path = folder1_path + '/'+ str(i)+ '/widefield.tif'
    #     HR_img_path = folder1_path + '/' + str(i) + '/data.tiff'
    #     care_3ch, our_3ch = generate_and_compare_imgs_from_models(care_data_model, our_data_model, widefield_img_path, HR_img_path, labels = ['CARE data', 'Our data'], resize_factor=2)
    #     output_path = output_folder1 + '/' + str(i)
    #     #check if output folder exists
    #     if not os.path.exists(output_folder1):
    #         os.makedirs(output_folder1)
    #     np.savez(output_path, care = care_3ch, ours = our_3ch)

    for i in range(1,6):
        widefield_img_path = folder1_path + '/' + str(i) + '/widefield.tif'
        HR_img_path = folder1_path + '/' + str(i) + '/data.tiff'
        care_3ch, our_3ch = generate_and_compare_imgs_from_models(care_data_model, our_data_model, widefield_img_path,
                                                                  HR_img_path, labels=['CARE data', 'Our data'], resize_factor=1.325)
        output_path = output_folder2 + '/' + str(i)
        if not os.path.exists(output_folder2):
            os.makedirs(output_folder2)
        np.savez(output_path, care=care_3ch, ours=our_3ch)

    #widefield_img_path = imgs_folder + '/widefield.tif'
    #HR_img_path = imgs_folder + '/data.tiff'
    #care_3ch, our_3ch = generate_and_compare_imgs_from_models(care_data_model, our_data_model, widefield_img_path, HR_img_path, labels = ['CARE data', 'Our data'])
    print('hey')
