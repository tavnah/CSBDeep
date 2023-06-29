
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
from PIL import Image

def compare_imgs_MS_SSIM(img1, img2):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    img1 = normalize(img1)
    img2 = normalize(img2)
    img1_int = (img1 * 255).astype(np.uint8)
    img2_int = (img2 * 255).astype(np.uint8)
    return msssim(img1_int, img2_int)

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def generate_recon_imgs(widefield_path, HR_path, model1, model2, resize_factor):

    widefield_img = imread(widefield_path)
    widefield_img_normd = normalize(widefield_img)
    widefield_img_normd_f = widefield_img_normd.astype(np.float32)
    HR_img = imread(HR_path)
    hr_new_size = [int(HR_img.shape[0] // resize_factor), int(HR_img.shape[1] // resize_factor)]
    widefield_new_size = [int(widefield_img_normd_f.shape[0] // resize_factor),
                          int(widefield_img_normd_f.shape[1] // resize_factor)]

    hr_resized = cv2.resize(HR_img, hr_new_size)
    widefield_img_normd_f_resized = cv2.resize(widefield_img_normd_f, widefield_new_size)

    data1_predicted = model1.predict(widefield_img_normd_f_resized, axes='YX', factor=4)
    data2_predicted = model2.predict(widefield_img_normd_f_resized, axes='YX', factor=4)

    data1_normed = normalize(data1_predicted)
    data2_normed = normalize(data2_predicted)

    return data1_normed, data2_normed, hr_resized

def create_2_overlaps(data1_normed, data2_normed, hr_resized):
    hr_res_m = hr_resized - np.mean(hr_resized)
    data1_nrmd_m = data1_normed - np.mean(data1_normed)
    correlation = scipy.signal.fftconvolve(hr_res_m, data1_nrmd_m[::-1, ::-1], mode='same')

    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift_y, shift_x = hr_resized.shape[0] // 2 - y, hr_resized.shape[1] // 2 - x
    shifted_hr = shift(hr_resized, (shift_y, shift_x), cval=0)
    shifted_hr[shifted_hr < 0] = 0
    data1_normed[data1_normed < 0] = 0
    shifted_hr_g = shifted_hr  # exposure.adjust_gamma(shifted_hr, 0.5)
    data1_normed_g = data1_normed  # exposure.adjust_gamma(data1_normed, 0.5)

    model1_3ch = np.zeros((hr_resized.shape[0], hr_resized.shape[1], 3))
    model2_3ch = np.zeros((hr_resized.shape[0], hr_resized.shape[1], 3))
    model1_3ch[:, :, 0] = data1_normed_g
    model1_3ch[:, :, 1] = shifted_hr_g
    model2_3ch[:, :, 0] = data2_normed
    model2_3ch[:, :, 1] = shifted_hr_g

    return model1_3ch, model2_3ch


def generate_and_compare_imgs_from_models(model1, model2, widefield_path, HR_path, labels, resize_factor):

    data1_normed, data2_normed, hr_resized = generate_recon_imgs(widefield_path, HR_path, model1, model2, resize_factor)

    model1_3ch, model2_3ch = create_2_overlaps(data1_normed, data2_normed, hr_resized)

    f1 = plt.figure(1)
    plt.imshow(model1_3ch)
    plt.title(labels[0])
    plt.show()
    f2 = plt.figure(2)
    plt.imshow(model2_3ch)
    plt.title(labels[1])
    plt.show()

    return model1_3ch, model2_3ch

def create_overlap_for_widefield_imgs(folder_path, imgs_num, output_folder, care_data_model,our_data_model, resize_factor):
    # compare CARE data and our data
    for i in range(1,imgs_num+1):
        widefield_img_path = folder_path + '/'+ str(i)+ '/widefield.tif'
        HR_img_path = folder_path + '/' + str(i) + '/data.tiff'
        care_3ch, our_3ch = generate_and_compare_imgs_from_models(care_data_model, our_data_model, widefield_img_path, HR_img_path, labels = ['CARE data', 'Our data'], resize_factor=resize_factor)
        output_path = output_folder + '/' + str(i)
        #check if output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        np.savez(output_path, care = care_3ch, ours = our_3ch)

def compare_MSE(img, gt):
    mse1 = np.mean((img - gt) ** 2)
    return mse1

if __name__ == '__main__':
    # Ours_FRC = [8.279, 7.105, 4.57, 3.579, 4.053, 9.670]
    # CARE_FRC = [8.093, 6.918, 4.534, 4.578, 3.799, 7.022]
    # diff_FRC = [Ours_FRC[i] - CARE_FRC[i] for i in range(len(Ours_FRC))]
    # mean_diff = np.mean(diff_FRC)
    sys.path.append('/data/GAN_project/CARE/CSBDeep')
    from csbdeep.models import Config, UpsamplingCARE
    create_overlap = False
    compare = False
    save_as_tif = True
    if save_as_tif:
        care_data_model = UpsamplingCARE(config=None, name='model_care',
                                         basedir='models')  # 'model_care', 'model_care_1000', 'model_care_1000_2'
        our_data_model = UpsamplingCARE(config=None, name='model_0806',
                                        basedir='models')  # 'model_0806', 'model_ours_1000', 'model_ours_1000_2'
        widefield_path ='/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/7/widefield.tif'
        HR_path = '/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/7/data.tiff'
        resize_factor = 1
        care_normed, ours_normed, hr_resized = generate_recon_imgs(widefield_path, HR_path, care_data_model,
                                                                     our_data_model, resize_factor)
        print('hey')

    if create_overlap:
        care_data_model = UpsamplingCARE(config=None, name='model_care_1000_2', basedir='models') #'model_care', 'model_care_1000', 'model_care_1000_2'
        our_data_model = UpsamplingCARE(config=None, name='model_ours_1000_2', basedir='models') #'model_0806', 'model_ours_1000', 'model_ours_1000_2'
        main_folder = '/data/GAN_project/test_imgs/'
        folder1 = 'shareloc_MT3D_160530_C1C2_758K/'
        folder2 = 'shareloc2'
        folder1_path = main_folder + folder1
        folder2_path = main_folder + folder2
        output_folder1 = folder1_path + '/output_orig_1000'
        output_folder2 = folder2_path + '/output_orig'

        create_overlap_for_widefield_imgs(folder1_path, 7, output_folder1, care_data_model, our_data_model,
                                          resize_factor=2)
        create_overlap_for_widefield_imgs(folder2_path, 7, output_folder2, care_data_model, our_data_model,
                                          resize_factor=1.35)
    if compare:
        img1_path = '/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/output_orig_1000/7_ours.tiff'
        img2_path = '/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/output_orig_1000/7_care.tiff'
        gt_path = '/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/output_orig_1000/7_gt.tiff'
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        gt = Image.open(gt_path)
        img1_np = np.asarray(img1)
        img2_np = np.asarray(img2)
        gt_np = np.asarray(gt)
        mse_ours = compare_MSE(img1_np, gt_np)
        mse_care = compare_MSE(img2_np, gt_np)
        mssim_ours = compare_imgs_MS_SSIM(img1_np, gt_np)
        mssim_care = compare_imgs_MS_SSIM(img2_np, gt_np)
        print('MSE for ours: ', mse_ours)
        print('MSE for CARE: ', mse_care)
        print('MS-SSIM for ours: ', mssim_ours)
        print('MS-SSIM for CARE: ', mssim_care)
        print('h')

    # for i in range(1,8):
    #     widefield_img_path = folder1_path + '/'+ str(i)+ '/widefield.tif'
    #     HR_img_path = folder1_path + '/' + str(i) + '/data.tiff'
    #     care_3ch, our_3ch = generate_and_compare_imgs_from_models(care_data_model, our_data_model, widefield_img_path, HR_img_path, labels = ['CARE data', 'Our data'], resize_factor=2)
    #     output_path = output_folder1 + '/' + str(i)
    #     #check if output folder exists
    #     if not os.path.exists(output_folder1):
    #         os.makedirs(output_folder1)
    #     np.savez(output_path, care = care_3ch, ours = our_3ch)

    # for i in range(1,8):
    #     widefield_img_path = folder1_path + '/' + str(i) + '/widefield.tif'
    #     HR_img_path = folder1_path + '/' + str(i) + '/data.tiff'
    #     care_3ch, our_3ch = generate_and_compare_imgs_from_models(care_data_model, our_data_model, widefield_img_path,
    #                                                               HR_img_path, labels=['CARE data', 'Our data'], resize_factor=1.325)
    #     output_path = output_folder2 + '/' + str(i)
    #     if not os.path.exists(output_folder2):
    #         os.makedirs(output_folder2)
    #     np.savez(output_path, care=care_3ch, ours=our_3ch)

    #widefield_img_path = imgs_folder + '/widefield.tif'
    #HR_img_path = imgs_folder + '/data.tiff'
    #care_3ch, our_3ch = generate_and_compare_imgs_from_models(care_data_model, our_data_model, widefield_img_path, HR_img_path, labels = ['CARE data', 'Our data'])
    print('hey')
