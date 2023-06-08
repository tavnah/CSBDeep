
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import torch
import sys
from tifffile import imread
import cv2
import matplotlib.pyplot as plt

def compare_imgs_MS_SSIM(img1, img2):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    return ms_ssim(img1, img2)

if __name__ == '__main__':
    sys.path.append('/data/GAN_project/CARE/CSBDeep')
    from csbdeep.models import Config, UpsamplingCARE

    care_data_model = UpsamplingCARE(config=None, name='model_care', basedir='models')
    our_data_model = UpsamplingCARE(config=None, name='model_0806', basedir='models')
    imgs_folder = '/data/GAN_project/test_imgs/shareloc_MT3D_160530_C1C2_758K/1'
    widefield_img_path = imgs_folder + '/widefield.tif'
    HR_img_path = imgs_folder + '/data.tiff'

    widefield_img = imread(widefield_img_path)
    HR_img = imread(HR_img_path)

    care_data_predicted = care_data_model.predict(widefield_img, axes='YX', factor=1)
    our_data_predicted = our_data_model.predict(widefield_img, axes='YX', factor=1)

    care_data_score = compare_imgs_MS_SSIM(care_data_predicted, HR_img)
    our_data_score = compare_imgs_MS_SSIM(our_data_predicted, HR_img)
