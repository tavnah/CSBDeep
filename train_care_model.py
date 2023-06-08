
import numpy as np
import matplotlib.pyplot as plt
import sys
from tifffile import imread

def check_microtubules_percentage(npz_path, threshold):

    with np.load(npz_path) as patches:
        HR_pathces = patches['Y']
        percentages = []
        for hr_patch in HR_pathces:
            hr_patch = hr_patch[0]
            cur_percentage = np.sum(hr_patch > threshold) / (hr_patch.size)
            percentages.append(cur_percentage)
    return percentages

if __name__ == '__main__':
    #percentages = check_microtubules_percentage('/data/GAN_project/CARE/Synthetic_tubulin_gfp/train_data/data_label.npz', 0.4)

    sys.path.append('/data/GAN_project/CARE/CSBDeep')
    from csbdeep.io import load_training_data
    from csbdeep.utils import axes_dict, plot_history, plot_some
    from csbdeep.models import Config, UpsamplingCARE
    from csbdeep.data import RawData, create_patches
    import os
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    #trained care model: model_care
    create_ptchs = False
    train = True
    print_hist = True
    show_vld = True
    load_model = False

    if create_ptchs:
        output_folder = '/data/GAN_project/CARE/simulated_LR/train_data/shareloc_4_small/1000_no_dense_thresh0.1_higherSNR_lowerPSF'
        output_path = output_folder + '/train_data.npz'
        raw_data = RawData.from_folder (
            basepath    = '/data/GAN_project/CARE/simulated_LR/train_data/shareloc_4_small/1000_no_dense_thresh0.1_higherSNR_lowerPSF', #path with
            source_dirs = ['low'],
            target_dir  = 'high',
            axes        = 'YX',
        )
        #train_npz_path = '/data/GAN_project/CARE/simulated_LR/train_data/train_data.npz'
        X, Y, XY_axes = create_patches(raw_data, patch_size=(128, 128), n_patches_per_image=2,save_file = output_path)
    output_path = '/data/GAN_project/CARE/Synthetic_tubulin_gfp/train_data/data_label.npz'
    if train:
        (X, Y), (X_val, Y_val), axes = load_training_data(output_path, validation_split=0.1, verbose=True ,axes = 'SCYX') ##only for original CARE data
        c = axes_dict(axes)['C']

        n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
        #axes = 'XYC' #for CARE original data
        config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=500, train_batch_size=15)
        model = UpsamplingCARE(config, 'model_care', basedir='models')
        history = model.train(X, Y, validation_data=(X_val, Y_val))
    if print_hist:
        plt.figure(figsize=(16, 5))
        plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])
    if show_vld:
        plt.figure(figsize=(12, 4.5))
        _P = model.keras_model.predict(X_val[:5])
        if config.probabilistic:
            _P = _P[..., :(_P.shape[-1] // 2)]
        plot_some(X_val[:5, ..., 0], Y_val[:5, ..., 0], _P[..., 0], pmax=99.5)
        plt.suptitle('5 example validation patches (ZY slice)\n'
                     'top row: input (source),  '
                     'middle row: target (ground truth),  '
                     'bottom row: predicted from source');
        plt.show()
    if load_model:
        model = UpsamplingCARE(config=None, name='model_care', basedir='models')

    x = imread('/data/GAN_project/CARE/Synthetic_tubulin_gfp/test_data/input_n_avg_10_all.tif')
    import cv2
    a = cv2.resize(x[0], (128, 64))
    upsampled = model.predict(a,'XY', 4 )
    plt.imshow(upsampled)
    plt.show()
