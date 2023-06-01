
import matplotlib.pyplot as plt
import sys
from tifffile import imread

if __name__ == '__main__':
    sys.path.append('/data/GAN_project/CARE/CSBDeep')
    from csbdeep.io import load_training_data
    from csbdeep.utils import axes_dict, plot_history, plot_some
    from csbdeep.models import Config, UpsamplingCARE
    from csbdeep.data import RawData, create_patches

    output_folder = '/data/GAN_project/CARE/simulated_LR/train_data/shareloc_4_small/1000'
    output_path = output_folder + '/train_data.npz'
    raw_data = RawData.from_folder (
        basepath    = '/data/GAN_project/CARE/simulated_LR/train_data/shareloc_4_small/1000', #path with
        source_dirs = ['low'],
        target_dir  = 'high',
        axes        = 'YX',
    )
    #train_npz_path = '/data/GAN_project/CARE/simulated_LR/train_data/train_data.npz'
    #X, Y, XY_axes = create_patches(raw_data, patch_size=(128, 128), n_patches_per_image=4,save_file = output_path)
    (X, Y), (X_val, Y_val), axes = load_training_data(output_path, validation_split=0.1, verbose=True)
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
    config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=400, train_batch_size=10)
    model = UpsamplingCARE(config, 'my_model', basedir='models')
    history = model.train(X, Y, validation_data=(X_val, Y_val))
    plt.figure(figsize=(16, 5))
    plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])

    x = imread('/data/GAN_project/CARE/Synthetic_tubulin_gfp/test_data/input_n_avg_10_all.tif')
    upsampled = model.predict(x, )
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
