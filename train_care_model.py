
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    sys.path.append('/data/GAN_project/CARE/CSBDeep')
    from csbdeep.io import load_training_data
    from csbdeep.utils import axes_dict, plot_history
    from csbdeep.models import Config, UpsamplingCARE
    output_folder = '/data/GAN_project/CARE/simulated_LR'
    output_path = output_folder + '/train_data.npz'
    (X, Y), (X_val, Y_val), axes = load_training_data(output_path, validation_split=0.1, verbose=True)
    c = axes_dict(axes)['C']
    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
    config = Config(axes, n_channel_in, n_channel_out, train_steps_per_epoch=25, train_batch_size=4)
    model = UpsamplingCARE(config, 'my_model', basedir='models')
    history = model.train(X, Y, validation_data=(X_val, Y_val))
    plt.figure(figsize=(16, 5))
    plot_history(history, ['loss', 'val_loss'], ['mse', 'val_mse', 'mae', 'val_mae'])