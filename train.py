from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
from signal_utils import get_multi_channel_spectral_array


if __name__ == '__main__':
    parser = get_arguments()

    # data params
    parser.add_argument('--input_dir', help='input image dir', default='data/')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--row', help='row number of file', type=int, required=True)

    # signal params
    parser.add_argument('--data_col_start_index',
                        help='number of columns in dataset that are not part of the emg signal', type=int, default=3)
    parser.add_argument('--num_channel_samples', type=int, help='number of samples per channel', default=200)
    parser.add_argument('--num_of_channels', help='number of channels', type=int, default=8)
    parser.add_argument('--channel_skip_count', help='fraction of channels to use', type=int, default=1)
    parser.add_argument('--samp_freq', help='number of channels', type=int, default=1000)
    parser.add_argument('--spectral_type', help='number of channels', default='stft')

    # SinGAN parameters
    parser.add_argument('--mode', help='set generation mode', default='train')

    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        opt.nc_im = opt.num_of_channels//opt.channel_skip_count
        opt.nc_z = opt.num_of_channels//opt.channel_skip_count
        x = get_multi_channel_spectral_array(opt)

        real = functions.np2torch(x, opt)

        functions.adjust_scales2image(real, opt)
        if opt.mode == 'animation_train':
            opt.min_size = 20
            train_from_signal(opt, Gs, Zs, reals, NoiseAmp, real)
        else:  # opt.mode == train
            train_from_signal(opt, Gs, Zs, reals, NoiseAmp, real)

