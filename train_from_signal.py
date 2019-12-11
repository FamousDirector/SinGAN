from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
from signal_utils import get_channel_array


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='emg_data/old/')
    parser.add_argument('--input_name', help='input file name', required=True)
    parser.add_argument('--row', help='row number of file', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    parser.add_argument('--num_of_channels', help='number of channels', type=int, default=5)
    parser.add_argument('--samp_freq', help='number of channels', type=int, default=1024)
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

        opt.nc_im = opt.num_of_channels
        opt.nc_z = opt.num_of_channels
        x = get_channel_array(opt)

        real = functions.np2torch(x, opt)

        functions.adjust_scales2image(real, opt)
        train_from_signal(opt, Gs, Zs, reals, NoiseAmp, real)

