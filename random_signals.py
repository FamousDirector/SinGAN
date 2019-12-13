from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions
from signal_utils import get_channel_array


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='emg_data/old/')
    parser.add_argument('--input_name', help='input image name', required=True)
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    parser.add_argument('--num_samples', type=int, help='number of samples to generate', default=10)
    parser.add_argument('--num_of_channels', help='number of channels', type=int, default=5)
    parser.add_argument('--samp_freq', help='number of channels', type=int, default=1024)
    parser.add_argument('--mode', help='set generation mode', default='random_samples')
    parser.add_argument('--row', help='row number of file', type=int, required=True)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    if dir2save is None:
        print('task does not exist')
    elif (os.path.exists(dir2save)):
        if opt.mode == 'random_samples':
            print('random samples for image %s, start scale=%d, already exist' % (opt.input_name, opt.gen_start_scale))
        elif opt.mode == 'random_samples_arbitrary_sizes':
            print('random samples for image %s at size: scale_h=%f, scale_v=%f, already exist' % (opt.input_name, opt.scale_h, opt.scale_v))
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
        Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
        in_s = functions.generate_in2coarsest(reals,1,1,opt)
        SinGAN_generate_signal(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=opt.gen_start_scale,
                        num_samples=opt.num_samples)





