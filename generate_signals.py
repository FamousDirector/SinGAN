import os
import shutil

from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
from signal_utils import get_multi_channel_spectral_array, reconstruct_signals, plot_signals


if __name__ == '__main__':
    parser = get_arguments()
    # data params
    parser.add_argument('--input_dir', help='input image dir', default='emg_data/old/')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--row', help='row number of file', type=int, required=True)

    # signal params
    parser.add_argument('--num_samples', type=int, help='number of samples to generate', default=10)
    parser.add_argument('--num_channel_samples', type=int, help='number of samples per channel', default=200)
    parser.add_argument('--num_of_channels', help='number of channels', type=int, default=5)
    parser.add_argument('--samp_freq', help='number of channels', type=int, default=1024)
    parser.add_argument('--spectral_type', help='number of channels', default='stft')

    # SinGAN parameters
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    parser.add_argument('--mode', help='set generation mode', default='random_samples')
    parser.add_argument('--animation_alpha', type=float, help='random walk first moment', default=0.8)
    parser.add_argument('--animation_beta', type=float, help='random walk second moment', default=0.05)

    # generation parameters
    parser.add_argument('--keep_npz', action='store_true')
    parser.add_argument('--plot_signals', action='store_true')

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
        else:
            print("output already exists")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass

        opt.nc_im = opt.num_of_channels
        opt.nc_z = opt.num_of_channels
        x = get_multi_channel_spectral_array(opt)

        real = functions.np2torch(x, opt)

        if opt.mode == 'random_samples':
            functions.adjust_scales2image(real, opt)
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
            SinGAN_generate_signal(Gs, Zs, reals, NoiseAmp, opt,
                                   start_scale=opt.gen_start_scale,
                                   num_samples=opt.num_samples)
        elif opt.mode == 'animation':
            opt.min_size = 20
            functions.adjust_scales2image(real, opt)
            Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt, mode_='animation_train')
            SinGAN_generate_signal(Gs, Zs, reals, NoiseAmp, opt,
                                   start_scale=opt.gen_start_scale,
                                   num_samples=opt.num_samples,
                                   alpha=opt.animation_alpha,
                                   beta=opt.animation_beta
                                   )

        # reconstruct and save signals
        reconstruct_signals(opt, dir2save)

        # remove npz files
        if not opt.keep_npz:
            for filename in os.listdir(dir2save):
                if filename.endswith(".npz"):
                    file_path = os.path.join(dir2save, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_path, e))

        if opt.plot_signals:
            plot_signals(opt, dir2save)






