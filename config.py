import argparse
import os


def make_argparser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_root', type=str, default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'data'))
    arg_parser.add_argument('--save_root', type=str, default=os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'checkpoints'))
    arg_parser.add_argument('--init_author_path', type=str, default='author_init.bin')
    arg_parser.add_argument('--batch_size', type=int, default=1<<14)
    arg_parser.add_argument('--epochs', type=int, default=100)
    arg_parser.add_argument('--train_frac', type=float, default=0.9)
    arg_parser.add_argument('--dims', type=int, nargs='+', default=[256, 128, 64, 32])
    arg_parser.add_argument('--add_write', action='store_true')
    arg_parser.add_argument('--enhance', action='store_true')
    arg_parser.add_argument('--from_dir', type=str, default='9')
    arg_parser.add_argument('--enhance_frac', type=float, default=2)
    arg_parser.add_argument('--load_pair', action='store_true')

    return arg_parser.parse_args()
