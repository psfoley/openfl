import argparse
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file


def main(args):
    origin_folder = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    path = get_file(
        'mnist.npz',
        origin=origin_folder + 'mnist.npz',
        file_hash=
        '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1')
    with np.load(path) as f:
        x_train = f['x_train'][args.train_start_index:args.train_end_index]
        y_train = f['y_train'][args.train_start_index:args.train_end_index]

        x_test = f['x_test'][args.val_start_index:args.val_end_index]
        y_test = f['y_test'][args.val_start_index:args.val_end_index]
        np.savez(args.output_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--is_iid', action='store_false')
    parser.add_argument('--train_start_index', '-ts', type=int, default=0)
    parser.add_argument('--train_end_index', '-te', type=int, default=60000)
    parser.add_argument('--val_start_index', '-vs', type=int, default=0)
    parser.add_argument('--val_end_index', '-ve', type=int, default=10000)
    parser.add_argument('--output_path', '-o', type=str, required=True)
    args = parser.parse_args()
    
    main(args)

    