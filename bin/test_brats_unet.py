#!/usr/bin/env python3
import argparse
import numpy as np
import tensorflow as tf

import tfedlrn
from tfedlrn.bratsunettest import TF2DUnet

# data-sharing test
def main(data_path=None, epochs=12):
    np.random.seed(0)
    tf.set_random_seed(0)

    # FIXME: find the right code pattern for this
    assert data_path is not None

    unet = TF2DUnet(data_path)

    score = unet.validate()
    print('initial validation score:', score)

    for e in range(epochs):
        print('starting epoch', e)
        loss = unet.train_epoch()
        print('epoch', e, 'completed with loss', loss)
        score = unet.validate()
        print('epoch', e, 'validation score:', score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-p', type=str, default='/raid/datasets/BraTS17/by_institution', required=True)
    parser.add_argument('--epochs', '-e', type=int, default=12)
    args = parser.parse_args()
    main(**vars(args))
