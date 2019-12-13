import argparse
import os
import numpy as np
from pytorch2dunet import PyTorch2DUNet, load_BraTS17_institution
from pytorchflutils import pt_create_loader


def choose_middle(X, y, p):
    p = 0.6
    n = int(p * 155)
    a = (155 - n) // 2
    X = X.reshape(-1, 1, 155, 128, 128)[:, :, a:(a+n)].reshape(-1, 1, 128, 128)
    y = y.reshape(-1, 1, 155, 128, 128)[:, :, a:(a+n)].reshape(-1, 1, 128, 128)
    return X, y

def main(epochs=1, data_dir_base='/raid/datasets/BraTS17/by_institution', optimizer='Adam', channels_first=True, batch_size=64, shuffle=True, device='cuda'):
    X_trains = []
    y_trains = []
    X_vals = []
    y_vals = []
    for i in range(10):
        data_dir = os.path.join(data_dir_base, str(i))
        X_train, y_train, X_val, y_val = load_BraTS17_institution(data_dir=data_dir, channels_first=channels_first)
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_vals.append(X_val)
        y_vals.append(y_val)
    X_train = np.concatenate(X_trains)
    y_train = np.concatenate(y_trains)
    X_val = np.concatenate(X_vals)
    y_val = np.concatenate(y_vals)

    n_initial_epochs = 4
    # include only the middle p% of data for the first n_initial_epochs epochs
    X_train_initial, y_train_initial = choose_middle(X_train, y_train, 0.6)

    train_loader_initial = pt_create_loader(X_train_initial, y_train_initial, batch_size=batch_size, shuffle=shuffle)
    train_loader_final = pt_create_loader(X_train, y_train, batch_size=batch_size, shuffle=shuffle)
    val_loader = pt_create_loader(X_val, y_val, batch_size=batch_size, shuffle=shuffle)
    for i in range(32):
        model = PyTorch2DUNet(train_loader=train_loader_initial, val_loader=val_loader, optimizer=optimizer, device=device)
        for e in range(epochs):
            if e == n_initial_epochs:
                model.train_loader = train_loader_final
            loss = model.train_epoch()
            print('loss for epoch', e, 'is', loss)
            score = model.validate()
            print('score for epoch', e, 'is', score)
        del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', '-opt', type=str, default='Adam')
    parser.add_argument('--device', '-d', type=str, default='cuda')
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--epochs', '-e', type=int, default=1)
    parser.add_argument('--shuffle', '-s', type=bool, default=True)
    parser.add_argument('--channels_first', '-cf', type=bool, default=True)
    parser.add_argument('--data_dir_base', '-ddb', type=str, default='/raid/datasets/BraTS17/by_institution')
    args = parser.parse_args()
    main(**vars(args))
