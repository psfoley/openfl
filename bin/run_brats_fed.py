# TODO: Do this instead from a docker container (including install of tensorflow-gpu)
# (should only be supported through opt-in)
from pickagpu import pick_a_gpu
pick_a_gpu()

from tf_utils import create_tf_session
sess = create_tf_session()

import argparse
from single_proc_fed import main as federate
from models.brats_2dunet_tensorflow.model_wrapper import get_model
from models.brats_2dunet_tensorflow.datasets import load_from_NIfTY


def main(**kwargs):
    # see end of script for a list of currently used kwargs

    # full kwargs is passed as a single object (ie not as kwargs) 
    # to single_proc_fed.py in order that it
    # gets recorded in the resulting dataframe results
    full_kwargs = kwargs.copy()

    # some of the kwargs are used to create the model and
    # datasets, the rest are passed to single_proc_fed.py
    inst_nums = [int(char)for char in kwargs.pop('inst_nums')]
    percent_train = kwargs.pop('percent_train')
    shuffle = kwargs.pop('shuffle')
    
    # we separate kwargs going to the model constructor
    model_kwargs = {}
    model_kwargs['batch_size'] = kwargs.pop('batch_size')
    model_kwargs['optimizer_type'] = kwargs.pop('optimizer_type')
    model_kwargs['learning_rate'] = kwargs.pop('learning_rate')
    model_kwargs['momentum'] = kwargs.pop('momentum')
    model_kwargs['loss_func'] = kwargs.pop('loss_func')
    model_kwargs['smooth'] = kwargs.pop('smooth')

    # the remaining kwargs go on to single_proc_fed.py
    fl_kwargs = kwargs
    


    # load institutional datasets
    inst_datasets = []
    for inst_num in inst_nums:
        dataset_path = '/raid/datasets/BraTS17/by_institution_NIfTY/{}'.format(inst_num) 
        inst_datasets.append(load_from_NIfTY(dataset_path, 
                                            percent_train=percent_train, 
                                            shuffle=shuffle))

    # TODO: Is it bad practice for me to plan overwriting of the model datset 
    # attributes inside single_proc_fed.py?
    # TODO: currently the model args above do not get incorported


    unet = get_model(dataset_path=None, sess=sess, **model_kwargs)
    federate(sess=sess, 
             model=unet, 
             inst_datasets=inst_datasets, 
             full_kwargs=full_kwargs, 
             **fl_kwargs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--optimizer_type', '-opt', type=str, choices=['Adam', 'RMSProp', 'Momentum'], default='Adam')
    parser.add_argument('--learning_rate', '-l', type=float, default=4e-5)
    parser.add_argument('--momentum', '-m', type=float, default=0.9)
    parser.add_argument('--loss_func', '-loss', type=str, choices=['mse', 'dice'], default='dice')
    parser.add_argument('--smooth', '-sm', type=float, default=32.0)
    parser.add_argument('--inst_nums', '-isnb', type=str, default='0123456789')
    parser.add_argument('--percent_train', '-pt', type=float, default=0.8)
    parser.add_argument('--shuffle', '-sh', type=bool, default=True)
    
    # federation args
    parser.add_argument('--epochs', '-e', type=int, default=1)
    parser.add_argument('--rounds', '-r', type=int, default=20)
    parser.add_argument('--seed', '-s', type=int, default=77)
    parser.add_argument('--abort_patience', '-p', type=int, default=5) 
    parser.add_argument('--iterations', '-i', type=int, default=3)
    parser.add_argument('--opt_mode', '-o', type=str, choices=['AGG', 'EDGE', 'RESET'], default='EDGE')
    parser.add_argument('--minimum_accept', '-ma', type=float, default=0.6)
    parser.add_argument('--training_dice_test', '-tst', type=bool, default=False)
    
    args = parser.parse_args()
    main(**vars(args))
