import argparse
import os
from single_proc_fed import main as federate
from models.brats_2dunet_tensorflow.model_wrapper import get_model
from models.brats_2dunet_tensorflow.brats_handler import BratsHandler


def main(inst_nums, 
        percent_train, 
        shuffle, 
        agg_id, 
        fed_id, 
        **kwargs):
    # see end of script for a list of currently used kwargs

    # full kwargs is passed as a single object (ie not as kwargs) 
    # to single_proc_fed.py in order that it
    # gets recorded in the resulting dataframe results
    full_kwargs = locals()

    # load institutional datasets
    inst_data_handlers = []
    for inst_num in inst_nums:
        dataset_path = '/raid/datasets/BraTS17/by_institution_NIfTY/{}'.format(inst_num) 
        inst_data_handlers.append(BratsHandler(dataset_path, 
                                               percent_train=percent_train, 
                                               shuffle=shuffle, 
                                               **kwargs))
    # TODO: bring this in via command line?
    init_model_fpath = '/raid/output/tfedlrn/simulation/model_protos/brats_2dunet_tensorflow' 
    latest_model_fpath = '/raid/output/tfedlrn/simulation/model_protos/brats_2dunet_tensorflow'
    best_model_fpath = '/raid/output/tfedlrn/simulation/model_protos/brats_2dunet_tensorflow'                                 

    federate(model_constructor=get_model, 
             col_data_handlers=inst_data_handlers,
             agg_id = agg_id,
             fed_id = fed_id,  
             full_kwargs=full_kwargs,
             init_model_fpath = init_model_fpath,
             latest_model_fpath = latest_model_fpath,
             best_model_fpath = best_model_fpath, 
             **kwargs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument('--inst_nums', '-isnb', type=str, default='0123456789')
    parser.add_argument('--percent_train', '-pt', type=float, default=0.8)
    parser.add_argument('--shuffle', '-sh', type=bool, default=True)
    
    # federation args
    parser.add_argument('--fed_id', '-fid', type=str, default='0')
    parser.add_argument('--agg_id', '-aid', type=str, default='0')
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
