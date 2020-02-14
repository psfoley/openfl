from pickagpu import pick_a_gpu
pick_a_gpu()

import numpy as np
import tensorflow as tf
from datetime import datetime
import pickle
import uuid
from enum import Enum
import os

from tfedlrn.aggregator.aggregator import Aggregator
from tfedlrn.collaborator import Collaborator

# determines whether at each institution at the beggining of each round, 
# the first and second moments for the adam optimizer are:
# (RESET) reset to initial values
# (AGG)  set to an aggregation of last round final values from all institutions, or
# (EDGE) kept as the last round final values from that particular institution

class OptMode(Enum):
    RESET = 1
    AGG = 2
    EDGE = 3

def aggregate_tensordicts(td0, td1, n0, n1):
    return {key: np.average(np.array([td0[key], td1[key]]), axis=0, weights=[n0, n1]) \
            for key in td0}


def parse_tensor_dict(tensor_dict, tensor_dict_opt_keys):
    layer_dict = tensor_dict.copy()
    opt_dict = {key: layer_dict.pop(key) for key in tensor_dict_opt_keys}
    return layer_dict, opt_dict

def get_col_data(get_data_func, col_ids, base_data_path,  **kwargs):
    col_data_paths = [os.path.join(base_data_path, col_id) for col_id in col_ids]
    col_data = []
    for path in col_data_paths:
        col_data.append(get_data_func(data_path = path, 
                                      **kwargs))
    return col_data

def get_collaborators(model, aggregator, col_ids, **kwargs):
    collaborators = [] 
    for col_id in col_ids:
        collaborators.append(Collaborator(id=col_id, 
                                          wrapped_model=model, 
                                          channel=aggregator, 
                                          **kwargs))
    return collaborators  
    

def federate(get_model_func,
             get_data_func, 
             col_config,
             agg_config,
             data_config, 
             model_config, 
             fed_config, 
             weights_dir, 
             base_data_path, 
             init_model_fpath, 
             latest_model_fpath, 
             best_model_fpath, 
             **kwargs):

    # get the number of collaborators
    col_ids = col_config['col_ids']
    num_cols = len(col_ids) 

    col_data = get_col_data(get_data_func=get_data_func, 
                            col_ids=col_ids,
                            base_data_path=base_data_path, 
                            **data_config)  
    
    # instantiate the model (using the first collaborator dataset for now)
    model = get_model_func(data= col_data[0],
                           **model_config)

    # create the aggregator
    aggregator = Aggregator(init_model_fpath = init_model_fpath,
                            latest_model_fpath = latest_model_fpath, 
                            best_model_fpath = best_model_fpath, 
                            **agg_config)

    # create the collaborataors
    collaborators = get_collaborators(model=model, 
                                       aggregator=aggregator, 
                                       **col_config)

    rounds = fed_config['rounds']
    
    # TODO: Enable flat score detection, minimum accept, etc.
    for r in range(rounds):
        print()
        print('Training Round {}'.format(r))
        print()
        for col_num in range(num_cols):

            collaborator = collaborators[col_num]

            # overwrite the model's data using current insitution
            model.set_data(col_data[col_num])

            # run the collaborator jobs for this round
            collaborator.run_to_yield_or_quit()

       

                