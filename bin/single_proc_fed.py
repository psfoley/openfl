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

def get_collaborators(model, aggregator, col_ids, **kwargs):
    collaborators = {} 
    for col_id in col_ids:
        collaborators[col_id] = Collaborator(id=col_id, 
                                             wrapped_model=model, 
                                             channel=aggregator, 
                                             **kwargs)
    return collaborators  
    

def federate(get_model_func,
             col_config,
             agg_config,
             col_data, 
             model_config, 
             fed_config, 
             weights_dir, 
             init_model_fpath, 
             latest_model_fpath, 
             best_model_fpath, 
             **kwargs):

    # get the number of collaborators
    col_ids = col_config['col_ids']
    num_cols = len(col_ids)
    
    # instantiate the model (using the first collaborator dataset for now)
    model = get_model_func(data= col_data[col_ids[0]],
                           model_kwargs=model_config['params'])

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
        for col_id in col_ids:

            collaborator = collaborators[col_id]

            # overwrite the model's data using current insitution
            model.set_data(col_data[col_id])

            # run the collaborator jobs for this round
            collaborator.run_to_yield_or_quit()

       

                