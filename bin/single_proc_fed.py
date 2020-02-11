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

# FIXME: figure out where results and model proto files should go


def main(model_constructor, 
         col_data_handlers,
         full_kwargs,
         init_model_fpath, 
         latest_model_fpath,
         best_model_fpath,
         agg_id, 
         fed_id, 
         epochs, 
         rounds, 
         seed, 
         abort_patience, 
         iterations, 
         opt_mode, 
         minimum_accept, 
         training_dice_test, 
         **kwargs):

    config = full_kwargs
    print('Running experiment with config {}'.format(config))    
    
    # infer the number of collaborators
    num_cols = len(col_data_handlers)    
    
    # instantiate the model (using the first collaborator dataset for now)
    model = model_constructor(col_data_handlers[0], **kwargs)

    while iterations > 0:
        # FIXME: 
        # note the seed coupling below. The random seed for tf effects the model weight initialization, whereas the 
        # ramdom seed for numpy effects the data set shuffling. If you want to explore these effects independently 
        # be sure to define an independent seed for numpy and track it in the config and results.
        tf.set_random_seed(seed)
        np.random.seed(seed+1)

        # start with a fresh model, initialized using the tf seed
        model.initialize_globals()

        # save the initalized model to proto
        # FIXME: Currently for testing and until a better solution this is
        # written over each itteration

        model.export_init_weights(model_name = type(model).__name__,
                                 version=0, 
                                 fpath=init_model_fpath)

        # create aggregator and all collaborator objects
        # FIXME: all proto paths are the same
        aggregator = Aggregator(id=agg_id, 
                               fed_id=fed_id, 
                               col_ids = [str(n) for n in range(num_cols)], 
                               init_model_fpath=init_model_fpath, 
                               latest_model_fpath=latest_model_fpath, 
                               best_model_fpath=best_model_fpath)

        # create the collaborator objects
        collaborators = [] 
        for col_num in range(num_cols):
            collaborators.append(Collaborator(id=str(col_num), 
                                             agg_id=agg_id, 
                                             fed_id=fed_id, 
                                             wrapped_model=model, 
                                             channel=aggregator, 
                                             model_version=0, 
                                             opt_treatment=opt_mode, 
                                             polling_interval=4))


        # TODO: Enable flat score detection, minimum accept, etc.
        for r in range(rounds):
            print('Training Round {}'.format(r))
            for col_num in range(num_cols):

                collaborator = collaborators[col_num]

                # overwrite the model's data handler using current insitution
                model.set_data_handler(col_data_handlers[col_num])

                # run the collaborator jobs for this round
                collaborator.run_to_yield()

                