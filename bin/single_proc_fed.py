# Copyright (C) 2020 Intel Corporation
# Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.

from enum import Enum
import os

from tfedlrn import get_object
from tfedlrn.aggregator.aggregator import Aggregator
from tfedlrn.collaborator import Collaborator


def get_collaborators(model, aggregator, col_ids, **kwargs):
    collaborators = {} 
    for col_id in col_ids:
        collaborators[col_id] = Collaborator(col_id=col_id, 
                                             wrapped_model=model, 
                                             channel=aggregator, 
                                             **kwargs)
    return collaborators  
    

def federate(col_config,
             agg_config,
             col_data, 
             model_config, 
             fed_config, 
             init_model_fpath, 
             latest_model_fpath, 
             best_model_fpath, 
             **kwargs):

    # get the number of collaborators and opt_treatment
    col_ids = fed_config['col_ids']
    opt_treatment = fed_config['opt_treatment']
    
    # instantiate the model (using the first collaborator dataset for now)
    model = get_object(data=col_data[col_ids[0]], **model_config)

    if opt_treatment == 'EDGE':
        # EDGE mode requires that we save the optimizer state for each collaborator at the
        # end of round training, and restore the state at the beginning of the next round.
        # (when a collaborator has its own process, this state is simply held in memory,
        #  but here we have the collaborators share a model object)

        # initialize last_round_states (including opt states) with initial model state.
        # TODO: (Simplify) Effect of using last_round_states is not easy to read from code. Though the model weights
        # below are written into the model during EDGE mode, they are always overwritten before the model is used, 
        # so that only the optimizer parts are effectively influenced.
        last_round_states = {col_id: model.get_tensor_dict(with_opt_vars=True) for col_id in col_ids}

    # create the aggregator
    aggregator = Aggregator(init_model_fpath=init_model_fpath, 
                            latest_model_fpath=latest_model_fpath, 
                            best_model_fpath=best_model_fpath, 
                            **agg_config)

    # create the collaborataors
    collaborators = get_collaborators(model=model, 
                                      aggregator=aggregator, 
                                      col_ids=col_ids,
                                      **col_config)

    rounds_to_train = fed_config['rounds_to_train']



    # TODO: Enable flat score detection, minimum accept, etc.
    for _ in range(rounds_to_train):
        for col_id in col_ids:

            collaborator = collaborators[col_id]

            # overwrite the model's data using current insitution
            model.set_data(col_data[col_id])

            
            if opt_treatment == 'EDGE':
                # restore the end of last round state for this collaborator
                # (The collaborator will overwrite the model weight piece of this before
                #  the model is used in all cases, and overwrites the optimizer part
                #  in all cases except opt_treatment='EDGE')
                model.set_tensor_dict(last_round_states[col_id], with_opt_vars=True)

            # run the collaborator jobs for this round
            collaborator.run_to_yield_or_quit()

            if opt_treatment == 'EDGE':
                last_round_states[col_id] = model.get_tensor_dict(with_opt_vars=True)

       

                