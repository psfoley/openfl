# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from enum import Enum
import os

from tfedlrn import get_object
from tfedlrn.aggregator.aggregator import Aggregator
from tfedlrn.collaborator import Collaborator

from tfedlrn.tensor_transformation_pipelines import get_compression_pipeline


def get_data(data_names_to_paths, data_name, module_name, class_name, **kwargs):
    data_path = data_names_to_paths[data_name]
    return get_object(module_name, class_name, data_path=data_path, **kwargs)


def get_collaborators(model, aggregator, col_ids, compression_pipeline, **kwargs):
    collaborators = {} 
    for col_id in col_ids:
        collaborators[col_id] = Collaborator(col_id=col_id, 
                                             wrapped_model=model, 
                                             channel=aggregator,
                                             compression_pipeline=compression_pipeline, 
                                             **kwargs)
    return collaborators  
    

def federate(data_config, 
             col_config,
             agg_config,
             model_config, 
             compression_config, 
             by_col_data_names_to_paths, 
             init_model_fpath, 
             latest_model_fpath, 
             best_model_fpath, 
             **kwargs):

    # get the number of collaborators
    col_ids = agg_config['col_ids']

    # get the BraTS data objects for each collaborator
    col_data = {col_id: get_data(by_col_data_names_to_paths[col_id], **data_config) for col_id in col_ids}
    
    
    # instantiate the model (using the first collaborator dataset for now)
    model = get_object(data=col_data[col_ids[0]], **model_config)
    
    # FL collaborators are statefull. Since this single process script utilizes one
    # shared model for all collaborators, model states need to be tracked.
    model_states = {col_id: None for col_id in col_ids}

    if compression_config is not None:
        compression_pipeline = get_compression_pipeline(**compression_config)
    else:
        compression_pipeline = None

    # create the aggregator
    aggregator = Aggregator(init_model_fpath=init_model_fpath, 
                            latest_model_fpath=latest_model_fpath, 
                            best_model_fpath=best_model_fpath,
                            compression_pipeline=compression_pipeline, 
                            **agg_config)

    # create the collaborataors
    collaborators = get_collaborators(model=model, 
                                      aggregator=aggregator, 
                                      col_ids=col_ids,
                                      compression_pipeline=compression_pipeline,
                                      **col_config)

    rounds_to_train = agg_config['rounds_to_train']



    # TODO: Enable flat score detection, minimum accept, etc.
    for round in range(rounds_to_train):
        for col_id in col_ids:

            collaborator = collaborators[col_id]

            # overwrite the model's data using current insitution
            model.set_data(col_data[col_id])

            
            if round != 0:
                # restore model state from when this collaborator last held the model
                model.set_tensor_dict(model_states[col_id], with_opt_vars=True)

            # run the collaborator jobs for this round
            collaborator.run_to_yield_or_quit()

            model_states[col_id] = model.get_tensor_dict(with_opt_vars=True)

                



       