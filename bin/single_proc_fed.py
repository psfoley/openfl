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


def get_collaborators(model, aggregator, tasks_config, collaborator_common_names, compression_pipeline, **kwargs):
    collaborators = {} 
    for collaborator_common_name in collaborator_common_names:
        collaborators[collaborator_common_name] = Collaborator(collaborator_name=collaborator_common_name, 
                                             model=model, 
                                             aggregator=aggregator,
                                             tasks_config=tasks_config,
                                             compression_pipeline=compression_pipeline, 
                                             **kwargs)
    return collaborators  
    

def federate(data_config, 
             col_config,
             agg_config,
             model_config, 
             compression_config, 
             tasks_config,
             task_assigner_config,
             by_col_data_names_to_paths, 
             init_model_fpath, 
             latest_model_fpath, 
             best_model_fpath, 
             **kwargs):

    # get the number of collaborators
    collaborator_common_names = agg_config['collaborator_common_names']

    # get the BraTS data objects for each collaborator
    col_data = {collaborator_common_name: get_data(by_col_data_names_to_paths[collaborator_common_name], **data_config) for collaborator_common_name in collaborator_common_names}
    
    
    # instantiate the model (using the first collaborator dataset for now)
    model = get_object(data=col_data[collaborator_common_names[0]], **model_config)

    rounds_to_train = agg_config['rounds_to_train']

    task_assigner = get_object(**task_assigner_config,
                               tasks=tasks_config,
                               collaborator_list=collaborator_common_names,
                               rounds=rounds_to_train)
    
    # FL collaborators are statefull. Since this single process script utilizes one
    # shared model for all collaborators, model states need to be tracked.
    model_states = {collaborator_common_name: None for collaborator_common_name in collaborator_common_names}

    if compression_config is not None:
        compression_pipeline = get_compression_pipeline(**compression_config)
    else:
        compression_pipeline = None

    # create the aggregator
    aggregator = Aggregator(initial_model_fpath=init_model_fpath, 
                            latest_model_fpath=latest_model_fpath, 
                            best_model_fpath=best_model_fpath,
                            custom_tensor_dir=None,
                            compression_pipeline=compression_pipeline, 
                            task_assigner=task_assigner,
                            **agg_config)

    # create the collaborataors
    collaborators = get_collaborators(model=model, 
                                      aggregator=aggregator, 
                                      tasks_config=tasks_config,
                                      collaborator_common_names=collaborator_common_names,
                                      compression_pipeline=compression_pipeline,
                                      **col_config)




    # TODO: Enable flat score detection, minimum accept, etc.
    for round in range(rounds_to_train):
        for collaborator_common_name in collaborator_common_names:

            collaborator = collaborators[collaborator_common_name]

            # overwrite the model's data using current insitution
            model.set_data(col_data[collaborator_common_name])

            
            if round != 0:
                # restore model state from when this collaborator last held the model
                model.set_tensor_dict(model_states[collaborator_common_name], with_opt_vars=True)

            # run the collaborator jobs for this round
            collaborator.run_simulation()

            model_states[collaborator_common_name] = model.get_tensor_dict(with_opt_vars=True)

                



       
