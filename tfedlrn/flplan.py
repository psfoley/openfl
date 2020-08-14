# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import os
import logging
import hashlib
import yaml

from tfedlrn import load_yaml, get_object
from tfedlrn.localconfig import get_data_path_from_local_config
from tfedlrn.tensor_transformation_pipelines import NoCompressionPipeline
from tfedlrn.aggregator.aggregator import Aggregator
from tfedlrn.collaborator.collaborator import Collaborator
from tfedlrn.comms.grpc.aggregatorgrpcserver import AggregatorGRPCServer
from tfedlrn.comms.grpc.collaboratorgrpcclient import CollaboratorGRPCClient


def parse_fl_plan(plan_path, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    flplan = load_yaml(plan_path)

    # ensure 'init_kwargs' appears in each top-level block
    for k in flplan.keys():
        if 'init_kwargs' not in flplan[k]:
            flplan[k]['init_kwargs'] = {}

    # collect all the plan filepaths used
    plan_files = [plan_path]

    # walk the top level keys for defaults_file in sorted order
    for k in sorted(flplan.keys()):
        defaults_file = flplan[k].get('defaults_file')
        if defaults_file is not None:
            defaults_file = os.path.join(os.path.dirname(plan_path), defaults_file)
            logger.info("Using FLPlan defaults for section '{}' from file '{}'".format(k, defaults_file))
            defaults = load_yaml(defaults_file)
            if 'init_kwargs' in defaults:
                defaults['init_kwargs'].update(flplan[k]['init_kwargs'])
                flplan[k]['init_kwargs'] = defaults['init_kwargs']
            defaults.update(flplan[k])
            flplan[k] = defaults
            plan_files.append(defaults_file)

    # create the hash of these files
    flplan_fname = os.path.splitext(os.path.basename(plan_path))[0]
    flplan_hash = hash_files(plan_files, logger=logger)

    federation_uuid = '{}_{}'.format(flplan_fname, flplan_hash[:8])
    aggregator_uuid = 'aggregator_{}'.format(federation_uuid)

    flplan['aggregator_object_init']['init_kwargs']['aggregator_uuid'] = aggregator_uuid
    flplan['aggregator_object_init']['init_kwargs']['federation_uuid'] = federation_uuid
    flplan['collaborator_object_init']['init_kwargs']['aggregator_uuid'] = aggregator_uuid
    flplan['collaborator_object_init']['init_kwargs']['federation_uuid'] = federation_uuid
    flplan['hash'] = flplan_hash

    logger.info("Parsed plan:\n{}".format(yaml.dump(flplan)))

    return flplan


def init_object(flplan_block, **kwargs):
    if 'class_to_init' not in flplan_block:
        raise ValueError("FLPLAN ERROR")

    init_kwargs = flplan_block.get('init_kwargs', {})
    init_kwargs.update(**kwargs)

    class_to_init   = flplan_block['class_to_init']
    class_name      = class_to_init.split('.')[-1]
    module_name     = '.'.join(class_to_init.split('.')[:-1])

    return get_object(module_name, class_name, **init_kwargs)


def create_compression_pipeline(flplan):
    if flplan.get('compression_pipeline_object_init') is not None:
        compression_pipeline = init_object(**flplan.get('compression_pipeline'))
    else:
        compression_pipeline = NoCompressionPipeline()
    return compression_pipeline


def create_model_object(flplan, data_object):
    return init_object(flplan['model_object_init'], data=data_object)


def resolve_autoport(flplan):
    config = flplan['network_object_init']
    flplan_hash_8 = flplan['hash'][:8]
 
    # check for auto_port convenience settings
    if config.get('auto_port',False) == True or config.get('agg_port') == 'auto':
        # replace the port number with something in the range of min-max
        # default is 49152 to 60999
        port_range = config.get('auto_port_range', (49152, 60999))
        port = (int(flplan_hash_8, 16) % (port_range[1] - port_range[0])) + port_range[0]
        config['init_kwargs']['agg_port'] = port


def create_aggregator_server_from_flplan(agg, flplan):
    # FIXME: this is currently only the GRPC server which takes no init kwargs!
    return AggregatorGRPCServer(agg)


def get_serve_kwargs_from_flpan(flplan, base_dir):
    config = flplan['network_object_init']

    if len(config['init_kwargs']) == 0:
      config['init_kwargs'] = flplan['network_object_init']

    resolve_autoport(flplan)

    # find the cert to use
    cert_dir         = os.path.join(base_dir, config.get('cert_folder', 'pki')) # default to 'pki
    cert_common_name = config['init_kwargs']['agg_addr']

    cert_chain_path = os.path.join(cert_dir, 'cert_chain.crt')
    certificate     = os.path.join(cert_dir, 'agg_{}'.format(cert_common_name), 'agg_{}.crt'.format(cert_common_name))
    private_key     = os.path.join(cert_dir, 'agg_{}'.format(cert_common_name), 'agg_{}.key'.format(cert_common_name))

    cert_common_name = config['init_kwargs']['agg_addr']

    serve_kwargs = config['init_kwargs']

    # patch in kwargs for certificates
    serve_kwargs['ca']          = cert_chain_path
    serve_kwargs['certificate'] = certificate
    serve_kwargs['private_key'] = private_key
    
    return serve_kwargs


def create_aggregator_object_from_flplan(flplan, collaborator_common_names, single_col_cert_common_name, weights_dir):
    init_kwargs = flplan['aggregator_object_init']['init_kwargs']
    task_assigner_config = flplan['task_assigner']
    tasks = flplan['tasks']
    task_assigner = get_object(**task_assigner_config,
                               tasks=tasks,
                               collaborator_list=collaborator_common_names,
                               rounds=init_kwargs['rounds_to_train'])


    # patch in the collaborators file and single_col_cert_common_name
    init_kwargs['collaborator_common_names']    = collaborator_common_names
    init_kwargs['single_col_cert_common_name']  = single_col_cert_common_name
    init_kwargs['task_assigner'] = task_assigner

    # path in the full model filepaths
    model_prefixes = ['init', 'latest', 'best']
    for p in model_prefixes:
        init_kwargs['{}_model_fpath'.format(p)] = os.path.join(weights_dir, init_kwargs['{}_model_fname'.format(p)])

    compression_pipeline = create_compression_pipeline(flplan)

    return Aggregator(compression_pipeline=compression_pipeline,
                      **init_kwargs)


def create_collaborator_network_object(flplan, collaborator_common_name, single_col_cert_common_name, base_dir):
    config = flplan['network_object_init']

    if len(config['init_kwargs']) == 0:
      config['init_kwargs'] = flplan['network_object_init']

    resolve_autoport(flplan)

    # find the cert to use
    cert_dir = os.path.join(base_dir, config.get('cert_folder', 'pki')) # default to 'pki

    # if a single cert common name is in use, then that is the certificate we must use
    if single_col_cert_common_name is None:
        cert_common_name = collaborator_common_name
    else:
        cert_common_name = single_col_cert_common_name

    cert_chain_path = os.path.join(cert_dir, 'cert_chain.crt')
    certificate     = os.path.join(cert_dir, 'col_{}'.format(cert_common_name), 'col_{}.crt'.format(cert_common_name))
    private_key     = os.path.join(cert_dir, 'col_{}'.format(cert_common_name), 'col_{}.key'.format(cert_common_name))

    # FIXME: support network objects other than GRPC
    return CollaboratorGRPCClient(ca=cert_chain_path,
                                  certificate=certificate,
                                  private_key=private_key,
                                  **config['init_kwargs'])


def create_collaborator_object_from_flplan(flplan, 
                                           collaborator_common_name, 
                                           local_config,
                                           base_dir,
                                           single_col_cert_common_name=None,
                                           data_object=None,
                                           model_object=None,
                                           compression_pipeline=None,
                                           network_object=None):
    if data_object is None:
        data_object = create_data_object(flplan, collaborator_common_name, local_config)

    if model_object is None:
        model_object = create_model_object(flplan, data_object)
    
    if compression_pipeline is None:
        compression_pipeline = create_compression_pipeline(flplan)

    if network_object is None:
        network_object = create_collaborator_network_object(flplan, collaborator_common_name, single_col_cert_common_name, base_dir)

    tasks_config = flplan['tasks']

    return Collaborator(collaborator_name=collaborator_common_name,
                        model=model_object, 
                        aggregator=network_object,
                        compression_pipeline=compression_pipeline,
                        tasks_config=tasks_config,
                        single_col_cert_common_name=single_col_cert_common_name,  
                        **flplan['collaborator_object_init']['init_kwargs'])


def create_data_object(flplan, collaborator_common_name, local_config):
    data_path = get_data_path_from_local_config(local_config,
                                                collaborator_common_name,
                                                flplan['data_object_init']['data_name_in_local_config'])
    return init_object(flplan['data_object_init'], data_path=data_path)


def hash_files(paths, logger=None):
    md5 = hashlib.md5()
    for p in paths:
        with open(p, 'rb') as f:
            md5.update(f.read())
        if logger is not None:
            logger.info("After hashing {}, hash is {}".format(p, md5.hexdigest()))
    return md5.hexdigest()
