# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Collaborator module."""

from logging import getLogger
from enum import Enum
from time import sleep
import inspect
from dill.source import getsource

from openfl.protocols import utils
from openfl.utilities import TensorKey
from openfl.pipelines import TensorCodec, NoCompressionPipeline
from openfl.databases import TensorDB
from openfl.federated.types import TypeHandlerFactory
from openfl.federated.data import FederatedDataLoader



class OptTreatment(Enum):
    """Optimizer Methods."""

    RESET = 1
    """
    RESET tells each collaborator to reset the optimizer state at the beginning
    of each round.
    """
    CONTINUE_LOCAL = 2
    """
    CONTINUE_LOCAL tells each collaborator to continue with the local optimizer
    state from the previous round.
    """
    CONTINUE_GLOBAL = 3
    """
    CONTINUE_GLOBAL tells each collaborator to continue with the federally
    averaged optimizer state from the previous round.
    """


class Collaborator:
    r"""The Collaborator object class.

    Args:
        collaborator_name (string): The common name for the collaborator
        aggregator_uuid: The unique id for the client
        federation_uuid: The unique id for the federation
        model: The model
        opt_treatment* (string): The optimizer state treatment (Defaults to
            "CONTINUE_GLOBAL", which is aggreagated state from previous round.)

        compression_pipeline: The compression pipeline (Defaults to None)

        num_batches_per_round (int): Number of batches per round
                                     (Defaults to None)

        delta_updates* (bool): True = Only model delta gets sent.
                               False = Whole model gets sent to collaborator.
                               Defaults to False.

        single_col_cert_common_name: (Defaults to None)

    Note:
        \* - Plan setting.
    """

    def __init__(self,
                 collaborator_name,
                 aggregator_uuid,
                 federation_uuid,
                 client,
                 task_runner,
                 tensor_pipe,
                 task_config,
                 data_loader_config,
                 opt_treatment=OptTreatment.RESET,
                 delta_updates=False,
                 db_store_rounds=1,
                 **kwargs):
        """Initialize."""
        self.single_col_cert_common_name = None

        if self.single_col_cert_common_name is None:
            self.single_col_cert_common_name = ''  # for protobuf compatibility
        # we would really want this as an object

        self.collaborator_name = collaborator_name
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid

        self.tensor_pipe = tensor_pipe or NoCompressionPipeline()
        self.tensor_codec = TensorCodec(self.tensor_pipe)
        self.tensor_db = TensorDB()
        self.db_store_rounds = db_store_rounds

        self.task_runner = task_runner
        self.delta_updates = delta_updates

        self.client = client

        self.task_config = task_config
        self.data_loader_config = data_loader_config

        self.logger = getLogger(__name__)

        # Determine rank and federation size
        self.rank,self.federation_size = self.get_rank_and_size()
        self.setup_task_runner()

        # RESET/CONTINUE_LOCAL/CONTINUE_GLOBAL
        if hasattr(OptTreatment, opt_treatment):
            self.opt_treatment = OptTreatment[opt_treatment]
        else:
            self.logger.error("Unknown opt_treatment: %s." % opt_treatment)
            raise NotImplementedError(
                "Unknown opt_treatment: %s." % opt_treatment)

    def setup_task_runner(self):
        """
        This function primarily determines whether data is in memory and should be split
        """
        if self.data_loader_config['shard_data']:
          for attr in inspect.getmembers(self.task_runner):
            if isinstance(attr[1],FederatedDataLoader):
                attr[1].shard_data(self.rank,self.federation_size)
                setattr(self.task_runner,attr[0],attr[1])
                self.logger.info(f'Set data shard {self.rank} out of federation size {self.federation_size} for {attr[0]}')


    def run(self):
        """Run the collaborator."""
        while True:
            tasks, round_number, sleep_time, time_to_quit = self.get_tasks()
            if time_to_quit:
                break
            elif sleep_time > 0:
                sleep(sleep_time)  # some sleep function
            else:
                self.logger.info(
                    'Received the following tasks: {}'.format(tasks))
                #Do beginning of round update
                self.update_task_runner(round_number)
                for task in tasks:
                    self.do_task(task, round_number)

                # Cleaning tensor db
                self.tensor_db.clean_up(self.db_store_rounds)

        self.logger.info('End of Federation reached. Exiting...')

    def run_simulation(self):
        """
        Specific function for the simulation.

        After the tasks have
        been performed for a roundquit, and then the collaborator object will
        be reinitialized after the next round
        """
        while True:
            tasks, round_number, sleep_time, time_to_quit = self.get_tasks()
            if time_to_quit:
                self.logger.info('End of Federation reached. Exiting...')
                break
            elif sleep_time > 0:
                sleep(sleep_time)  # some sleep function
            else:
                self.logger.info(
                    'Received the following tasks: {}'.format(tasks))
                for task in tasks:
                    self.do_task(task, round_number)
                self.logger.info(
                    'All tasks completed on {} for round {}...'.format(
                        self.collaborator_name, round_number))
                break

    def get_rank_and_size(self):
        """Determine collaborator's rank and size of federation"""
        rank, federation_size = self.client.get_rank_and_size(self.collaborator_name)
        return rank, federation_size


    def get_tasks(self):
        """Get tasks from the aggregator."""
        # logging wait time to analyze training process
        self.logger.info('Waiting for tasks...')
        tasks, round_number, sleep_time, time_to_quit = self.client.get_tasks(
            self.collaborator_name)

        return tasks, round_number, sleep_time, time_to_quit

    def update_task_runner(self, round_number):
        """Update task runner attributes"""

        self.attr_hashes = {}
        type_handler_factory = TypeHandlerFactory()
        for attr in inspect.getmembers(self.task_runner):
            if '__' not in attr[0] and attr[0][0] is not '_':
                if isinstance(attr[1],FederatedDataLoader):
                    # Set 'hash' to be number of times DataLoader has been accessed
                    # If this count changes, then it was accessed in some way by the executing task
                    self.attr_hashes[attr[0]] = attr[1].get_access_count()
                    continue
                self.logger.debug(f'Public Attribute: {attr[0]}, type: {type(attr[1])}')
                if type_handler_factory.is_supported(attr[1]):
                    type_handler = type_handler_factory.get_type_handler(attr[1])
                    tensorkeys = type_handler.get_tensorkeys(attr[1],attr[0],round_num=round_number)
                    if len(tensorkeys) == 0:
                        continue
                    #TODO transform tensorkeys to global tensorkeys
                    #tensorkeys = self.transform_to_global_tensorkeys(tensorkeys)
                    self.logger.debug(f'Update task runner tensorkeys: {tensorkeys}')
                    input_tensor_dict = self.get_numpy_dict_for_tensorkeys(
                            tensorkeys
                    )
                    self.logger.debug(f'input_tensor_dict: {input_tensor_dict}')
                    new_value = type_handler.map_to_attr(attr[1],input_tensor_dict)

                    # Do request for Tensorkeys 
                    setattr(self.task_runner,attr[0],new_value)
                    self.logger.info(f'Successfully updated {attr[0]}')
                    _hash = type_handler.get_hash(new_value)
                    self.logger.debug(f'{attr[0]} hash: {_hash}')
                    self.attr_hashes[attr[0]] = _hash


    def find_func_output_names(self,source):
        """
        Attempt to find the variable name that is returned by the function.
        This will not always be possible, in which case return 'func_name.index'
        """
        source = source.split('\n')
        return_names_size_dict = {}
        for line in source:
            if 'return' in line:
                l = line.lstrip().strip('return').lstrip().rstrip()
                if ',' in l:
                    names = l.split(',')
                else:
                    names = [l]
                size = len(names)
                #TODO need to handle the case when the variable name starts with an underscore
                if str(size) in return_names_size_dict:
                    # return names are possible non unique
                    return_names_size_dict[str(size)] = list(range(size))
                else:
                    return_names_size_dict[str(size)] = names
        return return_names_size_dict


    def do_task(self, task, round_number):
        """Do the specified task."""
        # map this task to an actual function name and kwargs
        func_name = self.task_config[task]['function']
        kwargs = self.task_config[task]['kwargs']

        ## now we have whatever the model needs to do the task
        func = getattr(self.task_runner, func_name)
        func_source = getsource(func)
        func_output_name_dict = self.find_func_output_names(func_source)
        self.logger.info(f'output of task {task}: {func_output_name_dict}')
        if kwargs is not None:
            results = func(**kwargs)
        else:
            results = func()

        type_handler_factory = TypeHandlerFactory()

        global_output_tensor_dict = {}
        local_output_tensor_dict = {}

        # This will be determined dynamically for the task depending on whether a FederatedDataLoader was used
        data_size = 1

        if results is not None:
            if results is not tuple:
                results = (results,)
            variable_names = func_output_name_dict[str(len(results))]
            for result,name in zip(results,variable_names):
                if name[0] is not '_':
                    self.logger.info(f'Attempting to send {name} of type {type(result)}')
                    if type_handler_factory.is_supported(result):
                        type_handler = type_handler_factory.get_type_handler(result)
                        # The special attribute name only applies for metrics being reported to guarantee uniqueness
                        tensor_output_dict = type_handler.attr_to_map(result,f'{task} : {name}',origin=self.collaborator_name,
                                round_num=round_number,report=True)
                        global_output_tensor_dict = {**global_output_tensor_dict,**tensor_output_dict}

        # Process changed attributes
        for attr in inspect.getmembers(self.task_runner):
            if isinstance(attr[1],FederatedDataLoader):
                if attr[1].get_access_count() != self.attr_hashes[attr[0]]:
                    self.logger.info(f'{attr[0]} has been accessed {attr[1].get_access_count() - self.attr_hashes[attr[0]]} more times since the previous function call')
                    data_size = attr[1].get_loader_data_size()
                    self.logger.info(f'Setting {task} data size to {data_size}')
                    self.attr_hashes[attr[0]] = attr[1].get_access_count()
            if '__' not in attr[0] and attr[0][0] is not '_':
                if type_handler_factory.is_supported(attr[1]):
                    # The attr was newly initialized
                    type_handler = type_handler_factory.get_type_handler(attr[1])
                    if attr[0] in self.attr_hashes: 
                        hash_ = type_handler.get_hash(attr[1])
                        if self.attr_hashes[attr[0]] == hash_:
                            continue
                        else:
                            self.attr_hashes[attr[0]] == hash_

                    tensorkeys = type_handler.attr_to_map(attr[1],attr[0])
                    tensor_output_dict = type_handler.attr_to_map(
                            attr[1],attr[0],origin=self.collaborator_name,
                            round_num=round_number,report=False)

                    global_output_tensor_dict = {**global_output_tensor_dict,**tensor_output_dict}


        self.logger.debug(f'global_output_tensor_dict = {global_output_tensor_dict}')

        # Save global and local output_tensor_dicts to TensorDB
        self.tensor_db.cache_tensor(global_output_tensor_dict)
        self.tensor_db.cache_tensor(local_output_tensor_dict)

        # send the results for this tasks; delta and compression will occur in
        # this function
        self.send_task_results(global_output_tensor_dict, round_number, task, data_size)

    def get_numpy_dict_for_tensorkeys(self, tensor_keys):
        """Get tensor dictionary for specified tensorkey set."""
        return {k.tensor_name: self.get_data_for_tensorkey(k) for k in tensor_keys}

    def get_data_for_tensorkey(self, tensor_key):
        """
        Resolve the tensor corresponding to the requested tensorkey.

        Args
        ----
        tensor_key:         Tensorkey that will be resolved locally or
                            remotely. May be the product of other tensors
        """
        # try to get from the store
        tensor_name, origin, round_number, round_phase, report, tags = tensor_key
        self.logger.debug(
            'Attempting to retrieve tensor {} from local store'.format(
                tensor_key)
        )
        nparray = self.tensor_db.get_tensor_from_cache(tensor_key)

        # if None and origin is our client, request it from the client
        if nparray is None:
            if origin == self.collaborator_name:
                self.logger.info(
                    'Attempting to find locally stored {} tensor from prior'
                    ' round...'.format(tensor_name))
                prior_round = round_number - 1
                while prior_round >= 0:
                    nparray = self.tensor_db.get_tensor_from_cache(
                        TensorKey(tensor_name, origin, prior_round, round_phase, report, tags))
                    if nparray is not None:
                        self.logger.debug(
                            'Found tensor {} in local TensorDB for round'
                            ' {}'.format(tensor_name, prior_round))
                        return nparray
                    prior_round -= 1
                self.logger.info('Cannot find any prior version of tensor {}'
                                 ' locally...'.format(tensor_name))
            self.logger.debug('Unable to get tensor from local store...'
                              'attempting to retrieve from client')
            # Determine whether there are additional compression related
            # dependencies.
            # Typically, dependencies are only relevant to model layers
            tensor_dependencies = self.tensor_codec.find_dependencies(
                tensor_key, self.delta_updates
            )
            # self.logger.info('tensor_dependencies = {}'.format(
            # tensor_dependencies))
            if len(tensor_dependencies) > 0:
                # Resolve dependencies
                # tensor_dependencies[0] corresponds to the prior version
                # of the model.
                # If it exists locally, should pull the remote delta because
                # this is the least costly path
                prior_model_layer = self.tensor_db.get_tensor_from_cache(
                    tensor_dependencies[0]
                )
                if prior_model_layer is not None:
                    uncompressed_delta = \
                        self.get_aggregated_tensor_from_aggregator(
                            tensor_dependencies[1]
                        )
                    new_model_tk, nparray = self.tensor_codec.apply_delta(
                        tensor_dependencies[1],
                        uncompressed_delta,
                        prior_model_layer
                    )
                    self.logger.debug('Applied delta to tensor {}'.format(
                        tensor_dependencies[0][0])
                    )
                else:
                    # The original model tensor should be fetched from client
                    nparray = self.get_aggregated_tensor_from_aggregator(
                        tensor_key
                    )
            elif 'model' in tags:
                # Pulling the model for the first time or
                nparray = self.get_aggregated_tensor_from_aggregator(
                    tensor_key,
                    require_lossless=True
                )
        else:
            self.logger.debug('Found tensor {} in local TensorDB'.format(
                tensor_key))

        return nparray

    def get_aggregated_tensor_from_aggregator(self, tensor_key,
                                              require_lossless=False):
        """
        Return the decompressed tensor associated with the requested tensor key.

        If the key requests a compressed tensor (in the tag), the tensor will
        be decompressed before returning
        If the key specifies an uncompressed tensor (or just omits a compressed
        tag), the decompression operation will be skipped

        Args
        ----
        tensor_key  :               The requested tensor
        require_lossless:   Should compression of the tensor be allowed
                                    in flight?
                                    For the initial model, it may affect
                                    convergence to apply lossy
                                    compression. And metrics shouldn't be
                                    compressed either

        Returns
        -------
        nparray     : The decompressed tensor associated with the requested
                      tensor key
        """
        tensor_name, origin, round_number, round_phase, report, tags = tensor_key

        self.logger.debug('Requesting aggregated tensor {}'.format(tensor_key))
        tensor = self.client.get_aggregated_tensor(
            self.collaborator_name, tensor_name, round_number, round_phase, report, tags, require_lossless)

        # this translates to a numpy array and includes decompression, as
        # necessary
        nparray = self.named_tensor_to_nparray(tensor)

        # cache this tensor
        self.tensor_db.cache_tensor({tensor_key: nparray})
        # self.logger.info('Printing updated TensorDB: {}'.format(
        # self.tensor_db))

        return nparray

    def send_task_results(self, tensor_dict, round_number, task_name, data_size):
        """Send task results to the aggregator."""
        named_tensors = [
            self.nparray_to_named_tensor(k, v) for k, v in tensor_dict.items()
        ]

        for tensor in tensor_dict:
            tensor_name, origin, fl_round, round_phase, report, tags = tensor

            if report:
                self.logger.info(
                    f'Sending metric for task {task_name},'
                    f' round number {round_number}:'
                    f' {tensor_name}\t{tensor_dict[tensor]}')

        self.client.send_local_task_results(
            self.collaborator_name, round_number, task_name, data_size, named_tensors)

    def nparray_to_named_tensor(self, tensor_key, nparray):
        """
        Construct the NamedTensor Protobuf.

        Includes logic to create delta, compress tensors with the TensorCodec, etc.
        """
        # if we have an aggregated tensor, we can make a delta
        tensor_name, origin, round_number, round_phase, report, tags = tensor_key
        if 'trained' in tags and self.delta_updates:
            # Should get the pretrained model to create the delta. If training
            # has happened,
            # Model should already be stored in the TensorDB
            model_nparray = self.tensor_db.get_tensor_from_cache(
                TensorKey(
                    tensor_name,
                    origin,
                    round_number,
                    round_phase,
                    report,
                    ('model',)
                )
            )

            # The original model will not be present for the optimizer on the
            # first round.
            if model_nparray is not None:
                delta_tensor_key, delta_nparray = \
                    self.tensor_codec.generate_delta(
                        tensor_key,
                        nparray,
                        model_nparray
                    )
                delta_comp_tensor_key, delta_comp_nparray, metadata = \
                    self.tensor_codec.compress(delta_tensor_key, delta_nparray)
                named_tensor = utils.construct_named_tensor(
                    delta_comp_tensor_key,
                    delta_comp_nparray,
                    metadata,
                    lossless=False
                )
                return named_tensor

        # Assume every other tensor requires lossless compression
        compressed_tensor_key, compressed_nparray, metadata = \
            self.tensor_codec.compress(
                tensor_key, nparray, require_lossless=True
            )
        named_tensor = utils.construct_named_tensor(
            compressed_tensor_key,
            compressed_nparray,
            metadata,
            lossless=True
        )

        return named_tensor

    def named_tensor_to_nparray(self, named_tensor):
        """Convert named tensor to a numpy array."""
        # do the stuff we do now for decompression and frombuffer and stuff
        # This should probably be moved back to protoutils
        raw_bytes = named_tensor.data_bytes
        metadata = [{'int_to_float': proto.int_to_float,
                     'int_list': proto.int_list,
                     'bool_list': proto.bool_list
                     } for proto in named_tensor.transformer_metadata]
        # The tensor has already been transfered to collaborator, so
        # the newly constructed tensor should have the collaborator origin
        tensor_key = TensorKey(
            named_tensor.name,
            self.collaborator_name,
            named_tensor.round_number,
            named_tensor.round_phase,
            named_tensor.report,
            tuple(named_tensor.tags)
        )
        tensor_name, origin, round_number, round_phase, report, tags = tensor_key
        if 'compressed' in tags:
            decompressed_tensor_key, decompressed_nparray = \
                self.tensor_codec.decompress(
                    tensor_key,
                    data=raw_bytes,
                    transformer_metadata=metadata,
                    require_lossless=True
                )
        elif 'lossy_compressed' in tags:
            decompressed_tensor_key, decompressed_nparray = \
                self.tensor_codec.decompress(
                    tensor_key,
                    data=raw_bytes,
                    transformer_metadata=metadata
                )
        else:
            # There could be a case where the compression pipeline is bypassed
            # entirely
            self.logger.warning('Bypassing tensor codec...')
            decompressed_tensor_key = tensor_key
            decompressed_nparray = raw_bytes

        self.tensor_db.cache_tensor(
            {decompressed_tensor_key: decompressed_nparray}
        )

        return decompressed_nparray
