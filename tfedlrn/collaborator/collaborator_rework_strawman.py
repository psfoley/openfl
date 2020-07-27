# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import pandas as pd
import numpy as np
import time
import logging

from ..proto.lowlevelstrawman_pb2 import MessageHeader, MetadataProto, ModelProto
from ..proto.lowlevelstrawman_pb2 import TasksRequest, TasksResponse, TensorRequest, TensorResponse, NamedTensor, Acknowledgement, TaskResults
from .. import check_type, check_equal, check_not_equal, split_tensor_dict_for_holdouts


from tfedlrn.tensor_transformation_pipelines import NoCompressionPipeline, TensorCodec
from tfedlrn.tensor_db import TensorDB
from tfedlrn import TensorKey

from tfedlrn.proto.protoutils import construct_proto, deconstruct_proto, construct_named_tensor


class Collaborator(object):

    def __init__(self, 
                 aggregator, 
                 model, 
                 collaborator_name, 
                 compression_pipeline,
                 aggregator_uuid, 
                 federation_uuid, 
                 tasks_config,
                 send_model_deltas = False,
                 single_col_cert_common_name = None,
                 **kwargs):
        self.single_col_cert_common_name = single_col_cert_common_name
        if self.single_col_cert_common_name is None:
            self.single_col_cert_common_name = '' #For protobuf compatibility
        # We would really want this as an object
        self.collaborator_name = collaborator_name
        self.logger = logging.getLogger(__name__)
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()
        self.tensor_codec = TensorCodec(self.compression_pipeline)
        self.tensor_db = TensorDB() 
        self.aggregator = aggregator
        self.model = model
        self.send_model_deltas = send_model_deltas
        self.header = MessageHeader(sender=collaborator_name, receiver=aggregator_uuid, \
                                    federation_uuid=federation_uuid, single_col_cert_common_name=self.single_col_cert_common_name)
        self.tasks_config = tasks_config # pulled from flplan

        # Check that the message was intended to go to this collaborator

    def validate_response(self, reply):
        check_equal(reply.header.receiver, self.collaborator_name, self.logger)
        check_equal(reply.header.sender, self.aggregator_uuid, self.logger)

        #Check that federation id matches
        check_equal(reply.header.federation_uuid, self.federation_uuid, self.logger)

        # check that there is aggrement on the single_col_cert_common_name
        check_equal(reply.header.single_col_cert_common_name, self.single_col_cert_common_name, self.logger)

    def run(self):
        while True:
            tasks = self.get_tasks()
            if tasks.quit:
                break
            elif tasks.sleep_time > 0:
                time.sleep(tasks.sleep_time) # some sleep function
            else:
                for task in tasks.tasks:
                    self.do_task(task, tasks.round_number)
        self.logger.info('End of experiment reached. Exiting...')

    def get_tasks(self):
        self.logger.info('Calling get_tasks...')
        request = TasksRequest(header=self.header)
        response = self.aggregator.GetTasks(request)
        self.validate_response(response) # sanity checks and validation
        return response

    def do_task(self, task, round_number):
        # map this task to an actual function name and kwargs
        func_name   = self.tasks_config[task]['function']
        kwargs      = self.tasks_config[task]['kwargs']

        # this would return a list of what tensors we require as TensorKeys
        required_tensorkeys_relative = self.model.get_required_tensorkeys_for_function(func_name, **kwargs)

        # models actually return "relative" tensorkeys of (name, LOCAL|GLOBAL, round_offset)
        # so we need to update these keys to their "absolute values"
        required_tensorkeys = []
        for tname, origin, rnd_num, tags in required_tensorkeys_relative:
            if origin == 'GLOBAL':
                origin = self.aggregator_uuid
            else:
                origin = self.collaborator_name
            
            #rnd_num is the relative round. So if rnd_num is -1, get the tensor from the previous round
            required_tensorkeys.append(TensorKey(tname, origin, rnd_num + round_number, tags))
        
        input_tensor_dict = self.get_numpy_dict_for_tensorkeys(required_tensorkeys)
        print('input_tensor_dict = {}'.format(input_tensor_dict))

        # now we have whatever the model needs to do the task
        func = getattr(self.model, func_name)
        output_tensor_dict = func(col_name=self.collaborator_name, round_num=round_number, input_tensor_dict=input_tensor_dict, **kwargs)

        # Save output_tensor_dict to TensorDB
        self.tensor_db.cache_tensor(output_tensor_dict)

        # send the results for this tasks; delta and compression will occur in this function
        self.send_task_results(output_tensor_dict, round_number, task)

    def get_numpy_dict_for_tensorkeys(self, tensor_keys):
        return {k.tensor_name: self.get_data_for_tensorkey(k) for k in tensor_keys}

    def get_data_for_tensorkey(self, tensor_key):
        """
        This function resolves the tensor corresponding to the requested tensorkey

        Args
        ----
        tensor_key:         Tensorkey that will be resolved locally or remotely. May be the product of other tensors
        extract_metadata:   The requested tensor may have metadata needed for decompression
        """
        # try to get from the store
        self.logger.info('Attempting to get tensor {} from local store'.format(tensor_key))
        nparray = self.tensor_db.get_tensor_from_cache(tensor_key)

        # if None and origin is our aggregator, request it from the aggregator
        if nparray is None:
            self.logger.info('Unable to get tensor from local store...attempting to retrieve from aggregator')
            #Determine whether there are additional compression related dependencies. 
            #Typically, dependencies are only relevant to model layers
            tensor_dependencies = self.tensor_codec.find_dependencies(tensor_key,self.send_model_deltas)
            self.logger.info('tensor_dependencies = {}'.format(tensor_dependencies))
            if len(tensor_dependencies) > 0:
                #Resolve dependencies
                #tensor_dependencies[0] corresponds to the prior version of the model. 
                #If it exists locally, should pull the remote delta because this is the least costly path
                prior_model_layer = self.tensor_db.get_tensor_from_cache(tensor_dependencies[0])
                if prior_model_layer != None:
                    uncompressed_delta = self.get_aggregated_tensor_from_aggregator(tensor_dependencies[1])
                    nparray = self.tensor_codec.apply_delta(tensor_dependencies[1],uncompressed_delta,prior_model_layer)
                else:
                    #The original model tensor should be fetched from aggregator
                    nparray = self.get_aggregated_tensor_from_aggregator(tensor_key)
            elif 'model' in tensor_key[3]:
                #Pulling the model for the first time or 
                nparray = self.get_aggregated_tensor_from_aggregator(tensor_key,require_lossless=True)

        return nparray

    def get_aggregated_tensor_from_aggregator(self, tensor_key, require_lossless=False):
        """
        This function returns the decompressed tensor associated with the requested tensor key. 
        If the key requests a compressed tensor (in the tag), the tensor will be decompressed before returning
        If the key specifies an uncompressed tensor (or just omits a compressed tag), the decompression operation will be skipped

        Args
        ----
        tensor_key  :               The requested tensor
        permit_lossy_compression:   Should compression of the tensor be allowed in flight? For the initial model, it may
                                    affect convergence to apply lossy compression. And metrics shouldn't be compressed either

        Returns
        -------
        nparray     : The decompressed tensor associated with the requested tensor key
        """
        tensor_name = tensor_key[0]
        round_number = tensor_key[2]
        tags = tensor_key[3]
        request = TensorRequest(header=self.header,
                                tensor_name=tensor_name,
                                round_number=round_number,
                                tags=tags,
                                require_lossless=require_lossless)

        self.logger.info('Getting aggregated tensor {}'.format(tensor_key))
        
        response = self.aggregator.GetAggregatedTensor(request)

        # also do other validation, like on the round_number
        self.validate_response(response)

        # this translates to a numpy array and includes decompression, as necessary
        nparray = self.named_tensor_to_nparray(response.tensor)
        
        # cache this tensor
        self.tensor_db.cache_tensor({tensor_key: nparray})
        self.logger.info('Printing updated TensorDB: {}'.format(self.tensor_db))

        return nparray
    
    def send_task_results(self, tensor_dict, round_number, task_name):
        named_tensors = [self.nparray_to_named_tensor(k, v) for k, v in tensor_dict.items()]
        #For general tasks, there may be no notion of data size to send. But that raises the question
        #how to properly aggregate results.
        data_size = -1
        if 'train' in task_name:
            data_size = self.model.get_training_data_size()
        if 'valid' in task_name:
            data_size = self.model.get_validation_data_size()
        request = TaskResults(header=self.header,
                              round_number=round_number,
                              task_name=task_name,
                              data_size=data_size,
                              tensors=named_tensors)
        response = self.aggregator.SendLocalTaskResults(request)
        self.validate_response(response)

    def nparray_to_named_tensor(self,tensor_key, nparray):
        """
        This function constructs the NamedTensor Protobuf and also includes logic to create delta, 
        compress tensors with the TensorCodec, etc.
        """

        # if we have an aggregated tensor, we can make a delta
        if('trained' in tensor_key[0] and self.send_model_deltas):
          #Should get the pretrained model to create the delta. If training has happened,
          #Model should already be stored in the TensorDB
          model_nparray = self.tensor_db.get_tensor_from_cache(TensorKey(tensor_key[0],\
                                                                  tensor_key[1],\
                                                                  tensor_key[2],\
                                                                  ('model',))) 
        
          assert(model_nparray != None), "The original model layer should be present if the trained model is present"
          delta_tensor_key, delta_nparray = self.tensor_codec.generate_delta(tensor_key,nparray,model_nparray)
          delta_comp_tensor_key,delta_comp_nparray,metadata = self.tensor_codec.compress(delta_tensor_key,delta_nparray)
          named_tensor = construct_named_tensor(delta_comp_tensor_key,delta_comp_nparray,metadata,lossless=False)

        else:
            #Assume every other tensor requires lossless compression
            compressed_tensor_key, compressed_nparray, metadata = \
                    self.tensor_codec.compress(tensor_key,nparray,require_lossless=True)
            named_tensor = construct_named_tensor(compressed_tensor_key,compressed_nparray,metadata,lossless=True)
        
        return named_tensor

    def named_tensor_to_nparray(self, named_tensor):
        # do the stuff we do now for decompression and frombuffer and stuff
        # This should probably be moved back to protoutils
        raw_bytes = named_tensor.data_bytes
        metadata = [{'int_to_float': proto.int_to_float,
                     'int_list': proto.int_list,
                     'bool_list': proto.bool_list} for proto in named_tensor.transformer_metadata]
        #The tensor has already been transfered to collaborator, so the newly constructed tensor should have the collaborator origin
        tensor_key = TensorKey(named_tensor.name, self.collaborator_name, named_tensor.round_number, tuple(named_tensor.tags))
        if 'compressed' in tensor_key[3]:
            decompressed_tensor_key, decompressed_nparray =  \
                    self.tensor_codec.decompress(tensor_key,data=raw_bytes,transformer_metadata=metadata,require_lossless=True)
        elif 'lossy_compressed' in tensor_key[3]:
            decompressed_tensor_key, decompressed_nparray =  \
                    self.tensor_codec.decompress(tensor_key,data=raw_bytes,transformer_metadata=metadata)
        else:
            #There could be a case where the compression pipeline is bypassed entirely
            print('Bypassing tensor codec...') 
            decompressed_tensor_key = tensor_key
            decompressed_nparray = raw_bytes

        self.tensor_db.cache_tensor({decompressed_tensor_key: decompressed_nparray})
        
        return decompressed_nparray
