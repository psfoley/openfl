# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import pandas as pd
import numpy as np

from tfedlrn.tensor_transformation_pipelines import NoCompressionPipeline

class Collaborator(object):

    def __init__(self, aggregator, model, collaborator_name, aggregator_uuid, federation_uuid, tasks_config):
        # We would really want this as an object
        self.tensor_db = pd.DataFrame([], columns=['tensor_name','origin','round','tags','nparray']) 
        self.tensor_codec = TensorCodec(self.compression_pipeline)
        self.aggregator = aggregator
        self.model = model
        self.header = MessageHeader(sender=collaborator_name, receiver=aggregator_uuid, federation_uuid=federation_uuid)
        self.tasks_config = tasks_config # pulled from flplan

    def run(self):
        while True:
            tasks = self.get_tasks()
            if tasks.quit:
                break
            elif tasks.sleep_time > 0:
                self.sleep(tasks.sleep_time) # some sleep function
            else:
                for task in tasks.tasks:
                    self.do_task(task, tasks.round_number)

    def get_tasks(self):
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
        for tname, origin, rnd_num in required_tensorkeys_relative:
            if origin == 'GLOBAL':
                origin = self.aggregator_uuid
            else:
                origin = self.collaborator_name
            
            #rnd_num is the relative round. So if rnd_num is -1, get the tensor from the previous round
            required_tensorkeys.append(TensorKey(tname, origin, rnd_num + round_number))
        
        input_tensor_dict = self.get_numpy_dict_for_tensorkeys(required_tensorkeys)

        # now we have whatever the model needs to do the task
        func = getattr(self.model, func_name)
        output_tensor_dict = func(self.collaborator_name, round_number, input_tensor_dict, **kwargs)

        # Save output_tensor_dict to TensorDB
        self.insert_to_tensor_db(output_tensor_dict)

        # send the results for this tasks
        self.send_task_results(output_tensor_dict, round_number, task)

    def get_numpy_dict_for_tensorkeys(self, tensor_keys):
        return {t.tensor_name: self.get_data_for_tensorkey(k) for k in tensor_keys}

    def get_data_for_tensorkey(self, tensor_key):
        """
        This function resolves the tensor corresponding to the requested tensorkey

        Args
        ----
        tensor_key:         Tensorkey that will be resolved locally or remotely. May be the product of other tensors
        extract_metadata:   The requested tensor may have metadata needed for decompression
        """
        # try to get from the store
        nparray = self.get_tensor_from_tensor_cache(tensor_key)

        # if None and origin is our aggregator, request it from the aggregator
        if nparray is None:
            #Determine whether there are additional compression related dependencies. 
            #Typically, dependencies are only relevant to model layers
            tensor_dependencies = self.tensor_codec.find_dependencies(tensor_key,self.send_model_deltas)
            if len(tensor_dependencies) > 0:
                #Resolve dependencies
                #tensor_dependencies[0] corresponds to the prior version of the model. 
                #If it exists locally, should pull the remote delta because this is the least costly path
                prior_model_layer = self.get_tensor_from_tensor_cache(tensor_dependencies[0])
                if prior_model_layer != None:
                    uncompressed_delta = self.get_aggregated_tensor_from_aggregator(tensor_dependencies[1])
                    nparray = self.tensor_codec.apply_delta(tensor_dependencies[0],prior_model_layer,uncompressed_delta)
                else:
                    #The original model tensor should be fetched from aggregator
                    nparray = self.get_aggregated_tensor_from_aggregator(tensor_key)
            elif 'model' in tensor_key[3]:
                #Pulling the model for the first time or 
                nparray = self.get_aggregated_tensor_from_aggregator(tensor_name,allow_compression=False)

        return nparray
    
    def get_tensor_from_cache(self, tensor_key):
        """
        Performs a lookup of the tensor_key in the TensorDB. Returns the nparray if it is available
        Otherwise, it returns 'None'
        """
        df = self.tensor_db[(self.tensor_db['tensor_name'] == tensor_key['tensor_name']) & \
                            (self.tensor_db['origin'] == tensor_key['origin']) & \
                            (self.tensor_db['round_num'] == tensor_key['round_num']) & \
                            (self.tensor_db['tags'] == tensor_key['tags'])]
        if len(df) == 0:
            return None
        return df['nparray'][0][0]

    def get_aggregated_tensor_from_aggregator(self, tensor_key,allow_compression=True):
        """
        This function returns the decompressed tensor associated with the requested tensor key. 
        If the key requests a compressed tensor (in the tag), the tensor will be decompressed before returning
        If the key specifies an uncompressed tensor (or just omits a compressed tag), the decompression operation will be skipped

        Args
        ----
        tensor_key  :       The requested tensor
        allow_compression:  Should compression of the tensor be allowed in flight? For the initial model, it may
                            affect convergence to apply lossy compression. And metrics shouldn't be compressed either

        Returns
        -------
        nparray     : The decompressed tensor associated with the requested tensor key
        """
        tensor_name = tensor_key[0]
        round_number = tensor_key[2]
        tags = tensor_key[3]
        request = TensorRequest(header=self.header,
                                round_number=round_number,
                                allow_compression=allow_compression,
                                tensor_name=tensor_name,
                                tags=tags)
        
        response = self.aggregator.GetAggregatedTensor(request)

        # also do other validation, like on the round_number
        self.validate_reponse(response)

        # this translates to a numpy array and includes decompression, as necessary
        nparray = self.named_tensor_to_nparray(response.named_tensor)
        
        # cache this tensor
        self.cache_tensor(tensor_key, nparray)

        return tensor
    
    def send_task_results(self, tensor_dict, round_number, task_name):
        named_tensors = [self.nparray_to_named_tensor(v, k, round_number) for k, v in tensor_dict.items()]
        request = TaskResults(header=self.header,
                              round_number=round_number,
                              task_name=task_name,
                              tensors=named_tensors)
        response = self.aggregator.SendLocalTaskResults(TaskResults)
        self.validate_reponse(response)

    def nparray_to_named_tensor(self,tensor_key, nparray, round_number):
        """
        This function constructs the NamedTensor Protobuf and also includes logic to create delta, 
        compress tensors with the TensorCodec, etc.
        """

        # if we have an aggregated tensor, we can make a delta
        previous_nparray = None
        #If 'model' is in the tensor, we can create a delta from the 
        if('model' in tensor_key['tags']):
          previous_nparray = self.get_tensor_from_cache(TensorKey(tensor_key['tensor_name'],\
                                                                  tensor_key['origin'],\
                                                                  tensor_key['round_num'] - 1,\
                                                                  tensor_key['tags'])) 
        if('trained' in tensor_key['tags']):
          #Should get the pretrained model to create the delta
          previous_nparray = self.get_tensor_from_cache(TensorKey(tensor_key['tensor_name'],\
                                                                  tensor_key['origin'],\
                                                                  tensor_key['round_num'],\
                                                                  ('model'))) 
        if previous_nparray is not None:
            nparray -= previous_nparray
            #self.cache_tensor(append_tensor_tag(tensor_key,'delta'),nparray)
            delta_tensor_key = TensorKey(tensor_key[0],tensor_key[1],tensor_key[2],tuple(list(tensor_key[3])+['delta']))
            self.cache_tensor(delta_tensor_key,nparray)
            is_delta = True
        else:
            is_delta = False
        
        # do the remaining stuff as we do now for compression and tobytes and stuff
        #output_tensor_dict = self.tensor_codec.compress(tensor_key,nparray)
        output_tensor_dict = TensorCodec('compress',self.compression_pipeline))


        ...
        return named_tensor

    def named_tensor_to_nparray(named_tensor):
        # do the stuff we do now for decompression and frombuffer and stuff
        # This should probably be moved back to protoutils
        raw_bytes = named_tensor.data_bytes
        metadata = [{'int_to_float': proto.int_to_float,
                     'int_list': proto.int_list,
                     'bool_list': proto.bool_list} for proto in named_tensor.transformer_metadata]
        #The tensor has already been transfered to collaborator, so the newly constructed tensor should have the collaborator origin
        tensor_key = TensorKey(named_tensor.name, self.collaborator_name, named_tensor.round_number, named_tensor.tags)
        if 'compressed' in tensor_key[3]:
            decompressed_tensor_key, decompressed_nparray =  \
                    self.tensor_codec.decompress(tensor_key,data=raw_bytes,transformer_metadata=metadata,override_with_lossless=True)
        elif 'lossy_compressed' in tensor_key[3]:
            decompressed_tensor_key, decompressed_nparray =  \
                    self.tensor_codec.decompress(tensor_key,data=raw_bytes,transformer_metadata=metadata)
        else:
            #There could be a case where the compression pipeline is bypassed entirely
            print('Bypassing tensor codec...') 
            decompressed_tensor_key = tensor_key
            decompressed_nparray = raw_bytes

        self.cache_tensor(decompressed_tensor_key,decompressed_nparray)
        
        return decompressed_nparray
    
    def cache_tensor(self, tensor_key, nparray):
        """
        Insert tensor into TensorDB (dataframe)
        """
        tensor_name = tensor_key['tensor_name']
        origin = tensor_key['origin']
        round_num = tensor_key['round_num']
        tags = tensor_key['tags']
        df = pd.DataFrame([tensor_name,origin,round,tags,[nparray]], \
                          columns=['tensor_name','origin','round','tags','nparray']) 
        self.tensor_db = pd.concat([self.tensor_db,df],ignore_index=True)
