# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

TensorKey = namedtuple('TensorKey', ['tensor_name', 'origin', 'round_number'])

class Collaborator(object):

    def __init__(self, aggregator, model, collaborator_name, aggregator_uuid, federation_uuid, tasks_config):
        self.tensor_cache = {} # We would really want this as an object
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

        # models actually return "relative" tensorkeys of (LOCAL|GLOBAL, name, round_offset)
        # so we need to update these keys to their "absolute values"
        required_tensorkeys = []
        for tname, origin, rnd_num in required_tensorkeys_relative:
            if origin == 'GLOBAL':
                origin = self.aggregator_uuid
            else:
                origin = self.collaborator_name
            
            required_tensorkeys.append(TensorKey(tname, origin, rnd_num + round_number))
        
        input_numpy_dict = self.get_numpy_dict_for_tensorkeys(required_tensorkeys)

        # now we have whatever the model needs to do the task
        func = getattr(self.model, func_name)
        output_tensor_dict = func(round_number, input_tensor_dict)

        # send the results for this tasks
        self.send_task_results(output_tensor_dict, round_number, task)

    def get_numpy_dict_for_tensorkeys(self, tensor_keys):
        return {t.tensor_name: self.get_numpy_dict_for_tensorkey(k) for k in tensor_keys}

    def get_numpy_dict_for_tensorkey(self, tensor_key):
        # try to get from the store
        nparray = self.get_tensor_from_store(tensor_key)

        # if None and origin is our aggregator, request it from the aggregator
        if nparray is None:
            nparray = self.get_aggregated_tensor_from_aggregator(tensor_name, round_number)

        return nparray
    
    def get_tensor_from_cache(self, tensor_name, round_number):
        return self.aggregated_tensor_cache.get(CacheKey(tensor_name, round_number))

    def get_aggregated_tensor_from_aggregator(self, tensor_name, round_number):
        request = TensorRequest(header=self.header,
                                round_number=round_number,
                                tensor_name=tensor_name)
        
        response = self.aggregator.GetAggregatedTensor(request)

        # also do other validation, like on the round_number
        self.validate_reponse(response)

        # this translates to a numpy array and includes decompression, as necessary
        nparray = self.named_tensor_to_nparray(response.tensor)
        
        # cache this tensor
        self.cache_tensor(tensor_name, round_number, nparray)

        return tensor
    
    def send_task_results(self, tensor_dict, round_number, task_name):
        named_tensors = [self.nparray_to_named_tensor(v, k, round_number) for k, v in tensor_dict.items()]
        request = TaskResults(header=self.header,
                              round_number=round_number,
                              task_name=task_name,
                              tensors=named_tensors)
        response = self.aggregator.SendLocalTaskResults(TaskResults)
        self.validate_reponse(response)

    def nparray_to_named_tensor(name, nparray, round_number):
        # if we have an aggregated tensor, we can make a delta
        previous_nparray = get_tensor_from_cache(name, round_number - 1)

        if previous_nparray is not None:
            nparray -= previous_nparray
            is_delta = True
        else:
            is_delta = False
        
        # do the remaining stuff as we do now for compression and tobytes and stuff
        ...

    def named_tensor_to_nparray(named_tensor):
        # do the stuff we do now for decompression and frombuffer and stuff
        nparray = ...

        # if the tensor is a delta, we need to get the previous tensor value and add it
        if response.is_delta:
            nparray += self.get_aggregated_tensor(tensor_name, round_number - 1)
        
        return nparray
    
    def cache_tensor(self, tensor_name, round_number, nparray):
        k = CacheKey(tensor_name, round_number)
        self.aggregated_tensor_cache[k] = nparray)
