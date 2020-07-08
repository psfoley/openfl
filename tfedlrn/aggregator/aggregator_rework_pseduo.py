# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

TaskResultKey   = namedtuple('TaskResultKey', ['task_name', 'origin', 'round_number'])
TensorKey       = namedtuple('TensorKey', ['tensor_name', 'origin', 'round_number'])

# summary pseudocode. More complete rework started in aggregator_rework_strawman.py
class Aggregator(object):

    def __init__(...):
        self.tensor_cache = self.load_initial_tensors() # We would really want this as an object

        # these enable getting all tensors for a task
        self.task_to_tensor_map = {} # {TaskResultKey: list of TensorKeys}

        self.task_assigner = task_assigner
        ...
        
    def GetTasks(self, request):
        
        # first, if it is time to quit, inform the collaborator
        if self.quitting_time():
            return TasksResponse(quit=True)
        
        # otherwise, get the tasks from our task assigner
        tasks = self.task_assigner.get_tasks_for_collaborator(self, collaborator_name) # fancy task assigners may want aggregator state
        
        # if no tasks, tell the collaborator to sleep
        if tasks is None:
            return TasksResponse(sleep_time=self.get_sleep_time()) # this could be an extensible function if we want

        # if we do have tasks, remove any that we already have results for
        tasks = [t for t in tasks if not self.collaborator_task_completed(collaborator_name, t, self.round_number)]

        return TasksResponse(tasks=tasks)

    def GetAggregatedTensor(self, request):
        nparray = self.tensor_cache.get(request.tensor_key)
        
        # quite a bit happens in here, including decompression, delta handling, etc...
        # we might want to cache these as well
        tensor = self.nparray_to_tensorproto(nparray)

        return TensorResponse(tensor=tensor)

    def SendLocalTaskResults(self, request):
        task_key = TaskResultKey(task_name, collaborator_name, round_number)

        # we mustn't have results already
        ...

        # initialize the list of tensors that go with this task
        self.task_to_tensor_map[task_key] = []

        # go through the tensors and add them to the tensor cache and the task to tensor map
        for tensor in request.tensors:
            # sanity checks
            ...

            # quite a bit happens in here, including decompression, delta handling, etc...
            nparray = self.tensorproto_to_nparray(tensor)

            tensor_key = TensorKey(tensor.tensor_key)

            self.tensor_cache[tensor_key] = nparray
            self.task_to_tensor_map[task_key].append(tensor_key)
        
        self.end_of_task_check(task_name)

        return Acknowledgement(header=self.get_header(collaborator_name))

    def end_of_task_check(self, task_name):
        if self.is_task_done(task_name):
            # aggregation functions store results in the tensor_cache
            self.aggregate_task_results(task_name)
            
            # now check for the end of the round
            self.end_of_round_check()

    def end_of_round_check(self):
        if self.is_round_done():
            # all the state reinit stuff, which I'm not sure there is much anymore
            self.round_number += 1
