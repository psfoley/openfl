# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

TensorKey = namedtuple('TensorKey', ['tensor_name', 'origin', 'round_number'])

# this is the pseudocode. A more detailed implementation is collaborator_rework_strawman.py
class Collaborator(object):
    def __init__(...):
        self.tensor_cache = {} # We would really want this as an object
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

    def do_task(self, task, round_number):
        # map this task to an actual function name and kwargs
        func_name   = self.tasks_config[task]['function']
        kwargs      = self.tasks_config[task]['kwargs']

        # this would return a list of what tensors we require as TensorKeys
        # models actually return "relative" tensorkeys of (LOCAL|GLOBAL, name, round_offset)
        # so we need to update these keys to their "absolute values"
        required_tensorkeys = self.model.get_required_tensorkeys_for_function(func_name, **kwargs)
        
        # anything it can't find in the local cache, and the "origin" is the aggregator, it calls the 
        # aggregator RPC call to get that specific tensor for that tensorkey
        input_numpy_dict = self.get_numpy_dict_for_tensorkeys(required_tensorkeys)

        # now we have whatever the model needs to do the task
        func = getattr(self.model, func_name)
        output_tensor_dict = func(round_number, input_tensor_dict, **kwargs)

        # turn the output dict into a dictionary of tensorkeys
        # send the results for this tasks
        # this will 
        self.send_task_results(output_tensor_dict, round_number, task)
