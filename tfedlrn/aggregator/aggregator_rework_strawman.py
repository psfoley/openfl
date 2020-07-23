# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np
import pandas as pd
from tfedlrn import TensorKey,TaskResultKey

class Aggregator(object):

    """An Aggregator is the central node in federated learning

    Parameters
    ----------
    aggregator_uuid : str
        Aggregation ID.
    federation_uuid : str
        Federation ID.
    collaborator_common_names : list of str
        The list of IDs of enrolled collaborators.
    connection : ?
        Used to be ZMQ connection, but deprecated in gRPC.
    init_model_fpath : str
        The location of the initial weight file.
    latest_model_fpath : str
        The file location to store the latest weight.
    best_model_fpath : str
        The file location to store the weight of the best model.
    """

    def __init__(self,
                 aggregator_uuid,
                 federation_uuid,
                 collaborator_common_names,
                 initial_model_file_path,
                 custom_tensor_dir,
                 task_assigner,
                 rounds_to_train=256,
                 minimum_reporting=-1,
                 straggler_cutoff_time=np.inf,
                 disable_equality_check=True,
                 single_col_cert_common_name=None,
                 compression_pipeline=None,
                 **kwargs):
        self.tensor_db = TensorDB()
        self.tensor_codec = TensorCodec(self.compression_pipeline)
        self.initial_model_file_path = initial_model_file_path
        self.custom_tensor_dir = custom_tensor_dir
        self.load_initial_tensors() # keys are TensorKeys

        self.collaborator_tensor_results = {} # {TensorKey: nparray}}

        # these enable getting all tensors for a task
        self.collaborator_tasks_results = {} # {TaskResultKey: list of TensorKeys}

        self.task_assigner = task_assigner
        ...

    def load_initial_tensors(self):
        """
        Load all of the tensors required to begin federated learning:

        1. Initial model
        2. Any custom tensors. These are previously serialized named tensors

        Parameters
        ----------
        """
        self.model = load_proto(self.initial_model_file_path)deconstruct_model_proto(self.

        
    def GetTasks(self, request):
        # all messages get sanity checked
        self.check_request(request)

        collaborator_name = request.sender

        # first, if it is time to quit, inform the collaborator
        if self.quitting_time():
            return TasksResponse(header=self.get_header(collaborator_name),
                                 round_number=self.round_number,
                                 tasks=None,
                                 sleep_time=0,
                                 quit=True)
        
        # otherwise, get the tasks from our task assigner
        tasks = self.task_assigner.get_tasks_for_collaborator(self, collaborator_name) # fancy task assigners may want aggregator state
        
        # if no tasks, tell the collaborator to sleep
        if tasks is None:
            return TasksResponse(header=self.get_header(collaborator_name),
                                 round_number=self.round_number,
                                 tasks=None,
                                 sleep_time=self.get_sleep_time(), # this could be an extensible function if we want
                                 quit=False)

        # if we do have tasks, remove any that we already have results for
        tasks = [t for t in tasks if not self.collaborator_task_completed(collaborator_name, t, self.round_number)]

        return TasksResponse(header=self.get_header(collaborator_name),
                             round_number=self.round_number,
                             tasks=tasks,
                             sleep_time=0,
                             quit=False)

    def GetAggregatedTensor(self, request):
        # all messages get sanity checked
        self.check_request(request)

        # get the values we need from the protobuf
        collaborator_name   = request.header.sender
        tensor_name         = request.name
        round_number        = request.round_number

        tensor_key = TensorKey(tensor_name, self.uuid, round_number)
        nparray = self.tensor_from_cache(key)
        if nparray is None:
            raise ValueError("Aggregator does not have an aggregated tensor for {}".format(k))

        # quite a bit happens in here, including decompression, delta handling, etc...
        # we might want to cache these as well
        named_tensor = self.nparray_to_named_tensor(nparray)

        return TensorResponse(header=self.get_header(collaborator_name),
                              round_number=round_number,
                              tensor=named_tensor)

    def SendLocalTaskResults(self, request):
        # all messages get sanity checked
        self.check_request(request)

        # get the values we need from the protobuf
        collaborator_name   = request.header.sender
        task_name           = request.task_name
        round_number        = request.round_number
        named_tensors       = request.tensors

        # TODO: do we drop these on the floor?
        # if round_number != self.round_number:
        #     return Acknowledgement(header=self.get_header(collaborator_name))

        task_key = TaskResultKey(task_name, collaborator_name, round_number)

        # we mustn't have results already
        if self.collaborator_task_completed(collaborator_name, task_name, round_number):
            raise ValueError("Aggregator already has task results from collaborator {} for task {}".format(collaborator_name, task_key))

        # initialize the list of tensors that go with this task
        self.collaborator_tasks_results[task_key] = []

        # go through the tensors and add them to the tensor dictionary and the task dictionary
        for named_tensor in named_tensors:
            # sanity check that this tensor has been updated
            if named_tensor.round_number != round_number:
                ... # any recovery?

            # quite a bit happens in here, including decompression, delta handling, etc...
            nparray = self.named_tensor_to_nparray(named_tensor)

            tensor_key = TensorKey(named_tensor.name, collaborator_name, named_tensor.round_number)

            self.collaborator_tensor_results[tensor_key] = nparray
            self.collaborator_tasks_results[task_key].append(tensor_key)
        
        self.end_of_task_check(task_name)

        return Acknowledgement(header=self.get_header(collaborator_name))

    def end_of_task_check(self, task_name):
        if self.is_task_done(task_name):
            self.aggregate_task_results(task_name)
            
            # now check for the end of the round
            self.end_of_round_check()

    def end_of_round_check(self):
        if self.is_round_done():
            # all the state reinit stuff, which I'm not sure there is much anymore
            self.round_number += 1

    def is_task_done(self, task_name):
        collaborators_needed = self.task_assigner.get_collaborators_for_task(task_name, self.round_number)

        return all([self.collaborator_task_completed(c, task_name, self.round_number) for c in collaborators_needed]):

    def is_round_done(self):
        tasks_for_round = self.task_assigner.get_all_tasks_for_round(self.round_number)

        return all([self.is_task_done(t) for t in tasks_for_round])


class GroupedTaskAssigner
