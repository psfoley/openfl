# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import numpy as np
import pandas as pd
from threading import Lock

from tfedlrn import TensorKey,TaskResultKey
from tfedlrn.tensor_transformation_pipelines import NoCompressionPipeline

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
        self.round_number = 0
        self.rounds_to_train = rounds_to_train
        self.collaborator_common_names = collaborator_common_names
        self.uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.task_assigner = task_assigner
        self.tensor_db = TensorDB()
        self.compression_pipeline = compression_pipeline or NoCompressionPipeline() 
        self.tensor_codec = TensorCodec(self.compression_pipeline)
        self.initial_model_file_path = initial_model_file_path
        self.custom_tensor_dir = custom_tensor_dir
        self.load_initial_tensors() # keys are TensorKeys

        self.collaborator_tensor_results = {} # {TensorKey: nparray}}

        # these enable getting all tensors for a task
        self.collaborator_tasks_results = {} # {TaskResultKey: list of TensorKeys}
        self.collaborator_task_completion_status = {}
        self.initialize_collaborator_task_status()


    def init_collaborator_task_status(self):
        """
        On initialization, all collaborators haven't completed any tasks. 
        Of course, there are exceptions that should be handled 
        (if aggregator goes down, should be able to recover state from TensorDB/Collaborators)
        """
        for col in self.collaborator_common_names:
            self.collaborator_task_completion_status[col] = {}
            for round in range(self.rounds_to_train):
                self.collaborator_task_completion_status[col][round] = {}
                tasks = self.task_assigner.get_tasks_for_collaborators(col,round)
                    for task in tasks:
                        self.collaborator_task_completion_status[col][round][task] = False


    def load_initial_tensors(self):
        """
        Load all of the tensors required to begin federated learning:

        1. Initial model
        2. Any custom tensors. These are previously serialized named tensors

        Parameters
        ----------
        """
        self.model = load_proto(self.initial_model_file_path)
        tensor_dict = deconstruct_model_proto(self.model,compression_pipeline=self.compression_pipeline)
        tensor_key_dict = {TensorKey(k,self.uuid,0,('model')):v for k,v in tensor_dict.items()}
        #All initial model tensors are loaded here
        self.tensor_db.cache_tensor(tensor_key_dict)
        if custom_tensor_dir != None:
            #Here is where the additional tensors should be loaded into the TensorDB
            pass

    def check_request(self, request):
        """
        Validate request header matches expected values
        """
        #TODO improve this check. The sender name could be spoofed
        assert(request.header.sender in collaborator_common_names), \
                'Collaborator {} is unrecognized by the Aggregator'.format(request.header.sender)
        assert(request.header.receiver == self.uuid), \
                'Aggregator UUID does not match'.format(request.header.receiver)
        assert(request.header.federation_uuid == self.federation_uuid), \
                'Federation UUID is unrecognized'.format(request.header.federation_uuid)

    def get_header(self,collaborator_name):
        """
        Compose and return MessageHeader
        """
        return MessageHeader(sender=self.uuid, receiver=collaborator_name, federation_uuid=self.federation_uuid)

    def get_sleep_time(self):
        """
        Sleep 10 seconds
        """
        return 10
        
    def GetTasks(self, request):
        # all messages get sanity checked
        self.check_request(request)

        collaborator_name = request.header.sender

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
        if len(tasks) == 0:
            return TasksResponse(header=self.get_header(collaborator_name),
                                 round_number=self.round_number,
                                 tasks=None,
                                 sleep_time=self.get_sleep_time(), # this could be an extensible function if we want
                                 quit=False)

        # if we do have tasks, remove any that we already have results for
        tasks = [t for t in tasks if not self.collaborator_task_completed(collaborator_name, t, self.round_number)]

        #Do the check again because it's possible that all tasks have been completed
        if len(tasks) == 0:
            return TasksResponse(header=self.get_header(collaborator_name),
                                 round_number=self.round_number,
                                 tasks=None,
                                 sleep_time=self.get_sleep_time(), # this could be an extensible function if we want
                                 quit=False)


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
        tags                = request.tags


        tensor_key = TensorKey(tensor_name, self.uuid, round_number,tags)
        collaborator_weight_dict = ???
        nparray = self.tensor_db.get_aggregated_tensor(key,collaborator_weight_dict)
        if nparray is None:
            raise ValueError("Aggregator does not have an aggregated tensor for {}".format(k))

        # quite a bit happens in here, including decompression, delta handling, etc...
        # we might want to cache these as well
        named_tensor = self.nparray_to_named_tensor(nparray)

        return TensorResponse(header=self.get_header(collaborator_name),
                              round_number=round_number,
                              tensor=named_tensor)


    def collaborator_task_completed(collaborator_name, task_name, round_number):
        """
        Check if the collaborator has completed the task for the round. 
        The aggregator doesn't actually know which tensors should be sent from the collaborator
        so it must to rely specifically on the presence of previous results

        Parameters
        ----------
        collaborator_name
        task_name
        round_number

        Returns
        -------
        bool

        """
        return self.collaborator_task_completion_status[collaborator_name][round_number][task_name]

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

