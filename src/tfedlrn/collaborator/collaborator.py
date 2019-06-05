#!/usr/bin/env python3
import sys
import os

from math import ceil

import time
import argparse

import numpy as np

from tfedlrn.proto.message_pb2 import *


class Collaborator(object):

    def __init__(self, id, agg_id, fed_id, wrapped_model, connection, model_id, model_version, polling_interval=4):
        self.connection = connection
        self.polling_interval = 4

        # this stuff is really about sanity/correctness checking to ensure the bookkeeping and control flow is correct
        self.message_header = MessageHeader(sender=id, recipient=agg_id, federation_id=fed_id, counter=0)
        self.model_header = ModelHeader(id=model_id, version=model_version)

        self.wrapped_model = wrapped_model


    def send(self, message):
        # set the message header
        message.header = self.message_header

        self.connection.send(message)
        reply = self.connection.receive()

        # validate the message pair

        # check message is from my agg to me
        assert reply.header.sender == self.message_header.recipient and reply.header.recipient == self.message_header.sender

        # check that the federation id matches
        assert reply.header.federation_id == self.message_header.federation_id

        # check that the counters match
        assert reply.header.counter == self.message_header.counter

        # now we can increment our counter
        self.message_header.counter += 1

        return reply


    def run(self):
        while True:
            # query for job
            job = self.query_for_job()

            # if time to quit
            if job is JOB_QUIT:
                print(f'{self} quitting')
                break
            elif job is JOB_TRAIN:
                self.do_train_job()
            elif job is JOB_YIELD:                
                self.do_yield_job()
            elif job is JOB_VALIDATE:
                self.do_validate_job()
            elif job is JOB_DOWNLOAD_MODEL:
                self.do_download_model_job()

    def query_for_job(self):
        reply = self.send(JobRequest(model_header=self.model_header))

        assert isinstance(reply, JobReply)

        return reply.Job

    def do_train_job(self):

        # get the initial tensor dict
        initial_tensor_dict = self.wrapped_model.get_tensor_dict()

        # train the model
        loss = self.wrapped_model.train_epoch()

        # get the training data size
        data_size = self.wrapped_model.get_training_data_size()

        # get the trained tensor dict and convert it to a delta
        tensor_dict = self.wrapped_model.get_tensor_dict()
        for k in tensor_dict.keys():
            tensor_dict[k] -= initial_tensor_dict[k]

        # create the tensor proto list
        tensor_protos = []
        for k, v in tensor_dict.items():
            tensor_protos.append(TensorProto(name=k, shape=v.shape, values=v.flatten(order='C')))

        model_proto = ModelProto(header=self.model_header, tensors=tensor_protos)

        reply = self.send(LocalModelUpdate(model=model_proto, data_size=data_size, loss=loss))
        assert isinstance(reply, LocalModelUpdateAck)

    def do_yield_job(self):
        time.sleep(self.polling_interval)

    def do_validate_job(self):
        results = self.wrapped_model.validate()
        data_size = self.wrapped_model.get_validation_data_size()

        reply = self.send(LocalValidationResults(model_header=self.model_header, results=results, data_size=data_size))
        assert isinstance(reply, LocalValidationResultsAck)

    def do_download_model_job(self):
        # sanity check on version is implicit in send
        reply = self.send(ModelDownloadRequest(model_header=self.model_header))

        assert isinstance(reply, GlobalModelUpdate)

        # ensure we actually got a new model version
        assert reply.model.header.version != message.model_header.version

        # set our model header
        self.model_header = reply.model.header

        # create the tensor dict
        tensor_dict = {}
        for tensor_proto in reply.model.tensors:
            tensor_dict[tensor_proto.name] = np.array(tensor_proto.shape, buffer=tensor_proto.values, order='C')

        self.wrapped_model.set_tensors_from_dict(tensor_dict)
