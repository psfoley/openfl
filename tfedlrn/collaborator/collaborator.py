import time

import numpy as np

from ..proto.message_pb2 import *


# FIXME: this is actually a tuple of a collaborator/flplan
# CollaboratorFLPlanExecutor?
class Collaborator(object):

    # FIXME: do we need a settable model version? Shouldn't col always start assuming out of sync?
    def __init__(self, id, agg_id, fed_id, wrapped_model, connection, model_version, polling_interval=4):
        self.connection = connection
        self.polling_interval = 4

        # this stuff is really about sanity/correctness checking to ensure the bookkeeping and control flow is correct
        self.id = id
        self.agg_id = agg_id
        self.fed_id = fed_id
        self.counter = 0
        self.model_header = ModelHeader(id=wrapped_model.__class__.__name__,
                                        version=model_version)

        self.wrapped_model = wrapped_model

    def create_message_header(self):
        header = MessageHeader(sender=self.id, recipient=self.agg_id, federation_id=self.fed_id, counter=self.counter)
        return header

    def __repr__(self):
        return 'collaborator {} of federation {}'.format(self.id, self.fed_id)

    def __str__(self):
        return self.__repr__()

    # FIXME: rename to send_req_rcv_reply
    def send_and_receive(self, message):
        self.connection.send(message)
        reply = self.connection.receive()

        # validate the message pair

        # check message is from my agg to me
        assert reply.header.sender == self.agg_id and reply.header.recipient == self.id

        # check that the federation id matches
        assert reply.header.federation_id == self.fed_id

        # check that the counters match
        assert reply.header.counter == self.counter

        # increment our counter
        self.counter += 1

        return reply

    def run(self):
        while True:
            # query for job
            # returns when a job has been received
            job = self.query_for_job()

            print(self, 'got job', Job.Name(job))

            # if time to quit
            if job is JOB_QUIT:
                print(self, 'quitting')
                break
            elif job is JOB_TRAIN:
                self.do_train_job()
            elif job is JOB_VALIDATE:
                self.do_validate_job()
            elif job is JOB_DOWNLOAD_MODEL:
                self.do_download_model_job()

    def query_for_job(self):
        # loop until we get a job other than 'yield'
        while True:
            reply = self.send_and_receive(JobRequest(header=self.create_message_header(), model_header=self.model_header))

            assert isinstance(reply, JobReply)
            if reply.job is not JOB_YIELD:
                break
            time.sleep(self.polling_interval)

        return reply.job

    def do_train_job(self):

        # get the initial tensor dict
        # initial_tensor_dict = self.wrapped_model.get_tensor_dict()

        # train the model
        # FIXME: model header "version" needs to be changed to "rounds_trained"
        loss = self.wrapped_model.train_epoch(epoch=self.model_header.version)

        # get the training data size
        data_size = self.wrapped_model.get_training_data_size()

        # get the trained tensor dict
        tensor_dict = self.wrapped_model.get_tensor_dict()

        # convert to a delta
        # for k in tensor_dict.keys():
        #     tensor_dict[k] -= initial_tensor_dict[k]

        # create the tensor proto list
        tensor_protos = []
        for k, v in tensor_dict.items():
            tensor_protos.append(TensorProto(name=k, shape=v.shape, values=v.flatten(order='C')))

        model_proto = ModelProto(header=self.model_header, tensors=tensor_protos)

        reply = self.send_and_receive(LocalModelUpdate(header=self.create_message_header(), model=model_proto, data_size=data_size, loss=loss))
        assert isinstance(reply, LocalModelUpdateAck)

    def do_validate_job(self):
        results = self.wrapped_model.validate()
        data_size = self.wrapped_model.get_validation_data_size()

        reply = self.send_and_receive(LocalValidationResults(header=self.create_message_header(), model_header=self.model_header, results=results, data_size=data_size))
        assert isinstance(reply, LocalValidationResultsAck)

    def do_download_model_job(self):
        # sanity check on version is implicit in send
        reply = self.send_and_receive(ModelDownloadRequest(header=self.create_message_header(), model_header=self.model_header))

        assert isinstance(reply, GlobalModelUpdate)

        # ensure we actually got a new model version
        assert reply.model.header.version != self.model_header.version

        # set our model header
        self.model_header = reply.model.header

        # create the tensor dict
        tensor_dict = {}
        for tensor_proto in reply.model.tensors:
            tensor_dict[tensor_proto.name] = np.array(tensor_proto.values, order='C').reshape(tensor_proto.shape)

        self.wrapped_model.set_tensor_dict(tensor_dict)
