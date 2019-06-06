import numpy as np

from tfedlrn.proto.message_pb2 import *


class Aggregator(object):

    # FIXME: no selector logic is in place

    def __init__(self, id, fed_id, col_ids, connection, model):
        self.connection = connection
        self.id = id
        self.fed_id = fed_id
        self.model = model
        self.col_ids = col_ids

        self.loss_results = {}
        self.collaborator_training_sizes = {}
        self.validation_results = {}
        self.collaborator_validation_sizes = {}

    def end_of_round_check(self):
        # FIXME: find a nice, clean way to manage these values without having to manually ensure
        # the keys are in sync

        # assert our dictionary keys are in sync
        assert self.loss_results.keys() == self.collaborator_training_sizes.keys()
        assert self.validation_results.keys() == self.collaborator_validation_sizes.keys()

        done = True

        # ensure we have results from all collaborators
        for c in self.col_ids:
            if (c not in self.loss_results or 
                c not in self.collaborator_training_sizes or 
                c not in self.validation_results or 
                c not in self.collaborator_validation_sizes):
                done = False
                break

        if done:
            self.end_of_round()

    def end_of_round(self):
        # FIXME: what all should we do to track results/metrics? It should really be an easy, extensible solution

        # compute the weighted loss average
        round_loss = np.average([self.loss_results[c] for c in self.col_ids],
                                weights=[self.collaborator_training_sizes[c] for c in self.col_ids])

        # compute the weighted validation average
        round_val = np.average([self.validation_results[c] for c in self.col_ids],
                               weights=[self.collaborator_validation_sizes[c] for c in self.col_ids])

        # FIXME: proper logging
        print('round results for model id/version {}/{}'.format(self.model.header.id, self.model.header.version))
        print('\tvalidation: {}'.format(round_val))
        print('\tloss: {}'.format(round_loss))

        self.model.header.version += 1

        self.loss_results = {}
        self.collaborator_training_sizes = {}
        self.validation_results = {}
        self.collaborator_validation_sizes = {}

    def run(self):
        while True:
            # receive a message
            message = self.connection.receive()

            # validate that the message is for me
            assert message.header.recipient == self.id

            # validate that the message is for my federation
            assert message.header.federation_id == self.fed_id

            # validate that the sender is one of my collaborators
            assert message.header.sender in self.col_ids

            if isinstance(message, LocalModelUpdate):
                reply = self.handle_local_model_update(message)
            elif isinstance(message, LocalValidationResults):
                reply = self.handle_local_validation_results(message)
            elif isinstance(message, JobRequest):
                reply = self.handle_job_request(message)
            elif isinstance(message, ModelDownloadRequest):
                reply = self.handle_model_download_request(message)

            self.connection.send(reply)

            # do end of round check
            self.end_of_round_check()

    def handle_local_model_update(self, message):
        model_proto = message.model
        model_header = model_proto.header

        # validate this model header
        assert model_header.id == self.model.header.id
        assert model_header.version == self.model.header.version

        # ensure we haven't received an update from this collaborator already
        assert message.header.sender not in self.loss_results
        assert message.header.sender not in self.collaborator_training_sizes        

        # get the current update size total
        total_update_size = np.sum(list(self.collaborator_training_sizes.values()))

        # if this is our very first update for the round, we take this model as-is
        if total_update_size == 0:
            self.model = model_proto

        # otherwise, we compute the weighted average
        else:
            # compute the weights for the global vs local tensors for our streaming average
            weight_g = total_update_size / (message.data_size + total_update_size)
            weight_l = message.data_size / (message.data_size + total_update_size)

            # FIXME: right now we're really using names just to sanity check consistent ordering

            # assert that the models include the same number of tensors
            assert len(self.model.tensors) == len(model_proto.tensors)

            # aggregate all the model tensors in the protobuf
            # this is a streaming average
            for i in range(len(self.model.tensors)):
                # global tensor
                g = self.model.tensors[i]

                # find the local collaborator tensor
                for l in model_proto.tensors:
                    if l.name == g.name:
                        break

                # validate that these are the same tensor in the model (e.g. weights for a specific layer)
                if g.name != l.name:
                    print([self.model.tensors[j].name for j in range(len(self.model.tensors))])
                    print([model_proto.tensors[j].name for j in range(len(self.model.tensors))])
                    raise ValueError('global tensor name {} not equal to local tensor name {}'.format(g.name, l.name))
                    
                if g.shape != l.shape:
                    raise ValueError('global tensor shape {} of {} not equal to local tensor shape {} of {}'.format(g.shape, g.name, l.shape, l.name))

                # now just weighted average these
                new_values = np.average([g.values, l.values], weights=[weight_g, weight_l])
                del self.model.tensors[i].values[:]
                self.model.tensors[i].values.extend(new_values)

        # store the loss results and training update size
        self.loss_results[message.header.sender] = message.loss
        self.collaborator_training_sizes[message.header.sender] = message.data_size

        # return LocalModelUpdateAck
        return LocalModelUpdateAck(header=self.create_reply_header(message))

    def handle_local_validation_results(self, message):
        model_header = message.model_header

        # validate this model header
        assert model_header.id == self.model.header.id
        assert model_header.version == self.model.header.version

        # ensure we haven't received an update from this collaborator already
        assert message.header.sender not in self.validation_results
        assert message.header.sender not in self.collaborator_validation_sizes        

        # store the validation results and validation size
        self.validation_results[message.header.sender] = message.results
        self.collaborator_validation_sizes[message.header.sender] = message.data_size

        # return LocalValidationResultsAck
        return LocalValidationResultsAck(header=self.create_reply_header(message))

    def handle_job_request(self, message):
        # check if the sender model is out of date
        if self.collaborator_out_of_date(message.model_header):
            job = JOB_DOWNLOAD_MODEL
        # else, check if this collaborator has not sent validation results
        elif message.header.sender not in self.collaborator_validation_sizes:
            job = JOB_VALIDATE
        # else, check if this collaborator has not sent training results
        elif message.header.sender not in self.collaborator_training_sizes:
            job = JOB_TRAIN
        # else this collaborator is done for the round
        else:
            job = JOB_YIELD

        return JobReply(header=self.create_reply_header(message), job=job)

    def handle_model_download_request(self, message):
        # assert that the models don't match
        assert self.collaborator_out_of_date(message.model_header)

        return GlobalModelUpdate(header=self.create_reply_header(message), model=self.model)

    def create_reply_header(self, message):
        return MessageHeader(sender=self.id, recipient=message.header.sender, federation_id=self.fed_id, counter=message.header.counter)

    def collaborator_out_of_date(self, model_header):
        # validate that this is the right model to be checking        
        assert model_header.id == self.model.header.id

        return model_header.version != self.model.header.version
