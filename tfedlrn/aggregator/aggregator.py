import time
import os

import numpy as np
import tensorflow as tf
import tensorboard.summary as tb_summary

from tfedlrn.proto.message_pb2 import *


# FIXME: simpler "stats" collection/handling
# FIXME: remove the round tracking/job-result tracking stuff from this?
# Feels like it conflates model aggregation with round management
class Aggregator(object):

    # FIXME: no selector logic is in place

    def __init__(self, id, fed_id, col_ids, connection, model):
        self.connection = connection
        self.id = id
        self.fed_id = fed_id
        self.model = model
        self.col_ids = col_ids
        self.round_num = 0

        #FIXME: close the handler before termination.
        log_dir = './logs/%s_%s' % (self.id, self.fed_id)
        self.tb_writers = {c:tf.summary.FileWriter(os.path.join(log_dir, 'plot_'+c)) for c in self.col_ids}
        self.tb_writers['federation'] = tf.summary.FileWriter(os.path.join(log_dir, 'plot_federation'))

        self.model_update_in_progress = None

        # these are per collaborator, for the current round
        # FIXME: call these "per_col_round_stats"
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
        # this only works this way because all collaborators have the same jobs every round
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
        # FIXME: log this to debug is good, but where does this output ultimately go?
        print('round results for model id/version {}/{}'.format(self.model.header.id, self.model.header.version))
        print('\tvalidation: {}'.format(round_val))
        print('\tloss: {}'.format(round_loss))

        for c in self.col_ids:
            self.tb_writers[c].add_summary(tb_summary.scalar_pb('training/loss', self.loss_results[c]), global_step=self.round_num)
            self.tb_writers[c].add_summary(tb_summary.scalar_pb('training/size', self.collaborator_training_sizes[c]), global_step=self.round_num)
            self.tb_writers[c].add_summary(tb_summary.scalar_pb('validation/result', self.validation_results[c]), global_step=self.round_num)
            self.tb_writers[c].add_summary(tb_summary.scalar_pb('validation/size', self.collaborator_validation_sizes[c]), global_step=self.round_num)
            self.tb_writers[c].flush()
        self.tb_writers['federation'].add_summary(tb_summary.scalar_pb('training/loss', round_loss), global_step=self.round_num)
        self.tb_writers['federation'].add_summary(tb_summary.scalar_pb('validation/result', round_val), global_step=self.round_num)
        self.tb_writers['federation'].flush()

        # copy over the model update in progress
        self.model = self.model_update_in_progress

        # increment the version
        self.model.header.version += 1

        # clear the update pointer
        self.model_update_in_progress = None

        self.loss_results = {}
        self.collaborator_training_sizes = {}
        self.validation_results = {}
        self.collaborator_validation_sizes = {}

    def run(self):
        while True:
            # receive a message
            message = self.connection.receive()
            t = time.time()

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

            if not isinstance(reply, JobReply) or reply.job is not JOB_YIELD:
                print('aggregator handled {} in time {}'.format(message.__class__.__name__, time.time() - t))

            self.round_num += 1

    def handle_local_model_update(self, message):
        model_proto = message.model
        model_header = model_proto.header

        # validate this model header
        assert model_header.id == self.model.header.id
        assert model_header.version == self.model.header.version

        # ensure we haven't received an update from this collaborator already
        assert message.header.sender not in self.loss_results
        assert message.header.sender not in self.collaborator_training_sizes        

        # if this is our very first update for the round, we take this model as-is
        # FIXME: move to model deltas, add with original to reconstructf
        # FIXME: this really only works with a trusted collaborator. Sanity check this against self.model
        if self.model_update_in_progress is None:
            self.model_update_in_progress = model_proto

        # otherwise, we compute the streaming weighted average
        else:            
            # get the current update size total
            total_update_size = np.sum(list(self.collaborator_training_sizes.values()))

            # compute the weights for the global vs local tensors for our streaming average
            weight_g = total_update_size / (message.data_size + total_update_size)
            weight_l = message.data_size / (message.data_size + total_update_size)

            # FIXME: right now we're really using names just to sanity check consistent ordering

            # assert that the models include the same number of tensors
            assert len(self.model_update_in_progress.tensors) == len(model_proto.tensors)

            # aggregate all the model tensors in the protobuf
            # this is a streaming average
            for i in range(len(self.model_update_in_progress.tensors)):
                # global tensor
                g = self.model_update_in_progress.tensors[i]

                # find the local collaborator tensor
                l = None
                for local_tensor in model_proto.tensors:
                    if local_tensor.name == g.name:
                        l = local_tensor
                        break

                assert l is not None

                # sanity check that the tensors are indeed different for non opt tensors                
                if (not g.name.startswith('__opt')) and (g.values == l.values):
                    raise ValueError('global tensor {} exactly equal to local tensor {}'.format(g.name, l.name))
                    
                if g.shape != l.shape:
                    raise ValueError('global tensor shape {} of {} not equal to local tensor shape {} of {}'.format(g.shape, g.name, l.shape, l.name))

                # now just weighted average these
                new_values = np.average([g.values, l.values], weights=[weight_g, weight_l], axis=0)
                del self.model_update_in_progress.tensors[i].values[:]
                self.model_update_in_progress.tensors[i].values.extend(new_values)

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
        # FIXME: is this an error case that should be handled?
        assert message.header.sender not in self.validation_results
        assert message.header.sender not in self.collaborator_validation_sizes        

        # store the validation results and validation size
        self.validation_results[message.header.sender] = message.results
        self.collaborator_validation_sizes[message.header.sender] = message.data_size

        # return LocalValidationResultsAck
        return LocalValidationResultsAck(header=self.create_reply_header(message))

    def handle_job_request(self, message):
        # FIXME: this flow needs to depend on a job selection output for the round
        # for now, all jobs require and in-sync model, so it is the first check
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
