# Copyright (C) 2020 Intel Corporation

import time
import os
import logging

import numpy as np
import tensorboardX
from threading import Lock

from .. import check_equal, check_not_equal, check_is_in, check_not_in
from ..proto.collaborator_aggregator_interface_pb2 import MessageHeader
from ..proto.collaborator_aggregator_interface_pb2 import Job, JobRequest, JobReply
from ..proto.collaborator_aggregator_interface_pb2 import JOB_DOWNLOAD_MODEL, JOB_QUIT, JOB_TRAIN, JOB_VALIDATE, JOB_YIELD
from ..proto.collaborator_aggregator_interface_pb2 import ModelProto, CompressedModelProto, ModelHeader, TensorProto
from ..proto.collaborator_aggregator_interface_pb2 import ModelDownloadRequest, GlobalModelUpdate
from ..proto.collaborator_aggregator_interface_pb2 import LocalModelUpdate, LocalValidationResults, LocalModelUpdateAck, LocalValidationResultsAck


from tfedlrn.proto.protoutils import load_proto, dump_proto

# FIXME: simpler "stats" collection/handling
# FIXME: remove the round tracking/job-result tracking stuff from this?
# Feels like it conflates model aggregation with round management
# FIXME: persistence of the trained weights.
class Aggregator(object):
    """An Aggregator is the centra node in federated learning.
    
    Parameters
    ----------
    id : str
        Aggregation ID.
    fed_id : str
        Federation ID.
    col_ids : list of str
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
    # FIXME: no selector logic is in place
    def __init__(self, 
                 agg_id, 
                 fed_id, 
                 col_ids, 
                 init_model_fpath, 
                 latest_model_fpath, 
                 best_model_fpath, 
                 rounds_to_train=256, 
                 disable_equality_check=False,
                 custom_update_pipeline=None, 
                 **kwargs):
        self.logger = logging.getLogger(__name__)
        self.id = agg_id
        self.fed_id = fed_id
        self.model = load_proto(init_model_fpath)
        self.latest_model_fpath = latest_model_fpath
        self.best_model_fpath = best_model_fpath
        self.col_ids = col_ids
        self.round_num = 1
        self.rounds_to_train = rounds_to_train
        self.quit_job_sent_to = []
        self.disable_equality_check = disable_equality_check

        #FIXME: close the handler before termination.
        log_dir = './logs/tensorboardX/%s_%s' % (self.id, self.fed_id)
        self.tb_writer = tensorboardX.SummaryWriter(log_dir, flush_secs=10)

        self.model_update_in_progress = None

        self.init_per_col_round_stats()
        self.best_model_score = None
        self.mutex = Lock()

        self.custom_update_pipeline = custom_update_pipeline

    def all_quit_jobs_sent(self):
        return sorted(self.quit_job_sent_to) == sorted(self.col_ids)

    def validate_header(self, message):
        # validate that the message is for me
        check_equal(message.header.recipient, self.id, self.logger)
        
        # validate that the message is for my federation
        check_equal(message.header.federation_id, self.fed_id, self.logger)
        
        # validate that the sender is one of my collaborators
        check_is_in(message.header.sender, self.col_ids, self.logger)


    def init_per_col_round_stats(self):
        """Initalize the metrics from collaborators for each round of aggregation. """
        keys = ["loss_results", "collaborator_training_sizes", "agg_validation_results", "preagg_validation_results", "collaborator_validation_sizes"]
        values = [{} for i in range(len(keys))]
        self.per_col_round_stats = dict(zip(keys, values))

    def end_of_round_check(self):
        # FIXME: find a nice, clean way to manage these values without having to manually ensure
        # the keys are in sync

        # assert our dictionary keys are in sync
        check_equal(self.per_col_round_stats["loss_results"].keys(), self.per_col_round_stats["collaborator_training_sizes"].keys(), self.logger)
        check_equal(self.per_col_round_stats["agg_validation_results"].keys(), self.per_col_round_stats["collaborator_validation_sizes"].keys(), self.logger)

        done = True

        # ensure we have results from all collaborators
        # this only works this way because all collaborators have the same jobs every round
        for c in self.col_ids:
            if (c not in self.per_col_round_stats["loss_results"] or 
                c not in self.per_col_round_stats["collaborator_training_sizes"] or 
                c not in self.per_col_round_stats["agg_validation_results"] or
                c not in self.per_col_round_stats["preagg_validation_results"] or
                c not in self.per_col_round_stats["collaborator_validation_sizes"]):
                done = False
                break

        if done:
            self.end_of_round()
            self.round_num += 1
            self.logger.debug("Start a new round %d." % self.round_num)

    def end_of_round(self):
        # FIXME: what all should we do to track results/metrics? It should really be an easy, extensible solution

        # compute the weighted loss average
        round_loss = np.average([self.per_col_round_stats["loss_results"][c] for c in self.col_ids],
                                weights=[self.per_col_round_stats["collaborator_training_sizes"][c] for c in self.col_ids])

        # compute the weighted validation average
        round_val = np.average([self.per_col_round_stats["agg_validation_results"][c] for c in self.col_ids],
                               weights=[self.per_col_round_stats["collaborator_validation_sizes"][c] for c in self.col_ids])

        # FIXME: proper logging
        self.logger.info('round results for model id/version {}/{}'.format(self.model.header.id, self.model.header.version))
        self.logger.info('\tvalidation: {}'.format(round_val))
        self.logger.info('\tloss: {}'.format(round_loss))

        self.tb_writer.add_scalars('training/loss', {**self.per_col_round_stats["loss_results"], "federation": round_loss}, global_step=self.round_num)
        self.tb_writer.add_scalars('training/size', self.per_col_round_stats["collaborator_training_sizes"], global_step=self.round_num)
        self.tb_writer.add_scalars('validation/preagg_result', self.per_col_round_stats["preagg_validation_results"], global_step=self.round_num)
        self.tb_writer.add_scalars('validation/size', self.per_col_round_stats["collaborator_validation_sizes"], global_step=self.round_num-1)
        self.tb_writer.add_scalars('validation/agg_result', {**self.per_col_round_stats["agg_validation_results"], "federation": round_val}, global_step=self.round_num-1)

        # copy over the model update in progress
        self.model = self.model_update_in_progress

        # increment the version
        self.model.header.version += 1

        # Save to file.
        dump_proto(self.model, self.latest_model_fpath)

        model_score = round_val
        if self.best_model_score is None or self.best_model_score < model_score:
            self.logger.info("Saved the best model with score {:f}.".format(model_score))
            self.best_model_score = model_score
            dump_proto(self.model, self.best_model_fpath)

        # clear the update pointer
        self.model_update_in_progress = None

        self.init_per_col_round_stats()

    def UploadLocalModelUpdate(self, message):
        self.mutex.acquire(blocking=True)
        try:
            t = time.time()
            self.validate_header(message)

            self.logger.info("Receive model update from %s " % message.header.sender)
            if self.custom_update_pipeline is not None:
                model_proto = self.custom_update_pipeline.protos_to_tensors(compressed_tensor_protos=reply.model.compressed_tensors, 
                                                                            meta_data=reply.model.metadata)
                model_proto
            else:
                model_proto = message.model
            model_header = model_proto.header

            # validate this model header
            check_equal(model_header.id, self.model.header.id, self.logger)
            check_equal(model_header.version, self.model.header.version, self.logger)

            # ensure we haven't received an update from this collaborator already
            check_not_in(message.header.sender, self.per_col_round_stats["loss_results"], self.logger)
            check_not_in(message.header.sender, self.per_col_round_stats["collaborator_training_sizes"], self.logger)

            # if this is our very first update for the round, we take this model as-is
            # FIXME: move to model deltas, add with original to reconstruct
            # FIXME: this really only works with a trusted collaborator. Sanity check this against self.model
            if self.model_update_in_progress is None:
                self.model_update_in_progress = model_proto

            # otherwise, we compute the streaming weighted average
            else:
                # get the current update size total
                total_update_size = np.sum(list(self.per_col_round_stats["collaborator_training_sizes"].values()))

                # compute the weights for the global vs local tensors for our streaming average
                weight_g = total_update_size / (message.data_size + total_update_size)
                weight_l = message.data_size / (message.data_size + total_update_size)

                # The model parameters are represented in float32 and will be transmitted in byte stream.
                weight_g = weight_g.astype(np.float32)
                weight_l = weight_l.astype(np.float32)

                # FIXME: right now we're really using names just to sanity check consistent ordering

                # assert that the models include the same number of tensors
                check_equal(len(self.model_update_in_progress.tensors), len(model_proto.tensors), self.logger)

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

                    check_not_equal(l, None, self.logger)
                    
                    # sanity check that the tensors are indeed different for non opt tensors 
                    # TODO: modify this to better comprehend for non pytorch how to identify the opt portion (use model opt info?)               
                    if (not self.disable_equality_check \
                        and not g.name.startswith('__opt') \
                        and 'RMSprop' not in g.name \
                        and 'Adam' not in g.name \
                        and 'RMSProp' not in g.name):
                        check_not_equal(g.npbytes, l.npbytes, self.logger, name=g.name)
                        
                    if g.shape != l.shape:
                        raise ValueError('global tensor shape {} of {} not equal to local tensor shape {} of {}'.format(g.shape, g.name, l.shape, l.name))

                    # now just weighted average these
                    g_values = np.frombuffer(g.npbytes, dtype=np.float32)
                    l_values = np.frombuffer(l.npbytes, dtype=np.float32)
                    new_values = np.average([g_values, l_values], weights=[weight_g, weight_l], axis=0)
                    del g_values, l_values
                    # FIXME: we shouldn't convert it to bytes until we are ready to send it back to the collaborators.
                    self.model_update_in_progress.tensors[i].npbytes = new_values.tobytes('C')

            # store the loss results and training update size
            self.per_col_round_stats["loss_results"][message.header.sender] = message.loss
            self.per_col_round_stats["collaborator_training_sizes"][message.header.sender] = message.data_size

            # return LocalModelUpdateAck
            self.logger.debug("Complete model update from %s " % message.header.sender)
            reply = LocalModelUpdateAck(header=self.create_reply_header(message))

            self.end_of_round_check()

            self.logger.debug('aggregator handled UploadLocalModelUpdate in time {}'.format(time.time() - t))
        finally:
            self.mutex.release()

        return reply

    def UploadLocalMetricsUpdate(self, message):
        self.mutex.acquire(blocking=True)
        try:
            t = time.time()
            self.validate_header(message)

            self.logger.debug("Receive local validation results from %s " % message.header.sender)
            model_header = message.model_header

            # validate this model header
            check_equal(model_header.id, self.model.header.id, self.logger)
            check_equal(model_header.version, self.model.header.version, self.logger)
            
            sender = message.header.sender

            if sender not in self.per_col_round_stats["agg_validation_results"]:
                # Pre-train validation
                # ensure we haven't received an update from this collaborator already
                # FIXME: is this an error case that should be handled?
                check_not_in(message.header.sender, self.per_col_round_stats["agg_validation_results"], self.logger)
                check_not_in(message.header.sender, self.per_col_round_stats["collaborator_validation_sizes"], self.logger)
                
                # store the validation results and validation size
                self.per_col_round_stats["agg_validation_results"][message.header.sender] = message.results
                self.per_col_round_stats["collaborator_validation_sizes"][message.header.sender] = message.data_size
            elif sender not in self.per_col_round_stats["preagg_validation_results"]:
                # Post-train validation
                check_not_in(message.header.sender, self.per_col_round_stats["preagg_validation_results"], self.logger)
                self.per_col_round_stats["preagg_validation_results"][message.header.sender] = message.results

            reply = LocalValidationResultsAck(header=self.create_reply_header(message))

            self.end_of_round_check()

            self.logger.debug('aggregator handled UploadLocalMetricsUpdate in time {}'.format(time.time() - t))
        finally:
            self.mutex.release()

        self.logger.debug('aggregator handled UploadLocalMetricsUpdate in time {}'.format(time.time() - t))

        return reply

    def RequestJob(self, message):
        t = time.time()
        self.validate_header(message)

        # FIXME: we should really have each collaborator validate one last time
        # check if we are done
        if self.round_num > self.rounds_to_train:
            job = JOB_QUIT
            self.quit_job_sent_to.append(message.header.sender)
        # FIXME: this flow needs to depend on a job selection output for the round
        # for now, all jobs require and in-sync model, so it is the first check
        # check if the sender model is out of date
        elif self.collaborator_out_of_date(message.model_header):
            job = JOB_DOWNLOAD_MODEL
        # else, check if this collaborator has not sent validation results
        elif message.header.sender not in self.per_col_round_stats["agg_validation_results"]:
            job = JOB_VALIDATE
        # else, check if this collaborator has not sent training results
        elif message.header.sender not in self.per_col_round_stats["collaborator_training_sizes"]:
            job = JOB_TRAIN
        elif message.header.sender not in self.per_col_round_stats["preagg_validation_results"]:
            job = JOB_VALIDATE
        # else this collaborator is done for the round
        else:
            job = JOB_YIELD
        
        self.logger.debug("Receive job request from %s and assign with %s" % (message.header.sender, job))

        reply = JobReply(header=self.create_reply_header(message), job=job)

        if reply.job is not JOB_YIELD:
            self.logger.debug('aggregator handled RequestJob in time {}'.format(time.time() - t))
        
        return reply       

    def DownloadModel(self, message):
        t = time.time()
        self.validate_header(message)

        self.logger.info("Received model download request from %s " % message.header.sender)

        # check if the models don't match
        if not(self.collaborator_out_of_date(message.model_header)):
            statement = "Assertion failed: self.collaborator_out_of_date(message.model_header)"
            self.logger.exception(statement)
            raise RuntimeError(statement)

        reply = GlobalModelUpdate(header=self.create_reply_header(message), model=self.model)

        self.logger.debug('aggregator handled RequestJob in time {}'.format(time.time() - t))
        
        return reply       

    def create_reply_header(self, message):
        return MessageHeader(sender=self.id, recipient=message.header.sender, federation_id=self.fed_id, counter=message.header.counter)

    def collaborator_out_of_date(self, model_header):
        # validate that this is the right model to be checking
        check_equal(model_header.id, self.model.header.id, self.logger)
        
        return model_header.version != self.model.header.version
