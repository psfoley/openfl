# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import time
import logging
import numpy as np

from .. import check_type, check_equal, check_not_equal, split_tensor_dict_for_holdouts
from ..proto.collaborator_aggregator_interface_pb2 import MessageHeader
from ..proto.collaborator_aggregator_interface_pb2 import Job, JobRequest, JobReply
from ..proto.collaborator_aggregator_interface_pb2 import JOB_DOWNLOAD_MODEL, JOB_QUIT, JOB_TRAIN, JOB_VALIDATE, JOB_YIELD
from ..proto.collaborator_aggregator_interface_pb2 import ModelProto, ModelHeader, TensorProto
from ..proto.collaborator_aggregator_interface_pb2 import ModelDownloadRequest, GlobalModelUpdate
from ..proto.collaborator_aggregator_interface_pb2 import LocalModelUpdate, LocalModelUpdateAck
from ..proto.collaborator_aggregator_interface_pb2 import LocalValidationResults, LocalValidationResultsAck


from enum import Enum

class OptTreatment(Enum):
    RESET = 1
    EDGE = 2
    AGG = 3


# FIXME: this is actually a tuple of a collaborator/flplan
# CollaboratorFLPlanExecutor?
class Collaborator(object):
    """The current class is not good for local test without channel. """
    # FIXME: do we need a settable model version? Shouldn't col always start assuming out of sync?
    def __init__(self, col_id, agg_id, fed_id, wrapped_model, channel, polling_interval=4, opt_treatment="AGG", **kwargs):
        self.logger = logging.getLogger(__name__)
        self.channel = channel
        self.polling_interval = polling_interval

        # this stuff is really about sanity/correctness checking to ensure the bookkeeping and control flow is correct
        self.id = col_id
        self.agg_id = agg_id
        self.fed_id = fed_id
        self.counter = 0
        self.model_header = ModelHeader(id=wrapped_model.__class__.__name__,
                                        version=-1)

        self.wrapped_model = wrapped_model
        self.tensor_dict_split_fn_kwargs = wrapped_model.tensor_dict_split_fn_kwargs or {}

        # AGG/EDGE/RESET
        if hasattr(OptTreatment, opt_treatment):
            self.opt_treatment = OptTreatment[opt_treatment]
        else:
            self.logger.error("Unknown opt_treatment: %s." % opt_treatment)
            raise NotImplementedError("Unknown opt_treatment: %s." % opt_treatment)

        # FIXME: this is a temporary fix for non-float values and other named params designated to hold out from aggregation. 
        # Needs updated when we have proper collab-side state saving.
        self._remove_and_save_holdout_params(self.wrapped_model.get_tensor_dict(with_opt_vars=self._with_opt_vars()))

    def _remove_and_save_holdout_params(self, tensor_dict):
        tensors_to_send, self.holdout_params = split_tensor_dict_for_holdouts(self.logger, tensor_dict, **self.tensor_dict_split_fn_kwargs)
        if self.holdout_params != {}:
            self.logger.debug("{} removed {} from tensor_dict".format(self, list(self.holdout_params.keys())))
        return tensors_to_send

    def create_message_header(self):
        header = MessageHeader(sender=self.id, recipient=self.agg_id, federation_id=self.fed_id, counter=self.counter)
        return header

    def __repr__(self):
        return 'collaborator {} of federation {}'.format(self.id, self.fed_id)

    def __str__(self):
        return self.__repr__()

    def validate_header(self, reply):
        # check message is from my agg to me
        check_equal(reply.header.sender, self.agg_id, self.logger)
        check_equal(reply.header.recipient, self.id, self.logger)
        
        # check that the federation id matches
        check_equal(reply.header.federation_id, self.fed_id, self.logger)

    def run(self):
        time_to_quit = False
        while True:
            time_to_quit = self.run_to_yield_or_quit()
            if time_to_quit:
                print(self, 'quitting')
                break
            else:
                time.sleep(self.polling_interval)

    def run_to_yield_or_quit(self):
        self.logger.info("Collaborator [%s] connects to federation [%s] and aggegator [%s]." % (self.id, self.fed_id, self.agg_id))
        self.logger.debug("The optimizer variable treatment is [%s]." % self.opt_treatment)
        while True:
            # query for job and validate it
            reply = self.channel.RequestJob(JobRequest(header=self.create_message_header(), model_header=self.model_header))
            self.validate_header(reply)
            check_type(reply, JobReply, self.logger)
            job = reply.job

            self.logger.debug("%s - Got a job %s" % (self, Job.Name(job)))
           
            if job is JOB_DOWNLOAD_MODEL:
                self.do_download_model_job()
            elif job is JOB_VALIDATE:
                self.do_validate_job()
            elif job is JOB_TRAIN:
                self.do_train_job()
            elif job is JOB_YIELD:
                return False
            elif job is JOB_QUIT:
                return True
            
    def _with_opt_vars(self):
        if self.opt_treatment in (OptTreatment.EDGE, OptTreatment.RESET):
            self.logger.debug("Not share the optimization variables.")
            return False
        elif self.opt_treatment == OptTreatment.AGG:
            self.logger.debug("Share the optimization variables.")
            return True

    def do_train_job(self):
        # get the initial tensor dict
        # initial_tensor_dict = self.wrapped_model.get_tensor_dict()

        # train the model
        # FIXME: model header "version" needs to be changed to "rounds_trained"
        loss = self.wrapped_model.train_epoch()
        self.logger.debug("{} Completed the training job.".format(self))

        # get the training data size
        data_size = self.wrapped_model.get_training_data_size()

        # get the trained tensor dict and store any desginated to be held out from aggregation
        tensor_dict = self._remove_and_save_holdout_params(self.wrapped_model.get_tensor_dict(with_opt_vars=self._with_opt_vars()))

        # convert to a delta
        # for k in tensor_dict.keys():
        #     tensor_dict[k] -= initial_tensor_dict[k]

        # create the tensor proto list
        tensor_protos = []
        for k, v in tensor_dict.items():
            tensor_protos.append(TensorProto(name=k, shape=v.shape, npbytes=v.tobytes('C')))

        model_proto = ModelProto(header=self.model_header, tensors=tensor_protos)
        self.logger.debug("{} - Sending the model to the aggeregator.".format(self))

        reply = self.channel.UploadLocalModelUpdate(LocalModelUpdate(header=self.create_message_header(), model=model_proto, data_size=data_size, loss=loss))
        self.validate_header(reply)
        check_type(reply, LocalModelUpdateAck, self.logger)
        self.logger.info("{} - Model update succesfully sent to aggregtor".format(self))

    def do_validate_job(self):
        results = self.wrapped_model.validate()
        self.logger.debug("{} - Completed the validation job.".format(self))
        data_size = self.wrapped_model.get_validation_data_size()

        reply = self.channel.UploadLocalMetricsUpdate(LocalValidationResults(header=self.create_message_header(), model_header=self.model_header, results=results, data_size=data_size))
        self.validate_header(reply)
        check_type(reply, LocalValidationResultsAck, self.logger)
        
    def do_download_model_job(self):

        # time the download
        download_start = time.time()

        # sanity check on version is implicit in send
        reply = self.channel.DownloadModel(ModelDownloadRequest(header=self.create_message_header(), model_header=self.model_header))

        self.logger.info("{} took {} seconds to download the model".format(self, round(time.time() - download_start, 3)))

        self.validate_header(reply)
        self.logger.info("{} - Completed the model downloading job.".format(self))

        check_type(reply, GlobalModelUpdate, self.logger)
        
        # ensure we actually got a new model version
        check_not_equal(reply.model.header.version, self.model_header.version, self.logger)
        
        # set our model header
        self.model_header = reply.model.header

        # create the aggregated tensors dict
        agg_tensor_dict = {}
        # Note: Tensor components of non-float type will not reconstruct correctly below,
        # which is why default behaviour is to hold out all non-float parameters from aggregation. 
        for tensor_proto in reply.model.tensors:
            try:
                agg_tensor_dict[tensor_proto.name] = np.frombuffer(tensor_proto.npbytes, dtype=np.float32).reshape(tensor_proto.shape)
            except ValueError as e:
                self.logger.debug("ValueError for proto {}".format(tensor_proto.name))
                raise e

        tensor_dict = {**agg_tensor_dict, **self.holdout_params}
        

        if self.opt_treatment == OptTreatment.AGG:
            with_opt_vars = True
        else:
            with_opt_vars = False

        # Ensuring proper initialization regardless of model state. Initial global models 
        # do not contain optimizer state, and so cannot be used to reset the optimizer params.
        if reply.model.header.version == 0:
            with_opt_vars = False
            self.wrapped_model.reset_opt_vars()

        self.wrapped_model.set_tensor_dict(tensor_dict, with_opt_vars=with_opt_vars)
        self.logger.debug("Loaded the model.")

        # FIXME: for the EDGE treatment, we need to store the status in case of a crash.
        if self.opt_treatment == OptTreatment.RESET:
            try:
                self.wrapped_model.reset_opt_vars()
            except:
                self.logger.exception("Failed to reset the optimization variables.")
                raise
            else:
                self.logger.debug("Reset the optimization variables.")
