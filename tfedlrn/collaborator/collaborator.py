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

from tfedlrn.tensor_transformation_pipelines import NoCompressionPipeline
from tfedlrn.proto.protoutils import construct_proto, deconstruct_proto

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
    def __init__(self, 
                 collaborator_common_name, 
                 aggregator_uuid, 
                 federation_uuid, 
                 wrapped_model, 
                 channel, 
                 polling_interval=4, 
                 opt_treatment="AGG", 
                 compression_pipeline=None,
                 epochs_per_round=1.0, 
                 num_batches_per_round=None, 
                 send_model_deltas = True,
                 single_col_cert_common_name=None,
                 **kwargs):
        self.logger = logging.getLogger(__name__)
        self.channel = channel
        self.polling_interval = polling_interval

        # this stuff is really about sanity/correctness checking to ensure the bookkeeping and control flow is correct
        self.common_name = collaborator_common_name
        self.aggregator_uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        self.single_col_cert_common_name = single_col_cert_common_name
        if self.single_col_cert_common_name is None:
            self.single_col_cert_common_name = '' # FIXME: this is just for protobuf compatibility. Cleaner solution?
        self.counter = 0
        self.model_header = ModelHeader(id=wrapped_model.__class__.__name__,
                                        is_delta=send_model_deltas,
                                        version=-1)
        # number of epochs to perform per round of FL (is a float that is converted 
        # to num_batches before calling the wrapped model train_batches method).
        # This is overridden by "num_batches_per_round"
        self.epochs_per_round = epochs_per_round
        self.num_batches_per_round = num_batches_per_round
        if num_batches_per_round is not None:
            self.logger.info("Collaborator {} overriding epochs_per_round of {} with num_batches_per_round of {}".format(self.common_name, self.epochs_per_round, self.num_batches_per_round))

        self.wrapped_model = wrapped_model
        self.tensor_dict_split_fn_kwargs = wrapped_model.tensor_dict_split_fn_kwargs or {}

        # pipeline translating tensor_dict to and from a list of tensor protos
        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()

        # AGG/EDGE/RESET
        if hasattr(OptTreatment, opt_treatment):
            self.opt_treatment = OptTreatment[opt_treatment]
        else:
            self.logger.error("Unknown opt_treatment: %s." % opt_treatment)
            raise NotImplementedError("Unknown opt_treatment: %s." % opt_treatment)

        # FIXME: this is a temporary fix for non-float values and other named params designated to hold out from aggregation. 
        # Needs updated when we have proper collab-side state saving.
        self._remove_and_save_holdout_tensors(self.wrapped_model.get_tensor_dict(with_opt_vars=self._with_opt_vars()))
        # when sending model deltas, baseline values for shared tensors must be kept
        self.send_model_deltas = send_model_deltas
        # the base_dict_for_deltas attibute is only used in the case that model deltas are being sent
        if self.send_model_deltas:
            self.base_dict_for_deltas = None

    def _remove_and_save_holdout_tensors(self, tensor_dict):
        shared_tensors, self.holdout_tensors = split_tensor_dict_for_holdouts(self.logger, tensor_dict, **self.tensor_dict_split_fn_kwargs)
        if self.holdout_tensors != {}:
            self.logger.debug("{} removed {} from tensor_dict".format(self, list(self.holdout_tensors.keys())))
        return shared_tensors

    def create_delta_dict(self, tensor_dict):
        if not self.send_model_deltas:
            raise ValueError("Should not be creating deltas when not sending deltas.")
        if self.base_dict_for_deltas is None:
            raise ValueError("Attempting to create deltas when no base tensors are known.")
        elif set(self.base_dict_for_deltas.keys()) != set(tensor_dict.keys()):
            raise ValueError("Attempting to convert to deltas when base tensor names do not match ones to convert.")
        delta_dict = {key: (tensor_dict[key] - self.base_dict_for_deltas[key]) for key in self.base_dict_for_deltas}
        return delta_dict

    def update_base_for_deltas(self, tensor_dict, is_delta):
        if not self.send_model_deltas:
            raise ValueError("Should not be handing a base for deltas when not sending deltas.")
        if self.base_dict_for_deltas is None:
            if is_delta:
                raise ValueError("Attempting to update undefined base tensors with a delta.")
            else:
                self.base_dict_for_deltas = tensor_dict
        else:
            if is_delta:
                if set(self.base_dict_for_deltas.keys()) != set(tensor_dict.keys()):
                    raise ValueError("Attempting to update base with tensors whose names do not match current base names.")
                self.base_dict_for_deltas = {key: (self.base_dict_for_deltas[key] + tensor_dict[key]) for key in tensor_dict}
            else:
                raise NotImplementedError("Non-delta updates of base tensors currrently only expected for first update.")


    def create_message_header(self):
        header = MessageHeader(sender=self.common_name, recipient=self.aggregator_uuid, federation_id=self.federation_uuid, counter=self.counter, single_col_cert_common_name=self.single_col_cert_common_name)
        return header

    def __repr__(self):
        return 'collaborator {} of federation {}'.format(self.common_name, self.federation_uuid)

    def __str__(self):
        return self.__repr__()

    def validate_header(self, reply):
        # check message is from my agg to me
        check_equal(reply.header.sender, self.aggregator_uuid, self.logger)
        check_equal(reply.header.recipient, self.common_name, self.logger)
        
        # check that the federation id matches
        check_equal(reply.header.federation_id, self.federation_uuid, self.logger)

        # check that we agree on single_col_cert_common_name
        check_equal(reply.header.single_col_cert_common_name, self.single_col_cert_common_name, self.logger)

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
        self.logger.info("Collaborator [%s] connects to federation [%s] and aggegator [%s]." % (self.common_name, self.federation_uuid, self.aggregator_uuid))
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

        # get the training data size
        data_size = self.wrapped_model.get_training_data_size()

        # train the model
        # FIXME: model header "version" needs to be changed to "rounds_trained"
        # FIXME: We assume the models allow training on partial batches.
        # FIXME: Currently, num_batches_per_round overrides epochs per round. Is this the correct behavior?
        if self.num_batches_per_round is not None:
            num_batches = self.num_batches_per_round
        else:
            batches_per_epoch = int(np.ceil(data_size/self.wrapped_model.data.batch_size))
            num_batches = int(np.floor(batches_per_epoch * self.epochs_per_round)) 
        loss = self.wrapped_model.train_batches(num_batches=num_batches)
        self.logger.debug("{} Completed the training job for {} batches.".format(self, num_batches))

        # get the trained tensor dict and store any designated to be held out from aggregation
        shared_tensors = self._remove_and_save_holdout_tensors(self.wrapped_model.get_tensor_dict(with_opt_vars=self._with_opt_vars()))
        
        # prep the model proto info
        if self.send_model_deltas:
            tensor_dict = self.create_delta_dict(tensor_dict=shared_tensors)
        else:
            tensor_dict = shared_tensors

        # create the model proto    
        model_proto = construct_proto(tensor_dict=tensor_dict, 
                                      model_id=self.model_header.id, 
                                      model_version=self.model_header.version, 
                                      compression_pipeline=self.compression_pipeline, 
                                      is_delta=self.send_model_deltas)

        self.logger.debug("{} - Sending the model to the aggregator.".format(self))

        reply = self.channel.UploadLocalModelUpdate(LocalModelUpdate(header=self.create_message_header(), model=model_proto, data_size=data_size, loss=loss))
        self.validate_header(reply)
        check_type(reply, LocalModelUpdateAck, self.logger)
        self.logger.info("{} - Model update succesfully sent to aggregator".format(self))

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
        received_model_proto = reply.model
        received_model_version = received_model_proto.header.version

        # handling possability that the recieved model is delta
        received_model_is_delta = received_model_proto.header.is_delta
        
    
        self.logger.info("{} took {} seconds to download the model".format(self, round(time.time() - download_start, 3)))

        self.validate_header(reply)
        self.logger.info("{} - Completed the model downloading job.".format(self))

        check_type(reply, GlobalModelUpdate, self.logger)
        
        # ensure we actually got a new model version
        check_not_equal(received_model_version, self.model_header.version, self.logger)
        
        # set our model header using the locally defined is_delta attribute and shared model id and version
        self.model_header = ModelHeader(id=received_model_proto.header.id,
                                        is_delta=self.send_model_deltas,
                                        version=received_model_proto.header.version)

        # compute the aggregated tensors dict from the model proto
        agg_tensor_dict = deconstruct_proto(model_proto=received_model_proto, compression_pipeline=self.compression_pipeline)
        
        # We update the base every round, so we can use the base to get the current global values of the shared tensors.
        if self.send_model_deltas:
            self.update_base_for_deltas(tensor_dict=agg_tensor_dict, 
                                        is_delta=received_model_is_delta)
            # base_for_deltas can provide the global shared tensor values here
            agg_tensor_dict = self.base_dict_for_deltas

        # restore any tensors held out from aggregation
        tensor_dict = {**agg_tensor_dict, **self.holdout_tensors}
        

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
