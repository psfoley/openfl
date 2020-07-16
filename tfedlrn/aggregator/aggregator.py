# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

import time
import os
import logging
import time

import numpy as np
import tensorboardX
from threading import Lock

from .. import check_equal, check_not_equal, check_is_in, check_not_in
from ..proto.collaborator_aggregator_interface_pb2 import MessageHeader
from ..proto.collaborator_aggregator_interface_pb2 import Job, JobRequest, JobReply
from ..proto.collaborator_aggregator_interface_pb2 import JOB_DOWNLOAD_MODEL, JOB_QUIT, JOB_TRAIN, JOB_VALIDATE, JOB_YIELD
from ..proto.collaborator_aggregator_interface_pb2 import ModelProto, ModelHeader, TensorProto
from ..proto.collaborator_aggregator_interface_pb2 import ModelDownloadRequest, GlobalModelUpdate
from ..proto.collaborator_aggregator_interface_pb2 import LocalModelUpdate, LocalValidationResults, LocalModelUpdateAck, LocalValidationResultsAck


from tfedlrn.proto.protoutils import dump_proto, load_proto, construct_proto, deconstruct_proto
from tfedlrn.tensor_transformation_pipelines import NoCompressionPipeline

# FIXME: simpler "stats" collection/handling
# FIXME: remove the round tracking/job-result tracking stuff from this?
# Feels like it conflates model aggregation with round management
# FIXME: persistence of the trained weights.
class Aggregator(object):
    """An Aggregator is the central node in federated learning.

    Parameters
    ----------
    id : str
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
    # FIXME: no selector logic is in place
    def __init__(self,
                 aggregator_uuid,
                 federation_uuid,
                 collaborator_common_names,
                 init_model_fpath,
                 latest_model_fpath,
                 best_model_fpath,
                 rounds_to_train=256,
                 minimum_reporting=-1,
                 straggler_cutoff_time=np.inf,
                 disable_equality_check=True,
                 single_col_cert_common_name=None,
                 compression_pipeline=None,
                 **kwargs):
        self.logger = logging.getLogger(__name__)
        self.uuid = aggregator_uuid
        self.federation_uuid = federation_uuid
        #FIXME: Should we do anything to insure the intial model is compressed?
        self.model = load_proto(init_model_fpath)
        self.latest_model_fpath = latest_model_fpath
        self.best_model_fpath = best_model_fpath
        self.collaborator_common_names = collaborator_common_names
        self.round_num = 1
        self.rounds_to_train = rounds_to_train
        self.quit_job_sent_to = []
        self.disable_equality_check = disable_equality_check
        self.minimum_reporting = minimum_reporting
        self.straggler_cutoff_time = straggler_cutoff_time
        self.round_start_time = None
        self.single_col_cert_common_name = single_col_cert_common_name

        if self.single_col_cert_common_name is not None:
            self.log_big_warning()
        else:
            self.single_col_cert_common_name = '' # FIXME: '' instead of None is just for protobuf compatibility. Cleaner solution?

        #FIXME: close the handler before termination.
        log_dir = './logs/tensorboardX/%s_%s' % (self.uuid, self.federation_uuid)
        self.tb_writer = tensorboardX.SummaryWriter(log_dir, flush_secs=10)

        self.model_update_in_progress = None

        self.init_per_col_round_stats()
        self.best_model_score = None
        self.mutex = Lock()

        self.compression_pipeline = compression_pipeline or NoCompressionPipeline()

    def valid_collaborator_CN_and_id(self, cert_common_name, collaborator_common_name):
        """Determine if the collaborator certificate and ID are valid for this federation.

        Args:
            cert_common_name: Common name for security certificate
            collaborator_common_name: Common name for collaborator

        Returns:
            boolean: True means the collaborator common name matches the name in the security certificate.

        """
        # if self.test_mode_whitelist is None, then the common_name must match collaborator_common_name and be in collaborator_common_names
        if self.single_col_cert_common_name == '':  # FIXME: '' instead of None is just for protobuf compatibility. Cleaner solution?
            return cert_common_name == collaborator_common_name and collaborator_common_name in self.collaborator_common_names
        # otherwise, common_name must be in whitelist and collaborator_common_name must be in collaborator_common_names
        else:
            return cert_common_name == self.single_col_cert_common_name and collaborator_common_name in self.collaborator_common_names

    def all_quit_jobs_sent(self):
        """Determines if all collaborators have been sent the QUIT command.

        Returns:
            boolean: True if all collaborators have been sent the QUIT command.

        """
        return sorted(self.quit_job_sent_to) == sorted(self.collaborator_common_names)

    def validate_header(self, message):
        """Validates the message is from valid collaborator in this federation.

        Returns:
            boolean: True if the message is from a valid collaborator in this federation.

        """

        # validate that the message is for me
        check_equal(message.header.recipient, self.uuid, self.logger)

        # validate that the message is for my federation
        check_equal(message.header.federation_id, self.federation_uuid, self.logger)

        # validate that the sender is one of my collaborators
        check_is_in(message.header.sender, self.collaborator_common_names, self.logger)

        # check that we agree on single_col_cert_common_name
        check_equal(message.header.single_col_cert_common_name, self.single_col_cert_common_name, self.logger)

    def init_per_col_round_stats(self):
        """Initalize the metrics from collaborators for each round of aggregation. """
        keys = ["loss_results", "collaborator_training_sizes", "agg_validation_results", "preagg_validation_results", "collaborator_validation_sizes"]
        values = [{} for i in range(len(keys))]
        self.per_col_round_stats = dict(zip(keys, values))

    def collaborator_is_done(self, c):
        """Determines if a collaborator is finished a round.

        Args:
            c: Collaborator name

        Returns:
            boolean: True if collaborator c is done.

        """
        assert c in self.collaborator_common_names

        # FIXME: this only works because we have fixed roles each round
        return (c in self.per_col_round_stats["loss_results"] and
                c in self.per_col_round_stats["collaborator_training_sizes"] and
                c in self.per_col_round_stats["agg_validation_results"] and
                c in self.per_col_round_stats["preagg_validation_results"] and
                c in self.per_col_round_stats["collaborator_validation_sizes"])

    def num_collaborators_done(self):
        """Returns the number of collaborators that have finished the training round.

        Returns:
            int: The number of collaborators that have finished this round of training.

        """
        return sum([self.collaborator_is_done(c) for c in self.collaborator_common_names])

    def straggler_time_expired(self):
        """Determines if there are still collaborators that have not returned past the expected round time.
        Returns:
            boolean: True if training round limit has expired (i.e. there are straggler collaborators that have not returned in the expected time)

        """
        return self.round_start_time is not None and ((time.time() - self.round_start_time) > self.straggler_cutoff_time)

    def minimum_collaborators_reported(self):
        """Determines if enough collaborators have returned to do the aggregation.

        Returns:
            boolean: True if the number of collaborators that have finished is greater than the minimum threshold set.

        """
        return self.num_collaborators_done() >= self.minimum_reporting

    def straggler_cutoff_check(self):
        """Determines if a collaborator is finished a round.

        Args:
            c: Collaborator name

        Returns:
            boolean: True if collaborator c is done.

        """
        cutoff = self.straggler_time_expired() and self.minimum_collaborators_reported()
        if cutoff:
            collaborators_done = [c for c in self.collaborator_common_names if self.collaborator_is_done(c)]
            self.logger.info('\tEnding round early due to straggler cutoff. Collaborators done: {}'.format(collaborators_done))
        return cutoff

    def end_of_round_check(self):
        """Determines if it is the end of a training round.

        Returns:
            boolean: True if training round has ended.

        """
        # FIXME: find a nice, clean way to manage these values without having to manually ensure
        # the keys are in sync

        # assert our dictionary keys are in sync
        check_equal(self.per_col_round_stats["loss_results"].keys(), self.per_col_round_stats["collaborator_training_sizes"].keys(), self.logger)
        check_equal(self.per_col_round_stats["agg_validation_results"].keys(), self.per_col_round_stats["collaborator_validation_sizes"].keys(), self.logger)

        # if everyone is done OR our straggler policy calls for an early round end
        if self.num_collaborators_done() == len(self.collaborator_common_names) or self.straggler_cutoff_check():
            self.end_of_round()

    def get_weighted_average_of_collaborators(self, value_dict, weight_dict):
        """Calculate the weighted average of the model updates from the collaborators.

        Args:
            value_dict: A dictionary of the values (model tensors)
            weight_dict: A dictionary of the collaborator weights (percentage of total data size)

        Returns:
            Dictionary of the weights average for all collaborator models

        """
        cols = [k for k in value_dict.keys() if k in self.collaborator_common_names]
        return np.average([value_dict[c] for c in cols], weights=[weight_dict[c] for c in cols])

    def end_of_round(self):
        """Runs required tasks when the training round has ended.
        """
        # FIXME: what all should we do to track results/metrics? It should really be an easy, extensible solution

        # compute the weighted loss average
        round_loss = self.get_weighted_average_of_collaborators(self.per_col_round_stats["loss_results"],
                                                                self.per_col_round_stats["collaborator_training_sizes"])

        # compute the weighted validation average
        round_val = self.get_weighted_average_of_collaborators(self.per_col_round_stats["agg_validation_results"],
                                                                self.per_col_round_stats["collaborator_validation_sizes"])

        # FIXME: proper logging
        self.logger.info('round results for model id/version {}/{}'.format(self.model.header.id, self.model.header.version))
        self.logger.info('\tvalidation: {}'.format(round_val))
        self.logger.info('\tloss: {}'.format(round_loss))

        self.tb_writer.add_scalars('training/loss', {**self.per_col_round_stats["loss_results"], "federation": round_loss}, global_step=self.round_num)
        self.tb_writer.add_scalars('training/size', self.per_col_round_stats["collaborator_training_sizes"], global_step=self.round_num)
        self.tb_writer.add_scalars('validation/preagg_result', self.per_col_round_stats["preagg_validation_results"], global_step=self.round_num)
        self.tb_writer.add_scalars('validation/size', self.per_col_round_stats["collaborator_validation_sizes"], global_step=self.round_num-1)
        self.tb_writer.add_scalars('validation/agg_result', {**self.per_col_round_stats["agg_validation_results"], "federation": round_val}, global_step=self.round_num-1)

        # construct the model protobuf from in progress tensors (with incremented version number)
        self.model = construct_proto(tensor_dict=self.model_update_in_progress["tensor_dict"],
                                     model_id=self.model.header.id,
                                     model_version=self.model.header.version + 1,
                                     is_delta=self.model_update_in_progress["is_delta"],
                                     delta_from_version=self.model_update_in_progress["delta_from_version"],
                                     compression_pipeline=self.compression_pipeline)

        # Save the new model as latest model.
        dump_proto(self.model, self.latest_model_fpath)

        model_score = round_val
        if self.best_model_score is None or self.best_model_score < model_score:
            self.logger.info("Saved the best model with score {:f}.".format(model_score))
            self.best_model_score = model_score
            # Save a model proto version to file as current best model.
            dump_proto(self.model, self.best_model_fpath)

        # clear the update pointer
        self.model_update_in_progress = None

        self.init_per_col_round_stats()

        self.round_num += 1
        self.logger.debug("Start a new round %d." % self.round_num)
        self.round_start_time = None

    def UploadLocalModelUpdate(self, message):
        """Parses the collaborator reply message to get the collaborator model update

        Args:
            message: Message from the collaborator

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """

        self.mutex.acquire(blocking=True)
        try:
            t = time.time()
            self.validate_header(message)

            self.logger.info("Receive model update from %s " % message.header.sender)

            # Get the model parameters from the model proto and additional model info
            model_tensors = deconstruct_proto(model_proto=message.model, compression_pipeline=self.compression_pipeline)
            is_delta = message.model.header.is_delta
            delta_from_version = message.model.header.delta_from_version

            # validate this model header
            check_equal(message.model.header.id, self.model.header.id, self.logger)
            check_equal(message.model.header.version, self.model.header.version, self.logger)

            # ensure we haven't received an update from this collaborator already
            check_not_in(message.header.sender, self.per_col_round_stats["loss_results"], self.logger)
            check_not_in(message.header.sender, self.per_col_round_stats["collaborator_training_sizes"], self.logger)

            # if this is our very first update for the round, we take these model tensors as-is
            # FIXME: move to model deltas, add with original to reconstruct
            # FIXME: this really only works with a trusted collaborator. Sanity check this against self.model
            if self.model_update_in_progress is None:
                self.model_update_in_progress = {"tensor_dict": model_tensors,
                                                 "is_delta": is_delta,
                                                 "delta_from_version": delta_from_version}

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

                # check that the models include the same number of tensors, and that whether or not
                # it is a delta and from what version is the same
                check_equal(len(self.model_update_in_progress["tensor_dict"]), len(model_tensors), self.logger)
                check_equal(self.model_update_in_progress["is_delta"], is_delta, self.logger)
                check_equal(self.model_update_in_progress["delta_from_version"], delta_from_version, self.logger)

                # aggregate all the model tensors in the tensor_dict
                # (weighted average of local update l and global tensor g for all l, g)
                for name, l in model_tensors.items():
                    g = self.model_update_in_progress["tensor_dict"][name]
                    # check that g and l have the same shape
                    check_equal(g.shape, l.shape, self.logger)

                    # sanity check that the tensors are indeed different for non opt tensors
                    # TODO: modify this to better comprehend for non pytorch how to identify the opt portion (use model opt info?)
                    if (not self.disable_equality_check \
                        and not name.startswith('__opt') \
                        and 'RMSprop' not in name \
                        and 'Adam' not in name \
                        and 'RMSProp' not in name):
                        check_equal(np.all(g == l), False, self.logger)


                    # now store a weighted average into the update in progress
                    self.model_update_in_progress["tensor_dict"][name] = np.average([g, l], weights=[weight_g, weight_l], axis=0)

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
        """Parses the collaborator reply message to get the collaborator metrics (usually the local validation score)

        Args:
            message: Message from the collaborator

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """

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
        """Parse message for job request and act accordingly.

        Args:
            message: Message from the collaborator

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """

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
            # check to see if we need to set our round start time
            self.mutex.acquire(blocking=True)
            try:
                if self.round_start_time is None:
                    self.round_start_time = time.time()
            finally:
                self.mutex.release()

            self.logger.debug('aggregator handled RequestJob in time {}'.format(time.time() - t))
        elif self.straggler_cutoff_time != np.inf:
            # we have an idle collaborator and a straggler cutoff time, so we should check for early round end
            self.mutex.acquire(blocking=True)
            try:
                self.end_of_round_check()
            finally:
                self.mutex.release()

        return reply

    def DownloadModel(self, message):
        """Sends a model to the collaborator

        Args:
            message: Message from the collaborator

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """

        t = time.time()
        self.validate_header(message)

        self.logger.info("Received model download request from %s " % message.header.sender)

        # ensure the models don't match
        if not(self.collaborator_out_of_date(message.model_header)):
            statement = "Collaborator asking for download when not out of date."
            self.logger.exception(statement)
            raise RuntimeError(statement)

        # check whether there is an issue related to the sending of deltas or non-deltas
        if message.model_header.version == -1:
            if self.model.header.is_delta:
                raise RuntimeError('First collaborator model download, and we only have a delta.')
        elif message.model_header.is_delta != self.model.header.is_delta:
            raise RuntimeError('Collaborator requesting non-initial download should hold a model with the same is_delta as aggregated model.')
        elif message.model_header.is_delta and (message.model_header.delta_from_version != self.model.header.delta_from_version):
            # TODO: In the future we could send non-delta model here to restore base model.
            raise NotImplementedError('Base of download model delta does not match current collaborator base, and aggregator restoration of base model not implemented.')

        reply = GlobalModelUpdate(header=self.create_reply_header(message), model=self.model)

        self.logger.debug('aggregator handled RequestJob in time {}'.format(time.time() - t))

        return reply

    def create_reply_header(self, message):
        """Creates a header for the reply to the message

        Args:
            message: Message from the collaborator

        Returns:
            The message header.

        """
        return MessageHeader(sender=self.uuid, recipient=message.header.sender, federation_id=self.federation_uuid, counter=message.header.counter, single_col_cert_common_name=self.single_col_cert_common_name)

    def collaborator_out_of_date(self, model_header):
        """Determines if the collaborator has the wrong version of the model (aka out of date)

        Args:
            model_header: Header for the model

        Returns:
            The reply to the message (usually just the acknowledgement to the collaborator)

        """
        # validate that this is the right model to be checking
        check_equal(model_header.id, self.model.header.id, self.logger)

        return model_header.version != self.model.header.version

    def log_big_warning(self):
        self.logger.warning("\n{}\nYOU ARE RUNNING IN SINGLE COLLABORATOR CERT MODE! THIS IS NOT PROPER PKI AND SHOULD ONLY BE USED IN DEVELOPMENT SETTINGS!!!! YE HAVE BEEN WARNED!!!".format(the_dragon))


the_dragon = """

 ,@@.@@+@@##@,@@@@.`@@#@+  *@@@@ #@##@  `@@#@# @@@@@   @@    @@@@` #@@@ :@@ `@#`@@@#.@
  @@ #@ ,@ +. @@.@* #@ :`   @+*@ .@`+.   @@ *@::@`@@   @@#  @@  #`;@`.@@ @@@`@`#@* +:@`
  @@@@@ ,@@@  @@@@  +@@+    @@@@ .@@@    @@ .@+:@@@:  .;+@` @@ ,;,#@` @@ @@@@@ ,@@@* @
  @@ #@ ,@`*. @@.@@ #@ ,;  `@+,@#.@.*`   @@ ,@::@`@@` @@@@# @@`:@;*@+ @@ @`:@@`@ *@@ `
 .@@`@@,+@+;@.@@ @@`@@;*@  ;@@#@:*@+;@  `@@;@@ #@**@+;@ `@@:`@@@@  @@@@.`@+ .@ +@+@*,@
  `` ``     ` ``  .     `     `      `     `    `  .` `  ``   ``    ``   `       .   `



                                            .**
                                      ;`  `****:
                                     @**`*******
                         ***        +***********;
                        ,@***;` .*:,;************
                        ;***********@@***********
                        ;************************,
                        `*************************
                         *************************
                         ,************************
                          **#*********************
                          *@****`     :**********;
                          +**;          .********.
                          ;*;            `*******#:                       `,:
                                          ****@@@++::                ,,;***.
                                          *@@@**;#;:         +:      **++*,
                                          @***#@@@:          +*;     ,****
                                          @*@+****           ***`     ****,
                                         ,@#******.  ,       ****     **;,**.
                                         * ******** :,       ;*:*+    **  :,**
                                        #  ********::      *,.*:**`   *      ,*;
                                        .  *********:      .+,*:;*:   :      `:**
                                       ;   :********:       ***::**   `       ` **
                                       +   :****::***  ,    *;;::**`             :*
                                      ``   .****::;**:::    *;::::*;              ;*
                                      *     *****::***:.    **::::**               ;:
                                      #     *****;:****     ;*::;***               ,*`
                                      ;     ************`  ,**:****;               ::*
                                      :     *************;:;*;*++:                   *.
                                      :     *****************;*                      `*
                                     `.    `*****************;  :                     *.
                                     .`    .*+************+****;:                     :*
                                     `.    :;+***********+******;`    :              .,*
                                      ;    ::*+*******************. `::              .`:.
                                      +    :::**********************;;:`                *
                                      +    ,::;*************;:::*******.                *
                                      #    `:::+*************:::;********  :,           *
                                      @     :::***************;:;*********;:,           *
                                      @     ::::******:*********************:         ,:*
                                      @     .:::******:;*********************,         :*
                                      #      :::******::******###@*******;;****        *,
                                      #      .::;*****::*****#****@*****;:::***;  ``  **
                                      *       ::;***********+*****+#******::*****,,,,**
                                      :        :;***********#******#******************
                                      .`       `;***********#******+****+************
                                      `,        ***#**@**+***+*****+**************;`
                                       ;         *++**#******#+****+`      `.,..
                                       +         `@***#*******#****#
                                       +          +***@********+**+:
                                       *         .+**+;**;;;**;#**#
                                      ,`         ****@         +*+:
                                      #          +**+         :+**
                                      @         ;**+,       ,***+
                                      #      #@+****      *#****+
                                     `;     @+***+@      `#**+#++
                                     #      #*#@##,      .++:.,#
                                    `*      @#            +.
                                  @@@
                                 #`@
                                  ,                                                        """
