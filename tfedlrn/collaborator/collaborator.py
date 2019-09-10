import time
import logging
import numpy as np

from .. import check_type, check_equal
from ..proto.message_pb2 import MessageHeader
from ..proto.message_pb2 import Job, JobRequest, JobReply
from ..proto.message_pb2 import JOB_DOWNLOAD_MODEL, JOB_QUIT, JOB_TRAIN, JOB_VALIDATE, JOB_YIELD
from ..proto.message_pb2 import ModelProto, ModelHeader, TensorProto
from ..proto.message_pb2 import ModelDownloadRequest, GlobalModelUpdate
from ..proto.message_pb2 import LocalModelUpdate, LocalModelUpdateAck
from ..proto.message_pb2 import LocalValidationResults, LocalValidationResultsAck


from enum import Enum

class OptTreatment(Enum):
    RESET = 1
    EDGE = 2
    AGG = 3


# FIXME: this is actually a tuple of a collaborator/flplan
# CollaboratorFLPlanExecutor?
class Collaborator(object):
    """The current class is not good for local test without connection. """
    # FIXME: do we need a settable model version? Shouldn't col always start assuming out of sync?
    def __init__(self, id, agg_id, fed_id, wrapped_model, connection, model_version, polling_interval=4, opt_treatment="AGG"):
        self.logger = logging.getLogger(__name__)
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

        # AGG/EDGE/RESET
        if hasattr(OptTreatment, opt_treatment):
            self.opt_treatment = OptTreatment[opt_treatment]
        else:
            self.logger.error("Unknown opt_treatment: %s." % opt_treatment)
            raise NotImplementedError("Unknown opt_treatment: %s." % opt_treatment)

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
        if not (reply.header.sender == self.agg_id and reply.header.recipient == self.id):
            self.logger.exception("Assertion failed: reply.header.sender == self.agg_id and reply.header.recipient == self.id")

        # check that the federation id matches
        if not (reply.header.federation_id == self.fed_id):
            self.logger.exception("Assertion failed: reply.header.federation_id == self.fed_id")

        # check that the counters match
        if not(reply.header.counter == self.counter):
            self.logger.exception("Assertion failed: reply.header.counter == self.counter")

        # increment our counter
        self.counter += 1

        return reply

    def run(self):
        self.logger.debug("Collaborator [%s] connects to federation [%s] and aggegator [%s]." % (self.id, self.fed_id, self.agg_id))
        self.logger.debug("The optimizer variable treatment is [%s]." % self.opt_treatment)
        while True:
            # query for job
            # returns when a job has been received
            job = self.query_for_job()

            self.logger.debug("Got a job %s" % Job.Name(job))

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

            if not(isinstance(reply, JobReply)):
                self.logger.exception("Assertion failed: isinstance(reply, JobReply)")

            if reply.job is not JOB_YIELD:
                break
            time.sleep(self.polling_interval)

        return reply.job

    def do_train_job(self):
        # get the initial tensor dict
        # initial_tensor_dict = self.wrapped_model.get_tensor_dict()

        # train the model
        # FIXME: model header "version" needs to be changed to "rounds_trained"
        loss = self.wrapped_model.train_epoch()
        self.logger.debug("Completed the training job.")

        # get the training data size
        data_size = self.wrapped_model.get_training_data_size()

        # get the trained tensor dict
        if self.opt_treatment in (OptTreatment.EDGE, OptTreatment.RESET):
            with_opt_vars = False
            self.logger.debug("Not share the optimization variables.")
        elif self.opt_treatment == OptTreatment.AGG:
            with_opt_vars = True
            self.logger.debug("Share the optimization variables.")
        tensor_dict = self.wrapped_model.get_tensor_dict(with_opt_vars)

        # convert to a delta
        # for k in tensor_dict.keys():
        #     tensor_dict[k] -= initial_tensor_dict[k]

        # create the tensor proto list
        tensor_protos = []
        for k, v in tensor_dict.items():
            tensor_protos.append(TensorProto(name=k, shape=v.shape, npbytes=v.tobytes('C')))

        model_proto = ModelProto(header=self.model_header, tensors=tensor_protos)
        self.logger.debug("Sending the model to the aggeregator.")
        reply = self.send_and_receive(LocalModelUpdate(header=self.create_message_header(), model=model_proto, data_size=data_size, loss=loss))
        if not(isinstance(reply, LocalModelUpdateAck)):
            self.logger.exception("Assertion failed: isinstance(reply, LocalModelUpdateAck)")
        self.logger.debug("Model sent.")

    def do_validate_job(self):
        results = self.wrapped_model.validate()
        self.logger.debug("Completed the validation job.")
        data_size = self.wrapped_model.get_validation_data_size()

        reply = self.send_and_receive(LocalValidationResults(header=self.create_message_header(), model_header=self.model_header, results=results, data_size=data_size))
        if not(isinstance(reply, LocalValidationResultsAck)):
            self.logger.exception("Assertion failed: isinstance(reply, LocalValidationResultsAck)")

    def do_download_model_job(self):
        # sanity check on version is implicit in send
        reply = self.send_and_receive(ModelDownloadRequest(header=self.create_message_header(), model_header=self.model_header))
        self.logger.debug("Completed the downloading job.")

        if not(isinstance(reply, GlobalModelUpdate)):
            self.logger.exception("Assertion failed: isinstance(reply, GlobalModelUpdate)")

        # ensure we actually got a new model version
        if not(reply.model.header.version != self.model_header.version):
            self.logger.exception("Assertion failed: reply.model.header.version != self.model_header.version")

        # set our model header
        self.model_header = reply.model.header

        # create the tensor dict
        tensor_dict = {}
        for tensor_proto in reply.model.tensors:
            tensor_dict[tensor_proto.name] = np.frombuffer(tensor_proto.npbytes, dtype=np.float32).reshape(tensor_proto.shape)

        if self.opt_treatment == OptTreatment.AGG:
            with_opt_vars = True
        else:
            with_opt_vars = False
        self.wrapped_model.set_tensor_dict(tensor_dict, with_opt_vars=with_opt_vars)
        self.logger.debug("Loaded the model.")

        # FIXME: for the EDGE treatment, we need to store the status in case of a crash.
        if self.opt_treatment == OptTreatment.RESET:
            try:
                self.wrapped_model.reset_opt_vars()
            except:
                self.logger.exception("Failed to reset the optimization variables.")
            else:
                self.logger.debug("Reset the optimization variables.")
