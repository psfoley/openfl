#!/usr/bin/env python3

"""
Tasks: 
1. Load FL plans, check which ones are active/completed;
2. Generate parameters for aggregators and start them. (e.g. simple_fl_agg.py)
3. Create and run an aggregator. 
4. Receive query from collaborators and reply with an FL plan.
    Query: I have [dataset] and collaborator software [version].
    Reply 1: You should join [federation] with [aggregator] at [url]. 
    Reply 2: I have nothing for you. Come back later.
    Reply 3: I have something for you but your software is outdated. Please update to tfedlrn v2.5.

    Query: I want to join [plan] but I don't have the code. Send me the code.
    Reply: Here is the [code], [size], [checksum].

"""

import yaml
import argparse
import logging
import time
import pathlib
import sys
import os
import socket
import hashlib
import multiprocessing
import importlib

from ..aggregator.aggregator import Aggregator
from ..collaborator.collaborator import Collaborator
from ..zmqconnection import ZMQServer, ZMQClient
from ..proto.message_pb2 import PlanRequest, PlanReply, CodeRequest, CodeReply
from .packaging import zip_files, unzip_files, check_bytes_sha256


"""
Compose an FL plan in JSON/YAML.
* Task name
* Task description
* The folder path to the locally runnable code.
* The software dependency.
* FL parameters: fed_id, opt_treatment.

* Added by the coordinator later: agg_id, agg_url, poll_interval.
"""

def start_aggregator(addr, port, agg_id, fed_id, num_collaborators, initial_weights_fpath, latest_weights_fpath):
    col_ids = ["{}".format(i) for i in range(num_collaborators)]
    connection = ZMQServer('{} connection'.format(agg_id), server_addr=addr, server_port=port)
    agg = Aggregator(agg_id, fed_id, col_ids, connection, initial_weights_fpath, latest_weights_fpath)
    agg.run()

class Server(object):
    """ The server-side coordinator for loading plans and starting aggregators.

    Parameters
    ----------
    connection : ZMQServer
        An established ZMQ server socket.
    plans_path : str
        Location of a file or a folder.
    """
    def __init__(self, connection, plans_path):
        self.connection = connection
        self.logger = logging.getLogger(__name__)
        self.plans = self.search_and_load_plans(plans_path)
        self.code_dir = "models"

    def run(self):
        self.logger.debug("Start the coordinator.")
        self.aggregators = self.start_aggregators()
        while True:
            # receive a message
            message = self.connection.receive()

            if isinstance(message, PlanRequest):
                reply = self.handle_plan_request(message)
            elif isinstance(message, CodeRequest):
                reply = self.handle_code_request(message)
            else:
                self.logger.debug("Unsupported message.")

            self.connection.send(reply)

    def handle_plan_request(self, msg):
        reply = None
        for _, plan in self.plans.items():
            # import pdb; pdb.set_trace()
            if plan['dataset'] == msg.dataset and plan['software_version'] == msg.software_version:
                reply = PlanReply(fed_id=plan['federation'], 
                        agg_id=plan['aggregator']['id'], 
                        agg_addr=socket.gethostbyname(socket.gethostname()), 
                        agg_port=plan['aggregator']['port'], 
                        opt_treatment=plan['collaborator']['opt_vars_treatment'],
                        code=plan['model']['name']
                )
                break
            elif plan['dataset'] == msg.dataset and plan['software_version'] != msg.software_version:
                reply = PlanReply(fed_id="invalid", description="The plan requires software version %s, but you have version %s" % (plan['software_version'], msg.software_version))
                break
        if reply == None:
            reply = PlanReply(fed_id="invalid", description="Plan not found for the dataset.")
        return reply

    def handle_code_request(self, msg):
        if msg.fed_id not in self.plans:
            return CodeReply(size=0)
        else:
            plan = self.plans[msg.fed_id]
            fpath = plan['model']['code']['path']
            # FIXME: compress the source code.

            cur_dir = os.getcwd()
            try:
                os.chdir(self.code_dir)
                bytes = zip_files(fpath)
            except:
                os.chdir(cur_dir)
                self.logger.debug("Failed to zip the code files.")
            else:
                self.logger.debug("Got a zip tarball of the code files.")
                os.chdir(cur_dir)
            checksum = hashlib.sha256(bytes).hexdigest()
            size = len(bytes)
            return CodeReply(size=size, 
                        checksum=checksum,
                        code_zip=bytes)

    def load_plan(self, path):
        plan = None
        with open(path, 'r') as f:
            try:
                plan = yaml.safe_load(f.read())
            except Exception:
                self.logger.error("Invalid federated learning plan [%s]." % path, exc_info=True)
        return plan

    def search_and_load_plans(self, path):
        """Load plans from a file or a folder.

        Parameters
        ----------
        path : str
            Path to a folder or a file.

        Returns
        -------
        list
            A list of loaded plans.
        """
        plans = {}
        if os.path.isdir(path):
            plan_files = sorted(pathlib.Path(path).glob("*.yaml"))
        elif os.path.isfile(path):
            plan_files = [path]
        else:
            plan_files = []
            self.logger.debug("Invalid plan path: %s." % path)
        for fpath in plan_files:
            fpath_str = str(fpath)

            plan = self.load_plan(fpath_str)
            # FIXME: sanity check.
            if plan is None:
                self.logger.debug("Failed to load plan [%s]" % fpath_str)
            else:
                self.logger.debug("Loaded plan [%s]" % fpath_str)
                plans[plan['federation']] = plan
        return plans

    def start_aggregators(self):
        """Start an aggregator for each FL plan.

        Returns
        -------
        list
            A list of aggregator processes.
        """
        aggregators = []
        for fed_id,plan in self.plans.items():
            fed_id = plan['federation']

            aggregator = plan['aggregator']
            agg_id = aggregator['id']
            addr = aggregator['addr']
            port = aggregator['port']
            initial_weights_fpath = aggregator['initial_weights']
            latest_weights_fpath = aggregator['latest_weights']
            # tfedlrn_version = aggregator['tfedlrn_version']

            num_collaborators = int(aggregator['collaborators'])

            # connection = ZMQServer('{} connection'.format(agg_id), server_addr=addr, server_port=port)


            # FIXME: we will pass the initial weights and receive the latest weights for persistant storage.

            p = multiprocessing.Process(target=start_aggregator, args=(addr, port, agg_id, fed_id, num_collaborators, initial_weights_fpath, latest_weights_fpath))
            aggregators.append(p)
            p.start()
            p.join(1)

            if p.is_alive():
                self.logger.debug("Started an aggregator for federation [%s]." % fed_id)
            else:
                self.logger.debug("Failed to start an aggregator for federation [%s]." % fed_id)

        return aggregators


class Client(object):
    """The client side coordinator to request a plan and join the federated learning.

    Parameters
    ----------
    connection : ZMQClient
        The pyzmq client side socket.
    col_id : str
        The collaborator ID.
    dataset_name : str
        The dataset name used to match with available FL plans.
    software_version : str
        The software version to match with available FL plans.
    models_folder : str
        The path to a folder that serves as temporary code storage.

    splits : list
        [Debug only] A list of integers to represent the size of data shards.
    split_idx : int
        [Debug only] Specify to load which data shard.
    """
    def __init__(self, connection, col_id, dataset_name, software_version, models_folder, splits=None, split_idx=None):
        self.logger = logging.getLogger(__name__)
        self.connection = connection
        self.col_id = col_id
        self.polling_interval = 4
        self.counter = 0

        self.dataset_name = dataset_name
        self.software_version = software_version

        self.models_folder = models_folder
        self.init_models_folder()

        self.splits = splits
        self.split_idx = split_idx

    def init_models_folder(self):
        models_folder = self.models_folder
        if not os.path.isdir(models_folder):
            os.makedirs(models_folder)
        sys.path.append(models_folder)

    def download_code(self, fed_id):
        """ Download and extract the code from the server.

        Parameters
        ----------
        fed_id : str
            The federation ID

        Returns
        -------
        bool
            If the download succeeds.
        """
        reply = self.send_and_receive(CodeRequest(fed_id=fed_id))

        if reply.size == len(reply.code_zip):
            verified = check_bytes_sha256(reply.code_zip, reply.checksum)
            if verified:
                self.logger.debug('Verified the code.')
                unzip_files(reply.code_zip, self.models_folder)
                self.logger.debug("Extracted the code.")
                return True
        return False

    def run(self):
        self.logger.debug("Client coordinator connects to Server coordinator at [%s]." % (self.connection.server_addr))
        while True:
            # query for plan
            # breaks when a plan has been received
            # plan = self.query_for_plan()
            plan = self.send_and_receive(PlanRequest(dataset=self.dataset_name, software_version=self.software_version))

            if plan.fed_id == "invalid":
                self.logger.debug("Plan not found: %s" % plan.description)
                time.sleep(self.polling_interval)
            else:
                break

        self.logger.debug("Got a plan [%s]" % plan.fed_id )
        code_name = plan.code

        self.logger.debug("Start to download the code.")
        if self.download_code(plan.fed_id):
            self.logger.debug("Downloaded the code.")
        else:
            self.logger.debug("Failed to download or extract the code")

        module = importlib.import_module(code_name)
        model = module.get_model(splits=self.splits, split_idx=self.split_idx)

        # Run the collaborator.
        agg_id = plan.agg_id
        fed_id = plan.fed_id
        # FIXME: need a mechanism to negotiate the col_id between agg and col.
        col_id = self.col_id
        opt_treatment = plan.opt_treatment
        agg_addr = plan.agg_addr
        agg_port = plan.agg_port

        connection = ZMQClient('{} connection'.format(col_id), server_addr=agg_addr, server_port=agg_port)

        col = Collaborator(col_id, agg_id, fed_id, model, connection, -1, opt_treatment=opt_treatment)
        col.run()

    def send_and_receive(self, message):
        self.connection.send(message)
        reply = self.connection.receive()
        return reply
