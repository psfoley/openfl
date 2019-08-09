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
from ..proto.message_pb2 import *


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
    def __init__(self, connection, plans_folder):
        self.connection = connection
        self.logger = logging.getLogger(__name__)
        self.plans = self.search_and_load_plans(plans_folder)        

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
            bytes = open(fpath, "rb").read()
            checksum = hashlib.sha256(bytes).hexdigest()
            size = len(bytes)
            return CodeReply(size=size, 
                        checksum=checksum,
                        bytes=bytes)

    def load_plan(self, path):
        plan = None
        with open(path, 'r') as f:
            try:
                plan = yaml.safe_load(f.read())
            except Exception:
                self.logger.error("Invalid federated learning plan [%s]." % path, exc_info=True)
        return plan
    
    def search_and_load_plans(self, working_dir):
        plans = {}
        plan_files = sorted(pathlib.Path(working_dir).glob("*.yaml"))
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
    def __init__(self, connection, dataset_name, software_version, models_folder):
        self.logger = logging.getLogger(__name__)
        self.connection = connection
        self.polling_interval = 4
        self.counter = 0

        self.dataset_name = dataset_name
        self.software_version = software_version
        
        self.models_folder = models_folder
        self.init_models_folder()

    def init_models_folder(self):
        models_folder = self.models_folder
        if not os.path.isdir(models_folder):
            os.makedirs(models_folder)
        sys.path.append(models_folder)

    def has_code(self, model_name):
        # FIXME: check the integrity with checksum.
        if os.path.isdir(os.path.join(self.models_folder, model_name)):
            return True
        
    def download_code(self, fed_id):
        # FIXME: 1. download code; 2. file size and checksum; 3. extend zip locally.
        reply = self.send_and_receive(CodeRequest(fed_id=fed_id))
        reply.size
        reply.checksum
        reply.code_zip


    def run(self):
        self.logger.debug("Client coordinator connects to Server coordinator at [%s]." % (self.connection.server_addr))
        while True:
            # query for plan
            # breaks when a plan has been received
            # plan = self.query_for_plan()
            plan = self.send_and_receive(PlanRequest(dataset=self.dataset_name, software_version=self.software_version))

            if plan.fed_id == "invalid":
                time.sleep(self.polling_interval)
            else:
                break

        self.logger.debug("Got a plan [%s]" % plan.fed_id )
        code_name = plan.code
        if not self.has_code(code_name):
            self.logger.debug("Start to download the code.")
            self.download_code(code_name)
            self.logger.debug("Downloaded the code.")

        module = importlib.import_module(code_name)
        model = module.get_model()

        # Run the collaborator.
        agg_id = plan.agg_id
        fed_id = plan.fed_id
        # col_id = plan.col_id
        # FIXME: need a mechanism to negotiate the col_id between agg and col.
        col_id = "0"
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
