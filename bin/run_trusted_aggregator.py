#!/usr/bin/env python3
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)

import argparse
import numpy as np
import json
from math import ceil

from scipy.spatial.distance import cdist, euclidean
from AGG_alpha import ReplyCode, Role, ZMQServer, Message, MessageType, Layer, verify_file


# TODO: add verification of signature on file
def get_work_defs():
    if not verify_file('./work_defs.json', './work_defs.sig', 'public.pem'):
        raise ValueError("Signature verification failure on work_defs.json!")

    with open('./work_defs.json', 'r') as f:
        work_defs = json.load(f)

    return work_defs


# FIXME: make this a round manager whose state can be saved
class WorkState(object):
    def __init__(self):
        self.version = 0
        self.clear_for_new_round()

        # <version>:
        # {
        #     <client_id>:
        #     {
        #         "results": <float>,
        #         "size": <int>
        #     }
        # }
        self.validation_results = {}

    def clear_for_new_round(self):
        # <role>: <client_id list>
        self.roles = {r: [] for r in Role}

        # <layer_id>: <client_id list>
        self.expected_layer_updates = {}

    def get_role(self, id):
        for r, l in self.roles.items():
            if id in l:
                return r
        return None


# wraps the work def dictionary for convenience
class WorkDef(object):
    def __init__(self, work_def_dict):
        self.w = work_def_dict

    def is_agg(self, id):
        return id == self.w['aggregator']['name']

    def is_col(self, id):
        return id in self.w['collaborators']

    def is_layer(self, id):
        return id in self.w['model config']['layer ids']

    def layers(self):
        return self.w['model config']['layer ids']

    def cols(self):
        return self.w['collaborators']

    # TODO: any policies/methods here?
    def n_training_for_round(self, round):
        return self.w['aggregation config']['COL per round']

    def aggregation_method(self):
        return self.w['aggregation config']['aggregation method']


class FederatedAggregator():

    # initialization and creation functions

    def __init__(self, id):
        self.id = id
        self.work_defs = {}
        self.work_state = {}

    def ensure_work_def_exists(self, work_id):
        if work_id not in self.work_defs:
            self.update_work_defs()

    # TODO: work state should be saved in case of crash
    def ensure_work_state_exists(self, work_id):
        if work_id not in self.work_state:
            # create a new work_state from our work_defs item
            self.work_state[work_id] = WorkState()
            # FIXME: this is not correct. Starting a round should happen somewhere else
            self.start_round(work_id)

    # TODO: can a work def for a given id change? Probably not, due to COL permission model
    def update_work_defs(self):
        work_defs = get_work_defs()
        for k, v in work_defs.items():
            if k not in self.work_defs:
                self.work_defs[k] = WorkDef(v)

    # ERROR CHECKS

    def error_check_message(self, message):
        self.ensure_work_def_exists(message.work_id)
        work_id = message.work_id

        if work_id not in self.work_defs:
            raise ValueError(f'{work_id} is not known to aggregator {self.id}. {message}')

        if not self.work_defs[work_id].is_agg(self.id):
            raise ValueError(f'{self.id} is not the correct aggregator for {message.work_id}. {message}')

        if message.recipient != self.id:
            raise ValueError(f'{self.id} not intended recipient for {message}')

        if not self.work_defs[message.work_id].is_col(message.sender):
            raise ValueError(f'Message from collaborator not part of work def. {message}')

        self.ensure_work_state_exists(message.work_id)

    # ROUND MANAGEMENT

    # FIXME: what about validation state? How to ensure trainers validated?
    # FIXME: this is spaghetti. Need either better OO design or something else
    def update_round_state(self, work_id):
        ws = self.work_state[work_id]

        # update training roles
        training_complete = []
        for col in ws.roles[Role.TRAIN]:
            done_training = True
            # if col not in any expected layer updates, move it to yield
            for layer_id, cols_for_layer in ws.expected_layer_updates.items():
                if col in cols_for_layer:
                    done_training = False
                    break
            if done_training:
                training_complete.append(col)
        for col in training_complete:
            ws.roles[Role.TRAIN].remove(col)
            ws.roles[Role.DONE].append(col)

        # update validation roles
        val_complete = []
        for col in ws.roles[Role.VALIDATE]:
            if col in ws.validation_results[ws.version]:
                val_complete.append(col)
        for col in val_complete:
            ws.roles[Role.VALIDATE].remove(col)
            ws.roles[Role.DONE].append(col)

        # if the round is complete, start the next round
        if self.round_complete(work_id):
            self.complete_and_start_round(work_id)

    def round_complete(self, work_id):
        ws = self.work_state[work_id]
        outstanding = 0
        for k, v in ws.roles.items():
            if k == Role.DONE:
                continue
            outstanding += len(v)
        return outstanding == 0

    def complete_and_start_round(self, work_id):
        assert self.round_complete(work_id)

        # for each training layer, load it and save it as complete with version + 1
        for layer_id in self.work_defs[work_id].layers():
            layer = Layer.load_training(work_id, layer_id)
            layer.version += 1
            layer.save_complete()

        ws = self.work_state[work_id]

        # TODO: what to do with the validation results?
        results = []
        sizes = []
        for v in ws.validation_results[ws.version].values():
            results.append(v['results'])
            sizes.append(v['size'])
        val_score = np.average(results, weights=sizes)
        print(f'{work_id} completed round {ws.version}. Previous val score: {val_score}')

        # clear the roles
        ws.clear_for_new_round()

        ws.version += 1
        self.start_round(work_id)

    # init structures and choose roles
    def start_round(self, work_id):
        work_state = self.work_state[work_id]
        work_def = self.work_defs[work_id]

        # create the empty dict for validation results
        work_state.validation_results[work_state.version] = {}

        # set the roles
        n_training = work_def.n_training_for_round(work_state.version)

        shuffled_cols = work_def.cols().copy()
        # FIXME: need a random seed for this
        np.random.shuffle(shuffled_cols)

        work_state.roles[Role.TRAIN].extend(shuffled_cols[:n_training])
        work_state.roles[Role.VALIDATE].extend(shuffled_cols[n_training:])

        for col in work_state.roles[Role.TRAIN]:
            for layer_id in work_def.layers():
                work_state.expected_layer_updates[layer_id] = shuffled_cols[:n_training].copy()

    # REQUEST HANDLING

    def handle_request(self, message):
        """must return a reply"""
        self.error_check_message(message)

        func_name = f'handle_{message.message_type.name.lower()}'
        return getattr(self, func_name)(message)

    def handle_version_request(self, message):
        version = self.work_state[message.work_id].version
        return self.create_reply(message, MessageType.VERSION, ReplyCode.OK, {'version': version})

    def handle_role_request(self, message):
        role = self.work_state[message.work_id].get_role(message.sender)
        return self.create_reply(message,
                                 MessageType.ROLE,
                                 ReplyCode.OK,
                                 role)

    def handle_layer_request(self, message):
        work_id = message.work_id
        layer_id = message.payload['layer_id']
        layer = Layer.load_complete(work_id, layer_id)

        if layer.version < message.payload['version']:
            raise ValueError(f'Erroneous request for future version of layer: {message.payload}. {message}')

        if layer.version > message.payload['version']:
            return self.create_reply(message,
                                     MessageType.REPLY_CODE,
                                     ReplyCode.STALE,
                                     None)

        return self.create_reply(message,
                                 MessageType.LAYER,
                                 ReplyCode.OK,
                                 layer)

    def handle_layer_update(self, message):
        col = message.sender
        layer = message.payload
        assert layer.work_id == message.work_id

        work_id = message.work_id
        version = self.work_state[work_id].version
        assert layer.version <= version

        if layer.version < version:
            return self.create_reply(message,
                                     MessageType.REPLY_CODE,
                                     ReplyCode.STALE,
                                     None)

        # ensure col should be training
        assert self.work_state[work_id].get_role(col) == Role.TRAIN

        # ensure col is expected to updated THIS layer
        layer_id = layer.layer_id
        assert col in self.work_state[work_id].expected_layer_updates[layer.layer_id]

        # load the training layer
        training_layer = Layer.load_training(work_id, layer_id)

        # if no training_layer or the stored version is less than the workstate version,
        # we need to overwrite the with this version
        if (training_layer is None or
           training_layer.version < self.work_state[work_id].version):
            training_layer = layer
        # else, aggregate it
        else:
            aggregation_method = self.get_aggregation_method(work_id)
            training_layer = self.aggregate_layers(training_layer, layer, aggregation_method)

        # save the layer
        training_layer.save_training()

        # remove the col from the list of expected_layer_updates
        self.work_state[work_id].expected_layer_updates[layer.layer_id].remove(col)

        # update round state
        self.update_round_state(work_id)

        # return the reply that layer was received
        return self.create_reply(message,
                                 MessageType.REPLY_CODE,
                                 ReplyCode.OK,
                                 None)

    def handle_validation_results(self, message):
        col = message.sender
        version = message.payload['version']
        work_id = message.work_id
        ws = self.work_state[work_id]

        if col in ws.validation_results[version]:
            raise ValueError(f'{col} already in validation results for round {version}. {ws.validation_results}')

        ws.validation_results[version][col] = message.payload[col]

        # update round state
        self.update_round_state(work_id)

        # return the reply that layer was received
        return self.create_reply(message,
                                 MessageType.REPLY_CODE,
                                 ReplyCode.OK,
                                 None)

    def create_reply(self, message, message_type, reply_code, payload):
        return Message(message_type,
                       message.work_id,
                       self.id,
                       message.sender,
                       payload,
                       reply_code=reply_code)

    # UNFILED FUNCTIONS

    def aggregate_layers(self, l1, l2, aggregation_method):
        assert l1.work_id == l2.work_id
        assert l1.layer_id == l2.layer_id
        assert l1.version == l2.version

        l1.parameters = aggregation_method([l1.parameters, l2.parameters],
                                           [l1.weight, l2.weight])
        l1.weight += l2.weight
        return l1

    def get_aggregation_method(self, work_id):
        return get_aggregation_method(**self.work_defs[work_id].aggregation_method())

    def run(self, port):
        connection = ZMQServer(self.id, server_port=port)
        while True:
            request = connection.receive()
            print(f'{self} received {request}')
            reply = self.handle_request(request)
            print(f'{self} replying {reply}')
            connection.send(reply)


class Aggregator(object):

    def __init__(self, work_id):
        self.work_id = work_id


def get_aggregation_method(method, n_buckets=None):
    if method == 'weighted_averaging':
        return weighted_averaging
    elif method == 'byzantine_gradient_descent':
        return partial(byzantine_gradient_descent, n_buckets=n_buckets)
    elif method == 'dpsgd':
        return NotImplementedError('DP-SGD not yet implemented')
    else:
        return ValueError(f'{method} not a valid method')


def weighted_averaging(parameters, weights):
    return np.average(parameters, axis=0, weights=weights)


def byzantine_gradient_descent(updates, weights, n_buckets=None):
    if n_buckets is None:
        n_buckets = ceil(np.sqrt(len(updates)))

    bucket_assignments = incremental_subset_sum(weights, n_buckets)

    # compute the average for each bucket
    buckets = []
    for a in bucket_assignments:
        # a is in array indexing updates and weights
        updates_for_bucket = [updates[b] for b in a]
        weights_for_bucket = [weights[b] for b in a]
        bucket = weighted_averaging(updates_for_bucket, weights_for_bucket)
        buckets.append(bucket)

    # compute the layer-wise geometric median
    return [geometric_median(np.array(x)) for x in zip(*buckets)]


def geometric_median(X, threshold=1e-4, eps=1e-16):
    shape = X.shape
    X = X.reshape(X.shape[0], -1)

    guess = np.mean(X, 0)

    while True:
        distances = cdist(X, [guess]).reshape(-1)
        distances[distances == 0] = eps
        weights = 1 / distances
        next_guess = np.average(X, weights=weights, axis=0)

        if euclidean(guess, next_guess) < threshold:
            return next_guess.reshape(shape[1:])

        guess = next_guess


if __name__ == '__main__':
    # port is 5555
    fa = FederatedAggregator("alpha_agg")
    fa.run(5555)
