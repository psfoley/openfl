#!/usr/bin/env python3
import sys
import os

from math import ceil

import time
import argparse

import numpy as np


    #     version = 0
    # sess = None

    # seed = 42

    # sess = reset_session(sess)

    # learning_rate = 1e-4
    
    # model = create_model(model='bratsunet',
    #                      loss_func='dice_correct',
    #                      dice_intersection_weight=2.0,
    #                      learning_rate=1e-4,
    #                      smooth=32.0,
    #                      seed=seed)

    # np.random.seed(seed)
    # model.tf_random_seed = seed

    # sess.run(tf.global_variables_initializer())

    # # load the data
    # imgs, msks = _get_data_per_institution_and_fold(xval_folds=5, seed=42)

    # # use fold 0 as test

    # # FIXME: remove this once test is done
    # fXs = []
    # fys = []
    # pXs = []
    # pys = []
    # for i, m in zip(imgs[id_num::2], msks[id_num::2]):
    #     fXs.append(np.concatenate(i[1:]))
    #     fys.append(np.concatenate(m[1:]))
    #     pXs.append(i[0])
    #     pys.append(m[0])

    # fX = np.concatenate(fXs)
    # fy = np.concatenate(fys)
    # pX = np.concatenate(pXs)
    # py = np.concatenate(pys)

    # # fX = np.concatenate(imgs[id_num][1:])
    # # fy = np.concatenate(msks[id_num][1:])
    # # pX = imgs[id_num][0]
    # # py = msks[id_num][0]

    # print(f'{fX.shape[0] / 155} training patients')

    # batch_size = 64
    # test_batches = ceil(pX.shape[0] / batch_size)
    # test_batches = [(pX[i::test_batches], py[i::test_batches]) for i in range(test_batches)]

    # connection = ZMQClient(id_num,
    #              server_addr='127.0.0.1',
    #              server_port=5555)

    # while True:

    #     # query for job
    #     job = query_for_job(id_num)

    #     # request version from aggregator
    #     agg_version = get_agg_version(connection)

    #     # if out of date, get the new parameters
    #     if agg_version > version:
    #         parameters, opt_parameters = get_parameters(connection, agg_version)
    #         model.run_assign_ops(sess, parameters)
    #         model.run_opt_assign_ops(sess, opt_parameters)
    #         version = agg_version

    #     # get our role
    #     role = get_role(connection)

    #     # if training or validating, get the score
    #     if role == Role.TRAIN or role == Role.VALIDATE:
    #         metric = 0
    #         for x, y in test_batches:
    #             _, m = model.run_test_batch(sess, x, y)
    #             metric += m / len(test_batches)
    #         size = pX.shape[0]
    #         send_validation_results(connection, metric, size, version)

    #     if role == Role.TRAIN:
    #         loss = model.train_epoch(sess, fX, fy, batch_size, learning_rate)
    #         print('update for version {} completed with loss {:.04}'.format(version, loss))
    #         # get and send the parameter values
    #         parameters, opt_parameters = model.get_parameters(sess)
    #         send_update(connection, parameters + opt_parameters, version, fX.shape[0])

    #     if role == Role.DONE:
    #         time.sleep(10)

    # close_session(sess)


class Collaborator(object):

    def __init__(self, id, data_path, agg_id, connection):
        self.id = id
        self.agg_id = agg_id
        self.connection = connection
        self.model_version = 0

        self.model = TF2DUnet(data_path)

    def run(self):
        while True:
            # query for job
            job = self.query_for_job()

            # if time to quit
            if job is CollaboratorJob.QUIT:
                print(f'{self} quitting')
                break
            elif job is CollaboratorJob.TRAIN:
                self.do_train_job()
            elif job is CollaboratorJob.YIELD:                
                time.sleep(4)
            elif job is CollaboratorJob.VALIDATE:
                self.do_validate_job()
            elif job is CollaboratorJob.DOWNLOAD_MODEL:
                self.do_download_model_job()

    def send(self, message_type, payload):
        self.connection.send(Message(message_type, self.id, self.agg_id, payload))
        reply = self.connection.receive()

        # check message is from agg to me
        assert reply.sender == self.agg_id and reply.recipient == self.id

        # check that the reply type matches the request type
        if message_type == MessageType.JOB_REQUEST:
            assert reply.message_type == MessageType.JOB_REPLY
        elif message_type == MessageType.TRAIN_UPDATE:
            assert reply.message_type == MessageType.TRAIN_ACK
        elif message_type == MessageType.VALIDATE_UPDATE:
            assert reply.message_type == MessageType.VALIDATE_ACK
        elif message_type == MessageType.MODEL_DOWNLOAD_REQUEST:
            assert reply.message_type == MessageType.MODEL_DOWNLOAD_REPLY
        else:
            raise NotImplementedError("No implementation to check reply type for message type: {}".format(message_type))

        return reply

    def query_for_job(self):
        reply = self.send(MessageType.JOB_REQUEST, self.model_version)
        return reply.payload

    def do_train_job(self):
        loss = self.model.train_epoch()
        # FIXME: make this a delta
        model_parameters = self.model.get_model_parameters()
        optimizer_parameters = self.model.get_optimizer_parameters()
        data_size = self.model.get_training_data_size()
        payload = {'model_parameters': model_parameters,
                   'optimizer_parameters': optimizer_parameters,
                   'loss': loss,
                   'data_size': data_size}
        self.send(MessageType.TRAIN_UPDATE, payload)

    def do_validate_job(self):
        score = self.model.validate()
        data_size = self.model.get_validation_data_size()
        payload = {'score': score, 'data_size': data_size}
        self.send(MessageType.VALIDATE_UPDATE, payload)

    def do_download_model_job(self):
        reply = self.send(MessageType.MODEL_DOWNLOAD_REQUEST, payload)
        self.version = payload['version']
        self.model.set_model_parameters(payload['model_parameters'])
        self.model.set_optimizer_parameters(payload['optimizer_parameters'])


def main(id=0, agg_id=None, data_path='/raid/datasets/BraTS17/by_institution', aggregator_address='127.0.0.1', aggregator_port=5555):
    connection = ZMQClient(id,
                 server_addr=aggregator_address,
                 server_port=aggregator_port)

    if agg_id is None:
        agg_id = '{}:{}'.format(aggregator_address, aggregator_port)

    data_path = os.path.join(data_path, id)

    collaborator = Collaborator(id, data_path, agg_id, connection)
    collaborator.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', '-i', type=int, required=True)
    parser.add_argument('--aggregator_address', '-aa', type=str, required=True)
    parser.add_argument('--aggregator_port', '-ap', type=int, required=True)
    parser.add_argument('--agg_id', '-ai', type=str, default=None)
    parser.add_argument('--data_path', '-ai', type=str, default='/raid/datasets/BraTS17/by_institution', required=True)
    args = parser.parse_args()
    main(**vars(args))
