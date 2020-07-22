# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


from tfedlrn.proto.message_pb2 import Model, Job, JobReply, JOB_TRAIN

def test_model():
	"""Unit test for the Model
	"""
	model = Model()
	# model.bar = 'trouble'
	print(model)

def test_job_reply():
	"""Unit test for job reply
	"""
	job_reply = JobReply(job=JOB_TRAIN)
	print(job_reply)
