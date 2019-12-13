from tfedlrn.proto.message_pb2 import Model, Job, JobReply, JOB_TRAIN

def test_model():
	model = Model()
	# model.bar = 'trouble'
	print(model)

def test_job_reply():
	job_reply = JobReply(job=JOB_TRAIN)
	print(job_reply)
