from tfedlrn.tensordict_to_proto_pipelines import TensorDictToModelProtoPipeline




class NoCompressionPipeline(TensorDictToModelProtoPipeline):
    
    def __init__(self, transformers=[], **kwargs):
        super(NoCompressionPipeline, self).__init__(transformers=transformers)

    
