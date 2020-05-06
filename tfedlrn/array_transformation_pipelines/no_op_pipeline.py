from tfedlrn.array_transformation_pipelines import TransformationPipeline




class NoOpPipeline(TransformationPipeline):
    
    def __init__(self, transformers=[], **kwargs):
        super(NoOpPipeline, self).__init__(transformers=transformers)

    
