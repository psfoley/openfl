from tfedlrn.tensordict_to_proto_pipelines import Transformer, Pipeline




class TransformTensorDictToModelProto(Transformer):

    def __init__(self, **kwargs):
        return

    def forward(self, data, metadata, **kwargs):
        model_name, model_version
        return 

    def backward(self, data, **kwargs):
        return model_proto_to_float32_tensor_dict(model_proto=data)

class TensorDictToModelProto(Pipeline):
    
    def __init__(self, transformers=[TransformTensorDictToModelProto], **kwargs):
        super(TensorDictToModelProto, self).__init__(transformers=transformers)

    
