from tfedlrn.proto.protoutils import model_proto_to_float32_tensor_dict, construct_model_proto

class Transformer(object):
    
    def __init__(self):
        raise NotImplementedError


    def forward(self, data, **kwargs):
        """
        Implement the data transformation.
        returns: transformed_data, metadata
        """
        raise NotImplementedError

    def backward(self, data, metadata, **kwargs):
        """
        Implement the data transformation needed when going the oppposite
        direction to the forward method.
        returns: transformed_data
        """
        raise NotImplementedError


class TensorDictToModelProtoPipeline(object):
    """
    A pipeline from model tensor dict to model protobuf. The last stage is defined here, and when run 
    forward converts from:transformed(eg. compressed tensors) data,  metadata list, model header info, into a model protobuf. 
    Running the pipeline backward, the last stage reconstructs: list of metadata, transformed data from the proto, 
    each stage then subsequently pops off an entry of the metadata list and uses it to transform the data. 
    """

    def __init__(self, transformers, **kwargs):
        self.transformers = transformers

    def forward(self, tensor_dict, model_id, model_version, **kwargs):
        stage_metadata = []
        data = tensor_dict.copy()
        for transformer in self.transformers:
            data, metadata = transformer.forward(data=data, **kwargs)
            stage_metadata.append(metadata)
        return construct_model_proto(tensor_dict=data, 
                                     model_id=model_id, 
                                     model_version=model_version, 
                                     stage_metadata=stage_metadata)

    def backward(self, model_proto, **kwargs):
        data = model_proto_to_float32_tensor_dict(model_proto)
        stage_metadata = model_proto.stage_metadata
        print('TESTING')
        print(type(stage_metadata))
        print('TESTING')
        for transformer in self.transformers[::-1]:
            data = transformer.backward(data=data, metadata=stage_metadata.pop(), **kwargs)
        return data
            
