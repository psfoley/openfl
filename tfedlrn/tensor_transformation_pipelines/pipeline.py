
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
        Implement the data transformation needed when going the opposite
        direction to the forward method.
        returns: transformed_data
        """
        raise NotImplementedError


class TransformationPipeline(object):
    """
    A sequential pipeline to transform (e.x. compress) data (e.x. layer of model_weights) as well as return
    metadata (if needed) for the reconstruction process carried out by the backward method. 
    """

    def __init__(self, transformers, **kwargs):
        self.transformers = transformers

    def forward(self, data, **kwargs):
        transformer_metadata = []
        '''
        # datafomat::numpy::float.32
        # model proto:: a collection of tensor_dict proto
        # protobuff::-> a layer of weights
        # input::tensor_dict:{"layer1":np.array(float32, [128,3,28,28]), "layer2": np.array()}
        # output::meta data::numpy::int array
        # (data, transformer_metadata)::(float32, dictionary of key+float32 vlues)
        # input:: numpy_data (float32)
        # input:: (data(bytes), transformer_metadata_list::a list of dictionary from int to float)
        '''
        metadata_list = []
        data = data.copy()
        for transformer in self.transformers:
            data, metadata = transformer.forward(data=data, **kwargs)
            transformer_metadata.append(metadata)
        return data, transformer_metadata

    def backward(self, data, transformer_metadata, **kwargs):
        for transformer in self.transformers[::-1]:
            data = transformer.backward(data=data, metadata=transformer_metadata.pop(), **kwargs)
        return data
            