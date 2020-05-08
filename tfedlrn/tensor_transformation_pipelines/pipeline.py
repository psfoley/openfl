
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
    A sequential pipeline to transform (e.x. compress) a numpy array (layer of model_weights) as well as return
    metadata (if needed) for the (potentially lossy) reconstruction process carried out by the backward method. 
    """

    def __init__(self, transformers, **kwargs):
        self.transformers = transformers

    def forward(self, data, **kwargs):
        metadata_list = []
        data = data.copy()
        for transformer in self.transformers:
            data, metadata = transformer.forward(data=data, **kwargs)
            metadata_list.append(metadata)
        return data, metadata_list

    def backward(self, data, transformer_metadata, **kwargs):
        data = data.copy()
        for transformer in self.transformers[::-1]:
            data = transformer.backward(data=data, metadata=transformer_metadata.pop(), **kwargs)
        return data
            
