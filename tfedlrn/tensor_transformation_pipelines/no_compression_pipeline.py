from tfedlrn.tensor_transformation_pipelines import TransformationPipeline, Transformer

import numpy as np

class Float32NumpyArrayToBytes(Transformer):
    def __init__(self, **kwargs):
        pass

    def forward(self, data, **kwargs):
        array_shape = data.shape
        metadata = {'int_list': list(array_shape)}
        data_bytes = data.tobytes(order='C')
        return data_bytes, metadata

    def backward(self, data, metadata):
        array_shape = tuple(metadata['int_list'])
        # DEBUG
        print(array_shape)
        flat_array = np.frombuffer(data, dtype=np.float32)
        return np.reshape(flat_array, newshape=array_shape, order='C')


class NoCompressionPipeline(TransformationPipeline):
    
    def __init__(self, **kwargs):
        super(NoCompressionPipeline, self).__init__(transformers=[Float32NumpyArrayToBytes()], **kwargs)

    
