import numpy as np
import gzip

from tfedlrn.tensor_transformation_pipelines import TransformationPipeline, Transformer

class SparsityTransformer(Transformer):
    
    def __init__(self):
        return


    def forward(self, data, **kwargs):
        """
        Implement the data transformation.
        returns: transformed_data, metadata

        here data is an array value from a model tensor_dict
        """
        shape = data.shape
        random_shift = np.random.uniform(low=-20, high=20, size=shape).astype(np.float32)
        transformed_data = data + random_shift
        
        # construct metadata
        metadata = {}
        for idx, val in enumerate(random_shift.flatten(order='C')):
            metadata[idx] = val
        
        # input::np_array, {}
        # output::np_array, {}
        return transformed_data, metadata



    def backward(self, data, metadata, **kwargs):
        """
        Implement the data transformation needed when going the oppposite
        direction to the forward method.
        returns: transformed_data
        """
        
        shape = data.shape
        # this is an awkward use of the metadata into to float dict, usually it will
        # trully be treated as a dict. Here (and in 'forward' above) we use it essentially as an array.
        shift = np.reshape(np.array([metadata[idx] for idx in range(len(metadata))]), 
                                    newshape=shape, 
                                    order='C')
        return data - shift 

class TernaryTransformer(Transformer):
    def __init__(self):
        return

    def foraward(self, data, kwargs**):
        '''
        ...............................
        Quantization:
        [4.234324, 2.23432, -2.23432, -4.23432]

        dict:maping 
        [4.234324, 2.23432, -2.23432, -4.23432]
        [0, 1, 2, 3]

        matrix;
        {1.232, ...}
        {0, 1, 2, 2,... ...}
        table mapping:dict{}
        ...............................
        quantization
        [1, 0, -1, 1, ...]
        representation with a table
            value set: {1,0,-1}
            table: {1:0.234, 0:0, -1:-0.23}
            table : {1, [(id1, id2,id2)]}
        '''
        pass

    def backward(self, data, metadata, kwargs**):
        pass

class GZIPTransformer(Transformer):
    '''
    How to reshape the integer value np_array?
    np_array -> bytes
    using object compression?
    input::
    output::
    '''
    def __init__(self):
        return

    def foraward(self, data, kwargs**):
        bytes_ = data.tobytes()
        compressed_bytes_ = gzip.compress(bytes_)
        shape_info = data.shape
        return compressed_bytes_

    def backward(self, data_bytes, metadata, kwargs**):
        decompressed_bytes_ = gzip.decompress(data_bytes)
        data = np.frombuffer(decompressed_bytes_, dtyp=np.float32)
        data = data.reshape(meta_data['dtype'])
        return data

class STCPipeline(TransformationPipeline):
    
    def __init__(self, transformers=[RandomShiftTransformer()], **kwargs):
        super(RandomShiftPipeline, self).__init__(transformers=transformers)

    def foraward(self, data, kwargs**):
        pass

    def backward(self, data, metadata, kwargs**):
        pass
    
