import numpy as np

from tfedlrn.tensor_transformation_pipelines import TransformationPipeline

class RandomShiftTransformer(object):
    
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


class RandomShiftPipeline(TransformationPipeline):
    
    def __init__(self, transformers=[RandomShiftTransformer()], **kwargs):
        super(RandomShiftPipeline, self).__init__(transformers=transformers)

    
