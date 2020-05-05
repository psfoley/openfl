import numpy as np

from tfedlrn.tensor_dict_to_proto_pipelines import TensorDictToModelProtoPipeline
from tfedlrn.proto.collaborator_aggregator_interface_pb2 import MetaDataProto, MetaDatumProto

class RandomShiftTransformer(object):
    
    def __init__(self):
        return


    def forward(self, data, **kwargs):
        """
        Implement the data transformation.
        returns: transformed_data, metadata

        here data is a tensor_dict
        """
        transformed_data = {}
        metadatum_list = []
        for key, value in data.items():
            param_shape = value.shape
            random_shift = np.random.uniform(low=0, high=20, size=param_shape).astype(np.float32)
            transformed_data[key] = value + random_shift
            metadatum_name = key
            metadatum_int_to_float = {}
            for idx, val in enumerate(random_shift.flatten(order='C')):
                metadatum_int_to_float[idx] = val
            metadatum_list.append(MetaDatumProto(name=metadatum_name, int_to_float=metadatum_int_to_float))
        return transformed_data, MetaDataProto(metadatum_list=metadatum_list)



    def backward(self, data, metadata, **kwargs):
        """
        Implement the data transformation needed when going the oppposite
        direction to the forward method.
        returns: transformed_data
        """
        transformed_data = {}
        for entry in metadata.metadatum_list:
            param_name = entry.name
            param = data[param_name]
            param_shape = param.shape
            random_shift = np.reshape(np.array([entry.int_to_float[idx] for idx in range(len(entry.int_to_float))]), 
                                      newshape=param_shape, 
                                      order='C')
            transformed_data[param_name] = param - random_shift
        if transformed_data.keys() != data.keys():
            raise RuntimeError('The metadata should have contained info on all params and it did not!')
        return transformed_data 


class RandomShiftPipeline(TensorDictToModelProtoPipeline):
    
    def __init__(self, transformers=[RandomShiftTransformer()], **kwargs):
        super(RandomShiftPipeline, self).__init__(transformers=transformers)

    
