import numpy as np
import gzip
import copy
from sklearn import cluster

from tfedlrn.tensor_transformation_pipelines import TransformationPipeline, Transformer

class KmeansTransformer(Transformer):
    """ A transformer class to quantize input data.
    """
    def __init__(self, n_cluster=6):
        self.n_cluster = n_cluster
        return

    def forward(self, data, **kwargs):
        """ Quantize data into n_cluster levels of values.
        data: an numpy array from the model tensor_dict.
        int_data: an numpy array being quantized.
        metadata: dictionary to store a list meta information.
        """
        metadata = {}
        metadata['int_list'] = list(data.shape)
        # clustering
        k_means = cluster.KMeans(n_clusters=self.n_cluster, n_init=self.n_cluster)
        data = data.reshape((-1, 1))
        k_means.fit(data)
        quantized_values = k_means.cluster_centers_.squeeze()
        indices = k_means.labels_ 
        quant_array = np.choose(indices, quantized_values)
        int_array, int2float_map = self._float_to_int(quant_array)
        metadata['int_to_float']  = int2float_map
        return int_array, metadata

    def backward(self, data, metadata, **kwargs):
        """ Recover data array back to the original numerical type and the shape.
        data: an flattened numpy array.
        metadata: dictionary to contain information for recovering ack to original data array.
        data (return): an numpy array with original numerical type and shape.
        """
        # convert back to float
        # TODO
        data = copy.deepcopy(data)
        int2float_map = metadata['int_to_float']
        for key in int2float_map:
            indices = data == key
            data[indices] = int2float_map[key]
        data_shape = list(metadata['int_list'])
        data = data.reshape(data_shape)
        return data

    def _float_to_int(self, np_array):
        """ Creating look-up table for conversion between floating and integer types.
        """
        flatten_array = np_array.reshape(-1)
        unique_value_array = np.unique(flatten_array)
        int_array = np.zeros(flatten_array.shape, dtype=np.int)
        int_to_float_map = {}
        float_to_int_map = {}
        # create table
        for idx, u_value in enumerate(unique_value_array):
            int_to_float_map.update({idx: u_value})
            float_to_int_map.update({u_value: idx})
            # assign to the integer array
            indices = np.where(flatten_array==u_value)
            int_array[indices] = idx
        int_array = int_array.reshape(np_array.shape)
        return int_array, int_to_float_map
            
class GZIPTransformer(Transformer):
    """ A transformer class to losslessly compress data.
    """
    def __init__(self):
        return

    def forward(self, data, **kwargs):
        """ Compress data into bytes.
        """
        bytes_ = data.astype(np.float32).tobytes()
        compressed_bytes_ = gzip.compress(bytes_)
        metadata = {}
        return compressed_bytes_, metadata

    def backward(self, data, metadata, **kwargs):
        """ Decompress data into numpy of float32.
        """
        decompressed_bytes_ = gzip.decompress(data)
        data = np.frombuffer(decompressed_bytes_, dtype=np.float32)
        return data

class KCPipeline(TransformationPipeline):
    """ A pipeline class to compress data lossly using k-means methods.
    """
    
    def __init__(self, p_sparsity=0.01, n_clusters=6, **kwargs):
        """ Initializing a pipeline of transformers.
        """
        # instantiate each transformer
        self.p = p_sparsity
        self.n_cluster = n_clusters
        transformers = [KmeansTransformer(self.n_cluster), GZIPTransformer()]
        super(KCPipeline, self).__init__(transformers=transformers, **kwargs)
