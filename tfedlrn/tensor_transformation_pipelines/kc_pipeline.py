import numpy as np
import gzip
import copy
from sklearn import cluster

from tfedlrn.tensor_transformation_pipelines import TransformationPipeline, Transformer

class KmeansTransformer(Transformer):
    def __init__(self, n_cluster=6):
        self.n_cluster = n_cluster
        return

    def forward(self, data, **kwargs):
        '''
        '''
        # clustering
        k_means = cluster.KMeans(n_clusters=self.n_cluster, n_init=self.n_cluster)
        k_means.fit(data)
        quantized_values = k_means.cluster_centers_.squeeze()
        indices = k_means.labels_ 
        quant_array = np.choose(indices, quantized_values)
        int_array, int2float_map = self._float_to_int(quant_array)
        metadata = {}
        metadata['int_to_float']  = int2float_map
        return int_array, metadata
        '''
        # ternarization, data is sparse and flattened
        mean_topk = np.mean(np.abs(data))
        out_ = np.where(data > 0.0, mean_topk, 0.0)
        out = np.where(data < 0.0, -mean_topk, out_)
        int_array, int2float_map = self._float_to_int(out)
        metadata = {}
        metadata['int_to_float']  = int2float_map
        return int_array, metadata
        '''

    def backward(self, data, metadata, **kwargs):
        # convert back to float
        # TODO
        data = copy.deepcopy(data)
        int2float_map = metadata['int_to_float']
        for key in int2float_map:
            indices = data == key
            data[indices] = int2float_map[key]
        return data

    def _float_to_int(self, np_array):
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
            
    def _int_to_float(self, np_array, int_to_float_map):
        flatten_array = np_array.reshape(-1)
        unique_value_array = np.unique(flatten_array)
        float_array = np.zeros(flatten_array.shape, dtype=np.int)
        # create table
        for idx, int_value in enumerate(unique_value_array):
            float_value = int_to_float_map(int_value)
            indices = np.where(np_array==int_value)
            float_array[indices] = float_value
        float_array = float_array.reshape(np_array.shape)
        return int_array, int_to_float_map

class GZIPTransformer(Transformer):
    '''
    '''
    def __init__(self):
        return

    def forward(self, data, **kwargs):
        bytes_ = data.astype(np.float32).tobytes()
        compressed_bytes_ = gzip.compress(bytes_)
        #shape_info = data.shape
        metadata = {}
        return compressed_bytes_, metadata

    def backward(self, data, metadata, **kwargs):
        decompressed_bytes_ = gzip.decompress(data)
        data = np.frombuffer(decompressed_bytes_, dtype=np.float32)
        return data

class KCPipeline(TransformationPipeline):
    
    def __init__(self, p_sparsity=0.01, n_clusters=6, **kwargs):
        # instantiate each transformer
        self.p = p_sparsity
        self.n_cluster = n_clusters
        transformers = [KmeansTransformer(self.n_cluster), GZIPTransformer()]
        super(KCPipeline, self).__init__(transformers=transformers, **kwargs)
