import numpy as np
import gzip
import copy
from sklearn import cluster

from tfedlrn.tensor_transformation_pipelines import TransformationPipeline, Transformer

class SparsityTransformer(Transformer):
    
    def __init__(self, p=0.01):
        self.p = p
        return

    def forward(self, data, **kwargs):
        """
        Implement the data transformation.
        returns: transformed_data, metadata

        here data is an array value from a model tensor_dict
        """
        print('======================================')
        self.p = 1
        print('sparsity::', self.p)
        print('raw data::', data)
        print('raw data::', data.shape)
        print('======================================')
        metadata = {}
        metadata['int_list'] = list(data.shape)
        # sparsification
        data = data.astype(np.float32)
        flatten_data = data.flatten()
        n_elements = flatten_data.shape[0]
        k_op = int(np.ceil(n_elements*self.p))
        topk, topk_indices = self._topk_func(flatten_data, k_op)
        #
        condensed_data = topk
        sparse_data = np.zeros(flatten_data.shape)
        sparse_data[topk_indices] = topk 
        nonzero_element_bool_indices = sparse_data != 0.0
        metadata['bool_list'] = list(nonzero_element_bool_indices)
        print('======================================')
        print('forward::')
        print('condensed_data::', condensed_data)
        print('======================================')
        return condensed_data, metadata
        #return sparse_data, metadata

    def backward(self, data, metadata, **kwargs):
        """
        Implement the data transformation needed when going the oppposite
        direction to the forward method.
        returns: transformed_data
        """
        print('======================================')
        print('backward::')
        print('condensed_data::', data)
        print('======================================')
        data = data.astype(np.float32)
        data_shape = metadata['int_list']
        nonzero_element_bool_indices = list(metadata['bool_list'])
        recovered_data = np.zeros(data_shape).reshape(-1).astype(np.float32)
        recovered_data[nonzero_element_bool_indices] = data
        recovered_data = recovered_data.reshape(data_shape) 
        return recovered_data
        

    def _topk_func(self, x, k):
        # quick sort as default on magnitude
        idx = np.argsort(np.abs(x))
        # sorted order, the right most is the largest magnitude
        length = x.shape[0]
        start_idx = length - k
        # get the top k magnitude
        topk_mag = np.asarray(x[idx[start_idx:]])
        indices = np.asarray(idx[start_idx:])
        if min(topk_mag)-0 < 10e-8:# avoid zeros
            topk_mag = topk_mag + 10e-8
        return topk_mag, indices

class KmeansTransformer(Transformer):
    def __init__(self, n_cluster=6):
        self.n_cluster = n_cluster
        return

    def forward(self, data, **kwargs):
        '''
        '''
        # clustering
        print('1data::', data.shape)
        data = data.reshape((-1,1))
        print('2data::', data.shape)
        k_means = cluster.KMeans(n_clusters=self.n_cluster, n_init=self.n_cluster)
        k_means.fit(data)
        quantized_values = k_means.cluster_centers_.squeeze()
        indices = k_means.labels_ 
        quant_array = np.choose(indices, quantized_values)
        int_array, int2float_map = self._float_to_int(quant_array)
        metadata = {}
        metadata['int_to_float']  = int2float_map
        print('3int_array::', int_array.shape)
        int_array = int_array.reshape(-1)
        print('4int_array::', int_array.shape)
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

class SKCPipeline(TransformationPipeline):
    
    def __init__(self, p_sparsity=0.01, n_clusters=6, **kwargs):
        # instantiate each transformer
        self.p = p_sparsity
        self.n_cluster = n_clusters
        transformers = [SparsityTransformer(self.p), KmeansTransformer(self.n_cluster), GZIPTransformer()]
        super(SKCPipeline, self).__init__(transformers=transformers, **kwargs)
