import numpy as np
import gzip

from tfedlrn.tensor_transformation_pipelines import TransformationPipeline, Transformer

'''
dictionary:
    'int_to_float': 2.23432
    'int_list': [128,32,32,3]
    'bool_array': [True, Flase, ...]

---

forward:
        shape
        x = flatten(x)
    sparse(x): topk, metadata = {'int_list': [123...], 'bool_array': np.array(indices )}
    quant(topk): int_array, metadata={'int_float':, int2float_map}
    gzip:(int_array): compressed_bytes; 

backward:
    gzip(compressed_bytes): int_array(flattened)
    quant(int_array, metadata): float_array
    sparse(float_array, metadata): x_original_sparse

'''

class SparsityTransformer(Transformer):
    
    def __init__(self):
        self.p = 0.01 
        self.n_cluster = None
        return

    def forward(self, data, **kwargs):
        """
        Implement the data transformation.
        returns: transformed_data, metadata

        here data is an array value from a model tensor_dict
        """
        '''
        w: model weights, numpy array
        p: sparsity ratio
        '''
        # sparsification
        meta_dict['original_shape'] = data.shape
        n_elements = data.flatten().shape[0]
        k_op = int(np.ceil(n_elements*p))
        flatten_w = w.flatten()
        topk, topk_indices = self._topk_func(flatten_w, k_op)
        #
        meta_dict = {}
        meta_dict['topk'] = [topk]
        meta_dict['topk_indices'] = [topk_dices]
        # make a sparse data
        return sarpse_data, meta_dict
        '''
        #
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
        '''

    def backward(self, data, metadata, **kwargs):
        """
        Implement the data transformation needed when going the oppposite
        direction to the forward method.
        returns: transformed_data
        ""andomShiftPipeline
        return data
        
        '''
        shape = data.shape
        # this is an awkward use of the metadata into to float dict, usually it will
        # trully be treated as a dict. Here (and in 'forward' above) we use it essentially as an array.
        shift = np.reshape(np.array([metadata[idx] for idx in range(len(metadata))]), 
                                    newshape=shape, 
                                    order='C')
        return data - shift 
        '''

    def _topk_func(self, x, k):
        # quick sort as default on magnitude
        idx = np.argsort(np.abs(x))
        # sorted order, the right most is the largest magnitude
        length = x.shape[0]
        start_idx = length - k
        # get the top k magnitude
        result = x[idx[start_idx:]]
        indices = idx[start_idx:]
        return result, indices

class TernaryTransformer(Transformer):
    def __init__(self):
        return

    def foraward(self, data, topk, kwargs**):
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
        results = self.ternary_quant(data, topk)
        return

    def backward(self, data, metadata, kwargs**):
        return data

    def ternary_quant(self, w, topk, plot_flag=False):
        # equation: mean * sign(w_masked)
        threshold = min(np.abs(topk))
        mean_topk = np.mean(abs(topk))
        print('threshold: ', threshold)
        print('mean_topk:', mean_topk)
        out_ = np.where(w >= threshold, mean_topk, 0.0)
        out = np.where(w <= -threshold, -mean_topk, out_)
        int_array, int2float_map = self._float_to_int(out)
        return int_array, int2float_map

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
        #data = data.reshape(metadata['shape'])
        return data

class STCPipeline(TransformationPipeline):
    
    def __init__(self, transformers=[RandomShiftTransformer()], **kwargs):
        # instantiate each transformer, name
        self.p = p_sparsity
        self.p = p_sparsity
        super(RandomShiftPipeline, self).__init__(transformers=transformers)
        # transformers
        #fun(**kwargs)
        #no::self.p = kwargs['p_sparsity']

    def foraward(self, data, kwargs**):
        pass

    def backward(self, data, metadata, kwargs**):
        pass
    
