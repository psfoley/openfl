from tfedlrn.tensor_db.tensor_db import TensorDB
from tfedlrn import TensorKey
import numpy as np

def test_tensor_db_init():
    tensor_db = TensorDB()
    assert('Empty DataFrame' in str(tensor_db))

def test_tensor_db_single_insert():
    tensor_db = TensorDB()
    tensor_key = TensorKey('conv1','col_1',0,'model')
    tensor_key_dict = {tensor_key: np.arange(10)}
    tensor_db.cache_tensor(tensor_key_dict)
    assert('model' in str(tensor_db))

def test_tensor_db_multiple_insert():
    tensor_db = TensorDB()
    tensor_key = TensorKey('conv1','col_1',0,'model')
    tensor_key2 = TensorKey('conv1','col_1',0,'trained')
    tensor_key_dict = {tensor_key: np.arange(10),tensor_key2: np.arange(10)*2}
    tensor_db.cache_tensor(tensor_key_dict)
    assert('model' in str(tensor_db) and 'trained' in str(tensor_db))


def test_tensor_db_retrieval():
    tensor_db = TensorDB()
    tensor_key = TensorKey('conv1','col_1',0,'model')
    nparray = np.arange(10)
    tensor_key_dict = {tensor_key: nparray}
    tensor_db.cache_tensor(tensor_key_dict)
    print(tensor_db)
    returned_nparray = tensor_db.get_tensor_from_cache(tensor_key)
    assert(np.array_equal(nparray,returned_nparray))

def test_tensor_db_aggregation():
    tensor_db = TensorDB()
    tensor_key = TensorKey('conv1','agg',0,('model','col_1'))
    nparray = np.arange(10)
    tensor_key_dict = {tensor_key: nparray}
    tensor_db.cache_tensor(tensor_key_dict)
    tensor_key2 = TensorKey('conv1','agg',0,('model','col_2'))
    nparray2 = np.arange(10)*3
    tensor_key_dict2 = {tensor_key2: nparray2}
    tensor_db.cache_tensor(tensor_key_dict2)
    #Lookup tensorkey
    lookup_tk = TensorKey('conv1','agg',0,'model')
    returned_nparray = tensor_db.get_aggregated_tensor(lookup_tk, collaborator_weight_dict={'col_1':0.5,'col_2':0.5})
    #The returned value should be the average of the two numpy arrays
    assert(np.array_equal(nparray*2,returned_nparray))

def test_tensor_db_aggregation_incomplete():
    tensor_db = TensorDB()
    tensor_key = TensorKey('conv1','agg',0,('model','col_1'))
    nparray = np.arange(10)
    tensor_key_dict = {tensor_key: nparray}
    tensor_db.cache_tensor(tensor_key_dict)
    tensor_key2 = TensorKey('conv1','agg',0,('model','col_2'))
    nparray2 = np.arange(10)*3
    tensor_key_dict2 = {tensor_key2: nparray2}
    tensor_db.cache_tensor(tensor_key_dict2)
    #Lookup tensorkey
    lookup_tk = TensorKey('conv1','agg',0,'model')
    returned_nparray = tensor_db.get_aggregated_tensor(lookup_tk, collaborator_weight_dict={'col_1':0.25,'col_2':0.5,'col_3':0.25})
    #The returned value should be None because the 3rd collaborator hasn't reported results
    assert(returned_nparray == None)

