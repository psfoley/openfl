# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from tfedlrn.tensor_transformation_pipelines import TensorCodec,NoCompressionPipeline,RandomShiftPipeline,\
                                                              STCPipeline,SKCPipeline,KCPipeline
from tfedlrn import TensorKey
import numpy as np

def test_require_lossless():
    #STC is a lossy pipeline
    tensor_codec = TensorCodec(STCPipeline())
    tensor_key = TensorKey('conv1','col_1',0,('model',))
    nparray = np.arange(10000).reshape(100,100)
    compressed_tk, compressed_nparray, metadata = tensor_codec.compress(tensor_key,nparray,require_lossless=True)
    #A tag of 'compressed' means the tensor has been losslessly compressed
    assert('compressed' in compressed_tk[3])
    decompressed_tk, decompressed_nparray = tensor_codec.decompress(compressed_tk, compressed_nparray, metadata)
    #Both the decompressed tensor key and data should be identical to the original 
    assert(decompressed_tk == tensor_key)
    assert(np.array_equal(nparray,decompressed_nparray))

def test_lossy_tag():
    #STC is a lossy pipeline
    tensor_codec = TensorCodec(STCPipeline())
    tensor_key = TensorKey('conv1','col_1',5,('trained',))
    nparray = np.arange(10000).reshape(100,100)
    compressed_tk, compressed_nparray, metadata = tensor_codec.compress(tensor_key,nparray)
    #A tag of 'lossy_compressed' means the tensor has been transformed through a lossy pipeline
    assert('lossy_compressed' in compressed_tk[3])

def test_lossless_pipeline_detection():
    #RandomShift is a lossless pipeline
    tensor_codec = TensorCodec(RandomShiftPipeline())
    tensor_key = TensorKey('conv1','col_1',0,('model','some_tag'))
    nparray = np.arange(10000,dtype=np.float32).reshape(10,10,10,10)/1000
    compressed_tk, compressed_nparray, metadata = tensor_codec.compress(tensor_key,nparray)
    #The compress function should detect that the pipeline was lossless, and tag correctly
    assert('compressed' in compressed_tk[3])
    decompressed_tk, decompressed_nparray = tensor_codec.decompress(compressed_tk, compressed_nparray, metadata)
    #Both the decompressed tensor key and data should be identical to the original 
    assert(decompressed_tk == tensor_key)
    assert(np.array_equal(nparray,decompressed_nparray))

