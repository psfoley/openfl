# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from tfedlrn.tensor_transformation_pipelines import TensorCodec,NoCompressionPipeline,RandomShiftPipeline,\
                                                              STCPipeline,SKCPipeline,KCPipeline
from tfedlrn import TensorKey
import numpy as np

def test_require_lossless():
    #STC is a lossy pipeline
    tensor_codec = TensorCodec(STCPipeline())
    tensor_key = TensorKey('conv1','col_1',0,False,('model',))
    nparray = np.arange(10000).reshape(100,100)
    compressed_tk, compressed_nparray, metadata = tensor_codec.compress(tensor_key,nparray,require_lossless=True)
    comp_tk_name,comp_tk_origin,comp_tk_round,comp_tk_report,comp_tk_tags = compressed_tk
    #A tag of 'compressed' means the tensor has been losslessly compressed
    assert('compressed' in comp_tk_tags)
    decompressed_tk, decompressed_nparray = tensor_codec.decompress(compressed_tk, compressed_nparray, metadata)
    #Both the decompressed tensor key and data should be identical to the original 
    assert(decompressed_tk == tensor_key)
    assert(np.array_equal(nparray,decompressed_nparray))

def test_lossy_tag():
    #STC is a lossy pipeline
    tensor_codec = TensorCodec(STCPipeline())
    tensor_key = TensorKey('conv1','col_1',5,False,('trained',))
    nparray = np.arange(10000).reshape(100,100)
    compressed_tk, compressed_nparray, metadata = tensor_codec.compress(tensor_key,nparray)
    comp_tk_name,comp_tk_origin,comp_tk_round,comp_tk_report,comp_tk_tags = compressed_tk
    #A tag of 'lossy_compressed' means the tensor has been transformed through a lossy pipeline
    assert('lossy_compressed' in comp_tk_tags)

def test_lossless_pipeline_detection():
    #RandomShift is a lossless pipeline
    tensor_codec = TensorCodec(NoCompressionPipeline())
    tensor_key = TensorKey('conv1','col_1',0,False,('model','some_tag'))
    nparray = np.arange(10000,dtype=np.float32).reshape(10,10,10,10)/1000
    compressed_tk, compressed_nparray, metadata = tensor_codec.compress(tensor_key,nparray)
    comp_tk_name,comp_tk_origin,comp_tk_round,comp_tk_report,comp_tk_tags = compressed_tk
    #The compress function should detect that the pipeline was lossless, and tag correctly
    assert('compressed' in comp_tk_tags)
    decompressed_tk, decompressed_nparray = tensor_codec.decompress(compressed_tk, compressed_nparray, metadata)
    #Both the decompressed tensor key and data should be identical to the original 
    assert(decompressed_tk == tensor_key)
    assert(np.array_equal(nparray,decompressed_nparray))

