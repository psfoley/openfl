from .pipeline import Transformer, TransformationPipeline

from .no_compression_pipeline import NoCompressionPipeline, Float32NumpyArrayToBytes

from .random_shift_pipeline import RandomShiftPipeline

from .stc_pipeline import STCPipeline
from .skc_pipeline import SKCPipeline
from .kc_pipeline import KCPipeline

from tfedlrn import get_object

def get_compression_pipeline(module_name, class_name, **kwargs):
    return get_object(module_name=module_name, class_name=class_name, **kwargs)


