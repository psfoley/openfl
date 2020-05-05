from .pipeline import Transformer, TensorDictToModelProtoPipeline

from .no_compression_pipeline import NoCompressionPipeline

from .random_shift_pipeline import RandomShiftPipeline

from tfedlrn import get_object

def get_custom_update_pipeline(module_name, class_name, **kwargs):
    return get_object(module_name=module_name, class_name=class_name, **kwargs)


