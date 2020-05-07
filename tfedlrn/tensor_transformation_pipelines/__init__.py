from .pipeline import Transformer, TransformationPipeline

from .no_op_pipeline import NoOpPipeline

from .random_shift_pipeline import RandomShiftPipeline

from tfedlrn import get_object

def get_compression_pipeline(module_name, class_name, **kwargs):
    return get_object(module_name=module_name, class_name=class_name, **kwargs)


