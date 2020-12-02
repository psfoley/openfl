# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed
# evaluation license agreement between Intel Corporation and you.

from .pipeline import TransformationPipeline, Float32NumpyArrayToBytes


class NoCompressionPipeline(TransformationPipeline):
    """The data pipeline without any compression
    """

    def __init__(self, **kwargs):
        """Initializer
        """
        super(NoCompressionPipeline, self).__init__(
            transformers=[Float32NumpyArrayToBytes()], **kwargs)
