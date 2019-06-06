import abc


class FLModel(metaclass=abc.ABCMeta):

	@abc.abstractmethod
	def get_tensor_dict(self):
		"""Returns all parameters for aggregation, including optimizer parameters, if appropriate"""
		pass

	@abc.abstractmethod
	def set_tensor_dict(self, tensor_dict):
		"""Returns all parameters for aggregation, including optimizer parameters, if appropriate"""
		pass

	@abc.abstractmethod
	def train_epoch(self):
		pass

	@abc.abstractmethod
	def get_training_data_size(self):
		pass

	@abc.abstractmethod
	def validate(self):
		pass

	@abc.abstractmethod
	def get_validation_data_size(self):
		pass
