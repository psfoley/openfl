import sys
sys.path.append('/tfl')

import tensorflow as tf

from models.mnist_cnn_keras.model import ConvModel



seed = 1234


model = ConvModel()

val_results = model.validate()
print("Initial val_results", val_results)

from models.export_init_weights import export_weights

tensor_dict = model.get_tensor_dict(True)
fpath = '/tfl/federations/weights/mnist_cnn_keras_init.pbuf'
export_weights(model.__class__.__name__, 0, tensor_dict, fpath)

ret = model.train_epoch()
print("training results", ret)
val_results = model.validate()
print("Epoch 1 val_results", val_results)

# We don't have the optimizer status before training.
tensor_dict = model.get_tensor_dict(with_opt_vars=True)
print("Exported the epoch 1 weights.")

tf.set_random_seed(seed)
model.train_epoch()
val_results = model.validate()
print("Epoch 2 val_results", val_results)

model.set_tensor_dict(tensor_dict, with_opt_vars=True)
# model.reset_opt_vars()
val_results = model.validate()
print("Restored to Epoch 1", val_results)


# Tensorflow doesn't support intializing the random state of a session until v2.0: https://github.com/tensorflow/community/pull/38
tf.set_random_seed(seed)
model.train_epoch()
val_results = model.validate()
print("Reran Epoch 2 val_results", val_results)