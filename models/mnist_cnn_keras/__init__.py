from .model import ConvModel

def get_model(dataset_path, **kwargs):
    if dataset_path is not None:
        input_shape, num_classes, x_train, y_train, x_val, y_val = ConvModel.load_dataset(dataset_path)
    else:
        input_shape = (28, 28, 1)
        num_classes = 10
        x_train = None
        y_train = None
        x_val = None
        y_val = None

    return ConvModel(input_shape, num_classes, x_train, y_train, x_val, y_val, **kwargs)
