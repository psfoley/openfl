from .tensorflow2dunet import TensorFlow2DUNet


def get_model(**kwargs):
    # FIXME: fix as part of issue #42.
    # FIXME: Put BraTS loading function into its own shared spot outside TF-specific directory
    X_train, y_train, X_val, y_val = load_BraTS17_insitution(path="/opt/datasets/BraTS17", channels_first=False)

    model = TensorFlow2DUNet(X_train, y_train, X_val, y_val)
    return model
