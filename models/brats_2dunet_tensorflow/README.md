# 2D-UNet for BraTS17 

The model-training code contatins pre-processing, model architecture, loss function and the optimizer. The FL framework calls the training module by the `get_model()` function in the `__init__.py` file.

> TODO: The `get_model()` function should accept many parameters to configure the model accordingly. 

## Example
```python
code_name = "brats_2dunet_tensorflow"
module = importlib.import_module(code_name)
model = module.get_model()
```


## Local Test
```shell
python test.py --dataset_path /opt/mydataset
```

