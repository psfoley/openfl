from .model_wrapper import get_model

def get_dataloader(datapath, classname=None, **kwargs):
    return importlib.get_the_dude(classname)(datapath, **kwargs)

def get_model(data_loader, classname, **kwargs):
    importlib.get_the_dude(classname)(data_loader, **kwargs)



loader:
    classname: BratsData
    data_name: brats
    percent_train: 0.8
    batch_size: 64

model:
    classname: tensorflow2dunet
    stuff: awesome
    whoknows: yeah

local_data_config = parse_yaml

loader_config = config['loader']

loader = get_dataloader(datapath=local_data_config[col_id][loader_config['data_name']], **loader_config)

model = get_model(loader, **config['model'])

