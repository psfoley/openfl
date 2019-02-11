import os

from AGG_alpha import pickle_file, unpickle_file


def get_layer_path(work_id, layer_id, subdir, bucket=0):
    return os.path.join(work_id, subdir, f'{layer_id}_{bucket}.pkl')


class Layer(object):

    def __init__(self,
                 work_id,
                 layer_id,
                 parameters,
                 weight,
                 version,
                 bucket=0):
        self.work_id = work_id
        self.layer_id = layer_id
        self.parameters = parameters
        self.weight = weight
        self.version = version
        self.bucket = bucket

    @staticmethod
    def from_dict(d):
        return Layer(d['work_id'],
            d['layer_id'],
            d['parameters'],
            d['weight'],
            d['version'])

    def _get_path(self, subdir):
        return get_layer_path(self.work_id, self.layer_id, subdir, self.bucket)

    def _save(self, subdir):
        pickle_file(self, Layer._get_path(self, subdir))

    def save_training(self):
        self._save('training')

    def save_complete(self):
        self.bucket = 0
        self._save('complete')

    @staticmethod
    def _load(work_id, layer_id, subdir, bucket=0):
        return unpickle_file(get_layer_path(work_id, layer_id, subdir, bucket))

    @staticmethod
    def load_training(work_id, layer_id, bucket=0):
        return Layer._load(work_id, layer_id, 'training', bucket)

    @staticmethod
    def load_complete(work_id, layer_id):
        return Layer._load(work_id, layer_id, 'complete', 0)
