import torch
import torch.nn as nn


class PyTorchFLModel(FLModel, nn.Module):
    """WIP code. Goal is to simplify porting a model to this framework.
    Currently, this creates a placeholder and assign op for every variable, which grows the graph considerably.
    Also, the abstraction for the tf.session isn't ideal yet."""

    def __init__(self):
        # calls nn.Module init
        super(PyTorchFLModel, self).__init__()

    @abc.abstractmethod
    def get_optimizer(self):
        pass

    def get_optimizer_tensors(self):
        optimizer = self.get_optimizer()

        tensor_dict = {}

        state = optimizer.state_dict()['state']

        # FIXME: this is really fragile. Need to understand what could change here
        for i, sk in enumerate(state.keys()):
            if isinstance(state[sk], dict):
                for k, v in state[sk].items():
                    if isinstance(v, torch.Tensor):
                        tensor_dict['{}_{}'.format(i, k)] = v.cpu().numpy()

        return tensor_dict
                    
    def set_optimizer_tensors(self, tensor_dict):
        optimizer = self.get_optimizer()

        state = optimizer.state_dict()

        # FIXME: this is really fragile. Need to understand what could change here
        for i, sk in enumerate(state['state'].keys()):
            if isinstance(state['state'][sk], dict):
                for k, v in state['state'][sk].items():
                    if isinstance(v, torch.Tensor):
                        key = '{}_{}'.format(i, k)
                        
                        if key not in tensor_dict:
                            raise ValueError('{} not in keys: {}'.format(key, list(tensor_dict.keys())))
                        
                        state['state'][sk][k] = torch.Tensor(tensor_dict[key]).to(v.device)
        optimizer.load_state_dict(state)

    def get_tensor_dict(self):
        # FIXME: should we use self.parameters()??? Unclear if load_state_dict() is better or simple assignment is better
        # for now, state dict gives us names, which is good

        # FIXME: do both and sanity check each time?

        # FIXME: can this have values other than the tensors????
        state = self.state_dict()
        for k, v in state.items():
            state[k] = v.cpu().numpy() # get as a numpy array
        return {**state, **self.get_optimizer_tensors()}

    def set_tensor_dict(self, tensor_dict):
        # FIXME: should we use self.parameters()??? Unclear if load_state_dict() is better or simple assignment is better
        # for now, state dict gives us names, which is good
        
        # FIXME: do both and sanity check each time?

        # get the model state so that we can determine the correct tensor values/device placements
        model_state = self.state_dict()

        new_state = {}
        for k, v in model_state.items():
            new_state[k] = torch.Tensor(tensor_dict[k]).to(v.device)

        # set model state
        self.load_state_dict(new_state)

        # next we have the optimizer state
        self.set_optimizer_tensors(tensor_dict)
