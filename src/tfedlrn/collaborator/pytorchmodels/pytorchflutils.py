import torch
import numpy as np


def _get_optimizer_tensors(optimizer):
    tensor_dict = {}

    # NOTE: this gave inconsistent orderings across collaborators, so does not work
    # state = optimizer.state_dict()['state']

    # # FIXME: this is really fragile. Need to understand what could change here
    # for i, sk in enumerate(state.keys()):
    #     if isinstance(state[sk], dict):
    #         for k, v in state[sk].items():
    #             if isinstance(v, torch.Tensor):
    #                 tensor_dict['{}_{}'.format(i, k)] = v.cpu().numpy()

    # FIXME: this is entirely broken. Appears to cause the optimizer to update the wrong tensors
    # # FIXME: not clear that this works consistently across optimizers
    # # FIXME: hard-coded naming convention sucks and could absolutely break
    # i = 0
    # for group in optimizer.param_groups:
    #     for p in group['params']:
    #         tensor_dict['__opt_{}'.format(i)] = p.detach().cpu().numpy()
    #         i += 1

    # return tensor_dict
    return {}

def _set_optimizer_tensors(optimizer, tensor_dict):

    # NOTE: the state dict ordering wasn't consistent. We'd like to use load_state_dict rather than
    # directly setting the tensors, if possible, but it's not clear that we can
    # state = optimizer.state_dict()

    # # FIXME: this is really fragile. Need to understand what could change here
    # for i, sk in enumerate(state['state'].keys()):
    #     if isinstance(state['state'][sk], dict):
    #         for k, v in state['state'][sk].items():
    #             if isinstance(v, torch.Tensor):
    #                 key = '{}_{}'.format(i, k)
                    
    #                 if key not in tensor_dict:
    #                     raise ValueError('{} not in keys: {}'.format(key, list(tensor_dict.keys())))
                    
    #                 state['state'][sk][k] = torch.Tensor(tensor_dict[key]).to(v.device)
    # optimizer.load_state_dict(state)

    # FIXME: this is entirely broken. Appears to cause the optimizer to update the wrong tensors        
    # # FIXME: not clear that this works consistently across optimizers
    # # FIXME: hard-coded naming convention sucks and could absolutely break
    # i = 0
    # for group in optimizer.param_groups:
    #     for idx, p in enumerate(group['params']):
    #         # group['params'][idx] = torch.Tensor(tensor_dict['__opt_{}'.format(i)]).to(p.device)
    #         i += 1
    pass


def pt_get_tensor_dict(torch_nn, torch_optimizer):
    # FIXME: should we use self.parameters()??? Unclear if load_state_dict() is better or simple assignment is better
    # for now, state dict gives us names, which is good

    # FIXME: do both and sanity check each time?

    # FIXME: can this have values other than the tensors????
    state = torch_nn.state_dict()
    for k, v in state.items():
        state[k] = v.cpu().numpy() # get as a numpy array

    return {**state, **_get_optimizer_tensors(torch_optimizer)}


def pt_set_tensor_dict(torch_nn, tensor_dict):
    # FIXME: should we use self.parameters()??? Unclear if load_state_dict() is better or simple assignment is better
    # for now, state dict gives us names, which is good
    
    # FIXME: do both and sanity check each time?

    # get the model state so that we can determine the correct tensor values/device placements
    model_state = torch_nn.state_dict()

    new_state = {}
    for k, v in model_state.items():
        new_state[k] = torch.Tensor(tensor_dict[k]).to(v.device)

    # set model state
    torch_nn.load_state_dict(new_state)

    # next we have the optimizer state
    _set_optimizer_tensors(torch_nn.get_optimizer(), tensor_dict)

def pt_validate(torch_nn, val_loader, device, metric):
    torch_nn.eval()
    val_score = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            samples = target.shape[0]
            total_samples += samples
            data, target = data.to(device), target.to(device)
            output = torch_nn(data)
            val_score += metric(output, target).cpu().numpy() * samples
    return val_score / total_samples


def pt_train_epoch(torch_nn, train_loader, device, optimizer, loss_fn):
    # set to "training" mode
    torch_nn.train()
    
    losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = torch_nn(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        
    return np.mean(losses)


def pt_create_loader(X, y, **kwargs):
    tX = torch.stack([torch.Tensor(i) for i in X])
    ty = torch.stack([torch.Tensor(i) for i in y])
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tX, ty), **kwargs)
