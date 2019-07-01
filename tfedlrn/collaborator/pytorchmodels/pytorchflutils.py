import torch
import numpy as np
from copy import deepcopy


def _derive_opt_state_dict(opt_state_dict):
    # Flattens the optimizer state dict so as to have key, value pairs with values as numpy arrays.
    # The keys have sufficient information to know if the array becomes an int (meaning it was a 'step' value),
    # or if it should be converted to torch.Tensor, or if the key is: '__opt_group_lengths' in which case the
    # value is a 1-d array with components telling us the lengths of each parameter group.
    # FIXME: Should I look into when to delete the opt_state_dict?

    derived_opt_state_dict = {}

    nb_params_per_group = []
    for group_idx, group in enumerate(opt_state_dict['param_groups']):
        for idx, param_id in enumerate(group['params']):
            for k, v in opt_state_dict['state'][param_id]:
                if isinstance(v, torch.Tensor):
                    derived_opt_state_dict['__opt_state_{}_{}_istensor_{}'.format(group_idx, idx, k)] = v.cpu().numpy()
                else:
                    derived_opt_state_dict['__opt_state_{}_{}_{}'.format(group_idx, idx, k)] = np.array([v])
        nb_params_per_group.append(idx + 1)
    derived_opt_state_dict['__opt_group_shape'] = np.array(nb_params_per_group)

    return derived_opt_state_dict


def expand_derived_opt_state_dict(derived_opt_state_dict, device):
    # Performs the inverse operations of _encode_and_flatten.
    # FIXME: I am spending a lot of time searching for the right keys here, should I add more
    #        structure? (will pay in message length?)

    opt_state_dict = {'param_groups': [], 'state': {}}
    nb_params_per_group = list(opt_state_dict.pop('__opt_group_lengths').astype(np.int))

    for group_idx, nb_params in enumerate(nb_params_per_group):
        these_group_ids = ['{}_{}'.format(group_idx, idx) for idx in range(nb_params)]
        opt_state_dict['param_groups'].append({'params': these_group_ids})
        for id in these_group_ids:
            opt_state_dict['state'][id] = {}

            # Here is the search referred to in the FIXME above
            for k in derived_opt_state_dict:
                target_string = '__opt_state_{}_'.format(id)

                if k.startswith(target_string):
                    # strip off the decoration on the key
                    nb_to_strip = len(target_string)
                    stripped_key = k[nb_to_strip:]
                    # determine if the array should become a tensor
                    is_tensor = stripped_key.startswith('istensor_')

                    if is_tensor:
                        opt_state_dict['state'][id][stripped_key[9:]] = \
                            torch.Tensor(derived_opt_state_dict.pop(k)).to(device)
                    else:
                        # here the stripped key should be 'step' and the length of array should be one.
                        assert stripped_key == 'step'
                        assert len(derived_opt_state_dict['k']) == 1
                        opt_state_dict['state'][id][stripped_key] = int(derived_opt_state_dict.pop(k))

    # sanity check that we did not miss any optimizer state
    assert len(derived_opt_state_dict) == 0

    return opt_state_dict


def _get_optimizer_state(optimizer):

    opt_state_dict = deepcopy(optimizer.state_dict())
    derived_opt_state_dict = _derive_opt_state_dict(opt_state_dict)

    return derived_opt_state_dict


def _set_optimizer_state(optimizer, device, derived_opt_state_dict):

    temp_state_dict = expand_derived_opt_state_dict(derived_opt_state_dict, device)
    optimizer.load_state_dict(temp_state_dict)


def pt_get_tensor_dict(torch_nn, torch_optimizer):
    # FIXME: should we use self.parameters()??? Unclear if load_state_dict() is better or simple assignment is better
    # for now, state dict gives us names, which is good

    # FIXME: do both and sanity check each time?

    # FIXME: can this have values other than the tensors????
    state = torch_nn.state_dict()
    for k, v in state.items():
        state[k] = v.cpu().numpy()  # get as a numpy array

    return {**state, **_get_optimizer_state(torch_optimizer)}


def pt_set_tensor_dict(torch_nn, tensor_dict):
    # FIXME: should we use self.parameters()??? Unclear if load_state_dict() is better or simple assignment is better
    # for now, state dict gives us names, which is good
    
    # FIXME: do both and sanity check each time?

    # get the model state and device so that we can determine the correct tensor values/device placements
    model_state = torch_nn.state_dict()
    device = torch_nn.device

    new_state = {}
    for k, v in model_state.items():
        new_state[k] = torch.Tensor(tensor_dict.pop(k)).to(device)

    # set model state
    torch_nn.load_state_dict(new_state)

    # next we have the optimizer state
    _set_optimizer_state(torch_nn.get_optimizer(), device, tensor_dict)

    # sanity check that we did not miss anything
    assert len(tensor_dict) == 0


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
