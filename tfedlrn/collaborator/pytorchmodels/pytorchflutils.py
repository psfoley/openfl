import torch
import numpy as np
from copy import deepcopy


def _derive_opt_state_dict(opt_state_dict):
    # Flattens the optimizer state dict so as to have key, value pairs with values as numpy arrays.
    # The keys have sufficient info to restore opt_state_dict using expand_derived_opt_state_dict.

    derived_opt_state_dict = {}

    # Determine if state is needed for this optimizer.
    if len(opt_state_dict['state']) == 0:
        derived_opt_state_dict['__opt_state_needed'] = 'false'
        return derived_opt_state_dict

    derived_opt_state_dict['__opt_state_needed'] = 'true'

    # We assume that the state collected for all parameter groups is the same.
    # Using one example state key, we collect keys for the corresponding dictionary value.
    example_state_key = opt_state_dict['param_groups'][0]['params'][0]
    state_subkeys = list(opt_state_dict['state'][example_state_key].keys())

    # Tags will record whether the value associated to the subkey is a tensor or not.
    state_subkey_tags = []
    for state_subkey in state_subkeys:
        if isinstance(opt_state_dict['state'][example_state_key][state_subkey], torch.Tensor):
            state_subkey_tags.append('istensor')
        else:
            state_subkey_tags.append('')
    state_subkeys_and_tags = list(zip(state_subkeys, state_subkey_tags))
    
    # Forming the flattened dict, using a concatenation of group index, subindex, tag,
    # and subkey inserted into the flattened dict key - needed for reconstruction.
    nb_params_per_group = []
    for group_idx, group in enumerate(opt_state_dict['param_groups']):
        for idx, param_id in enumerate(group['params']):
            for subkey, tag in state_subkeys_and_tags:
                if tag == 'istensor':
                    new_v = opt_state_dict['state'][param_id][subkey].cpu().numpy()
                else:
                    new_v = np.array([opt_state_dict['state'][param_id][subkey]])
                derived_opt_state_dict['__opt_state_{}_{}_{}_{}'.format(group_idx, idx, tag, subkey)] = new_v
        nb_params_per_group.append(idx + 1)
    # group lengths are also helpful for reconstructing original opt_state_dict structure
    derived_opt_state_dict['__opt_group_lengths'] = np.array(nb_params_per_group)

    return derived_opt_state_dict


def expand_derived_opt_state_dict(derived_opt_state_dict, device):
    # Performs the inverse operations of _encode_and_flatten.

    # Reconstructing state_subkeys_and_tags using the example key 
    # prefix, "__opt_state_0_0_", certain to be present.
    state_subkeys_and_tags = []
    for key in derived_opt_state_dict:
        if key.startswith('__opt_state_0_0_'):
            stripped_key = key[16:]
            if stripped_key.startswith('istensor_'):
                this_tag = 'istensor'
                subkey = stripped_key[9:]
            else:
                this_tag = ''
                subkey = stripped_key[1:]
            state_subkeys_and_tags.append((subkey, this_tag))

    opt_state_dict = {'param_groups': [], 'state': {}}
    nb_params_per_group = list(derived_opt_state_dict.pop('__opt_group_lengths').astype(np.int))

    # Construct the expanded dict.
    for group_idx, nb_params in enumerate(nb_params_per_group):
        these_group_ids = ['{}_{}'.format(group_idx, idx) for idx in range(nb_params)]
        opt_state_dict['param_groups'].append({'params': these_group_ids})
        for this_id in these_group_ids:
            opt_state_dict['state'][this_id] = {}
            for subkey, tag in state_subkeys_and_tags:
                flat_key = '__opt_state_{}_{}_{}'.format(this_id, tag, subkey)
                if tag == 'istensor':
                    new_v = torch.from_numpy(derived_opt_state_dict.pop(flat_key))
                else:
                    # Here (for currrently supported optimizers) the subkey should be 'step' 
                    # and the length of array should be one.
                    assert subkey == 'step'
                    assert len(derived_opt_state_dict[flat_key]) == 1
                    new_v = int(derived_opt_state_dict.pop(flat_key))
                opt_state_dict['state'][this_id][subkey] = new_v 


    # sanity check that we did not miss any optimizer state
    assert len(derived_opt_state_dict) == 0

    return opt_state_dict


def _get_optimizer_state(optimizer):

    opt_state_dict = deepcopy(optimizer.state_dict())
    derived_opt_state_dict = _derive_opt_state_dict(opt_state_dict)

    return derived_opt_state_dict


def _set_optimizer_state(optimizer, device, derived_opt_state_dict):

    temp_state_dict = expand_derived_opt_state_dict(derived_opt_state_dict, device)

    # FIXME: Figure out whether or not this breaks learning rate scheduling and the like.
    # Setting default values.
    # All optimizer.defaults are considered as not changing over course of training.
    for group in temp_state_dict['param_groups']:
        for k, v in optimizer.defaults.items():
            group[k] = v

    optimizer.load_state_dict(temp_state_dict)



def pt_get_tensor_dict(torch_nn, torch_optimizer):
    # FIXME: self.parameters() instead? Unclear if load_state_dict() or simple assignment is better
    # for now, state dict gives us names which is good
    # FIXME: do both and sanity check each time?

    # deep copy so as to decouple from active model
    state = deepcopy(torch_nn.state_dict())

    for k, v in state.items():
        # When restoring, we currently assume all values are tensors.
        if not torch.is_tensor(v):
            raise NotImplementedError('We do not currently support non-tensors '
                                      'coming from model.state_dict()')
        state[k] = v.cpu().numpy()  # get as a numpy array, making sure is on cpu

    return {**state, **_get_optimizer_state(torch_optimizer)}


def pt_set_tensor_dict(torch_nn, tensor_dict):
    # FIXME: self.parameters() instead? Unclear if load_state_dict() or simple assignment is better
    # for now, state dict gives us names, which is good
    # FIXME: do both and sanity check each time?

    # get device for correct placement of tensors
    device = torch_nn.device

    new_state = {}
    # Grabbing keys from model's state_dict helps to confirm we have everything
    for k in torch_nn.state_dict():
        new_state[k] = torch.from_numpy(tensor_dict.pop(k)).to(device)

    # set model state
    torch_nn.load_state_dict(new_state)

    # next we have the optimizer state, if there is state to restore
    if tensor_dict.pop('__opt_state_needed') == 'true':
        _set_optimizer_state(torch_nn.get_optimizer(), device, tensor_dict)

    # sanity check that we did not record any state that was not used
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
