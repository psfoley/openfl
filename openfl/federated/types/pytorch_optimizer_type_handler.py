from openfl.utilities import TensorKey, split_tensor_dict_for_holdouts
from logging import getLogger
import hashlib
import numpy as np
from openfl.federated.types import TypeHandler
from copy import deepcopy

logger = getLogger(__name__)

class PyTorchOptimizerTypeHandler(TypeHandler):

    def __init__(self):
        self.send_delta = True
        self.compress = True
        self.aggregation_type = 'weighted_mean'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_dependencies():
        """What are the dependencies for this type?"""
        return ['torch']

    @staticmethod
    def type():
        """The type that this class handles"""
        from torch.optim import Optimizer
        return Optimizer

    @staticmethod
    def _derive_opt_state_dict(opt_state_dict):
        """Separate optimizer tensors from the tensor dictionary.
    
        Flattens the optimizer state dict so as to have key, value pairs with
        values as numpy arrays.
        The keys have sufficient info to restore opt_state_dict using
        expand_derived_opt_state_dict.
    
        Args:
            opt_state_dict: The optimizer state dictionary
    
        """
        import torch
        derived_opt_state_dict = {}
    
        # Determine if state is needed for this optimizer.
        if len(opt_state_dict['state']) == 0:
            derived_opt_state_dict['__opt_state_needed'] = 'false'
            return derived_opt_state_dict
    
        derived_opt_state_dict['__opt_state_needed'] = 'true'
    
        # Using one example state key, we collect keys for the corresponding
        # dictionary value.
        example_state_key = opt_state_dict['param_groups'][0]['params'][0]
        example_state_subkeys = set(
            opt_state_dict['state'][example_state_key].keys()
        )
    
        # We assume that the state collected for all params in all param groups is
        # the same.
        # We also assume that whether or not the associated values to these state
        # subkeys is a tensor depends only on the subkey.
        # Using assert statements to break the routine if these assumptions are
        # incorrect.
        for state_key in opt_state_dict['state'].keys():
            assert example_state_subkeys == set(opt_state_dict['state'][state_key].keys())
            for state_subkey in example_state_subkeys:
                assert (isinstance(
                    opt_state_dict['state'][example_state_key][state_subkey],
                    torch.Tensor)
                    == isinstance(
                        opt_state_dict['state'][state_key][state_subkey],
                        torch.Tensor))
    
        state_subkeys = list(opt_state_dict['state'][example_state_key].keys())
    
        # Tags will record whether the value associated to the subkey is a
        # tensor or not.
        state_subkey_tags = []
        for state_subkey in state_subkeys:
            if isinstance(
                    opt_state_dict['state'][example_state_key][state_subkey],
                    torch.Tensor
            ):
                state_subkey_tags.append('istensor')
            else:
                state_subkey_tags.append('')
        state_subkeys_and_tags = list(zip(state_subkeys, state_subkey_tags))
    
        # Forming the flattened dict, using a concatenation of group index,
        # subindex, tag, and subkey inserted into the flattened dict key -
        # needed for reconstruction.
        nb_params_per_group = []
        for group_idx, group in enumerate(opt_state_dict['param_groups']):
            for idx, param_id in enumerate(group['params']):
                for subkey, tag in state_subkeys_and_tags:
                    if tag == 'istensor':
                        new_v = opt_state_dict['state'][param_id][
                            subkey].cpu().numpy()
                    else:
                        new_v = np.array(
                            [opt_state_dict['state'][param_id][subkey]]
                        )
                    derived_opt_state_dict[
                        '__opt_state_{}_{}_{}_{}'.format(
                            group_idx, idx, tag, subkey)
                    ] = new_v
            nb_params_per_group.append(idx + 1)
        # group lengths are also helpful for reconstructing
        # original opt_state_dict structure
        derived_opt_state_dict['__opt_group_lengths'] = np.array(
            nb_params_per_group
        )
    
        return derived_opt_state_dict

    def expand_derived_opt_state_dict(derived_opt_state_dict, device):
        """Expand the optimizer state dictionary.
    
        Takes a derived opt_state_dict and creates an opt_state_dict suitable as
        input for load_state_dict for restoring optimizer state.
    
        Reconstructing state_subkeys_and_tags using the example key
        prefix, "__opt_state_0_0_", certain to be present.
    
        Args:
            derived_opt_state_dict: Optimizer state dictionary
    
        Returns:
            dict: Optimizer state dictionary
        """
        import torch
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
        nb_params_per_group = list(
            derived_opt_state_dict.pop('__opt_group_lengths').astype(np.int)
        )
    
        # Construct the expanded dict.
        for group_idx, nb_params in enumerate(nb_params_per_group):
            these_group_ids = [
                '{}_{}'.format(group_idx, idx) for idx in range(nb_params)
            ]
            opt_state_dict['param_groups'].append({'params': these_group_ids})
            for this_id in these_group_ids:
                opt_state_dict['state'][this_id] = {}
                for subkey, tag in state_subkeys_and_tags:
                    flat_key = '__opt_state_{}_{}_{}'.format(this_id, tag, subkey)
                    if tag == 'istensor':
                        new_v = torch.from_numpy(derived_opt_state_dict.pop(flat_key))
                    else:
                        # Here (for currrently supported optimizers) the subkey
                        # should be 'step' and the length of array should be one.
                        assert subkey == 'step'
                        assert len(derived_opt_state_dict[flat_key]) == 1
                        new_v = int(derived_opt_state_dict.pop(flat_key))
                    opt_state_dict['state'][this_id][subkey] = new_v
    
        # sanity check that we did not miss any optimizer state
        assert len(derived_opt_state_dict) == 0
    
        return opt_state_dict


    @staticmethod
    def attr_to_map(attribute,attribute_name,round_phase='end',round_num=0, report=False, origin='LOCAL'):
        """Transform the attribute to a {TensorKey: nparray} map for transport."""
        import torch
        # deep copy so as to decouple from active optimizer

        opt_state_dict = deepcopy(attribute.state_dict())

        # Optimizer state might not have some parts representing frozen parameters
        # So we do not synchronize them
        param_keys_with_state = set(opt_state_dict['state'].keys())
        for group in opt_state_dict['param_groups']:
            local_param_set = set(group['params'])
            params_to_sync = local_param_set & param_keys_with_state
            group['params'] = sorted(list(params_to_sync))
    
        derived_opt_state_dict = PyTorchOptimizerTypeHandler._derive_opt_state_dict(opt_state_dict)

        if derived_opt_state_dict['__opt_state_needed'] == 'false':
            return {}

        derived_opt_state_dict, _ = split_tensor_dict_for_holdouts(
                logger, derived_opt_state_dict, {})

        tensorkey_nparray_dict = {
                TensorKey(tensor_name, origin, round_num, round_phase, report, (f'{attribute_name}',)):
                nparray for tensor_name, nparray in derived_opt_state_dict.items()
                if tensor_name is not '__opt_state_needed' 
        }

        return tensorkey_nparray_dict


    def map_to_attr(self,attribute,tensorkey_nparray_map):
        """Transform tensorkey map to attribute"""
        import torch

        temp_state_dict = PyTorchOptimizerTypeHandler.expand_derived_opt_state_dict(
            tensorkey_nparray_map, self.device)
    
        # FIXME: Figure out whether or not this breaks learning rate
        #  scheduling and the like.
        # Setting default values.
        # All optimizer.defaults are considered as not changing over course of
        # training.
        for group in temp_state_dict['param_groups']:
            for k, v in optimizer.defaults.items():
                group[k] = v
    
        attribute.load_state_dict(temp_state_dict)
        return attribute


    @staticmethod
    def get_tensorkeys(attribute,attribute_name,round_phase='start',round_num=0, report=False, origin='LOCAL'):
        """Get tensorkeys will always be run at the end of the round"""
        tensorkey_map = PyTorchOptimizerTypeHandler.attr_to_map(attribute, attribute_name,
                round_phase=round_phase, round_num=round_num, report=report, origin=origin)
        return tensorkey_map.keys()

    @staticmethod
    def get_hash(attribute):
        import torch
        hasher = hashlib.sha384()
        state_dict = attribute.state_dict()
        state = deepcopy(state_dict)
    
        for k, v in state.items():
            # When restoring, we currently assume all values are tensors.
            if not torch.is_tensor(v):
                raise ValueError('We do not currently support non-tensors '
                                 'coming from model.state_dict()')
            # get as a numpy array, making sure is on cpu
            state[k] = v.cpu().numpy()

        for layer in state.values():
            hasher.update(layer.data.tobytes())
        return hasher.hexdigest()

