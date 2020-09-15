
from cli_helper import *

from fledge.protocols import dump_proto, construct_model_proto
from fledge.utilities import split_tensor_dict_for_holdouts
from fledge.federated import Plan

logger = getLogger(__name__)

@group()
@pass_context
def plan(context):
    '''Manage Federated Learning Plans'''
    context.obj['group'] = 'plan'

@plan.command()
@pass_context
@option('-p', '--plan_config', required = False, help = 'Federated learning plan [plan/plan.yaml]',               default = 'plan/plan.yaml', type = ClickPath(exists = True))
@option('-c', '--cols_config', required = False, help = 'Authorized collaborator list [plan/cols.yaml]',          default = 'plan/cols.yaml', type = ClickPath(exists = True))
@option('-d', '--data_config', required = False, help = 'The data set/shard configuration file [plan/data.yaml]', default = 'plan/data.yaml', type = ClickPath(exists = True))
@option('-a', '--aggregator_address', required = False, help = 'The FQDN of the federation agregator')
@option('-f', '--feature_shape',      required = False, help = 'The input shape to the model')
def initialize(context, plan_config, cols_config, data_config, aggregator_address, feature_shape):
    """
    Initialize Data Science plan

    Create a protocol buffer file of the initial model weights for the federation.
    """

    plan = Plan.Parse(plan_config_path = Path(plan_config),
                      cols_config_path = Path(cols_config),
                      data_config_path = Path(data_config))

    init_state_path = plan.config['aggregator' ]['settings']['init_state_path']

    if  feature_shape is None:
        if  cols_config is None:
            exit('You must specify either a feature shape or authorized collaborator list in order for the script to determine the input layer shape')

        collaborator_cname = plan.authorized_cols[0]

    else:

        logger.info(f'Using data object of type {type(data)} and feature shape {feature_shape}')
        raise NotImplementedError()

    data_loader = plan.get_data_loader(collaborator_cname)
    task_runner = plan.get_task_runner(collaborator_cname)
    tensor_pipe = plan.get_tensor_pipe()

    tensor_dict_split_fn_kwargs = task_runner.tensor_dict_split_fn_kwargs or {}
    tensor_dict, holdout_params = split_tensor_dict_for_holdouts(logger,
                                                                 task_runner.get_tensor_dict(False),
                                                                 **tensor_dict_split_fn_kwargs)

    logger.warn(f'Following paramters omitted from global initial model, '\
                f'local initialization will determine values: {list(holdout_params.keys())}')

    model_snap = construct_model_proto(tensor_dict  = tensor_dict,
                                       round_number = 0,
                                       tensor_pipe  = tensor_pipe)

    logger.info(f'Creating Initial Weights File    🠆 {init_state_path}')

    dump_proto(model_proto = model_snap, fpath = init_state_path)

    plan_origin = Plan.Parse(Path(plan_config), resolve = False).config

    if  plan_origin['network']['settings']['agg_addr'] == 'auto' or aggregator_address:
        plan_origin['network']['settings']              = plan_origin['network'].get('settings', {})
        plan_origin['network']['settings']['agg_addr']  = aggregator_address or getfqdn()

        logger.warn(f"Patching Aggregator Addr in Plan 🠆 {plan_origin['network']['settings']['agg_addr']}")

        Plan.Dump(Path(plan_config), plan_origin)
