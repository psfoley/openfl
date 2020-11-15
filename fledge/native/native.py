#This file defines fledge entrypoints to be used directly through python (not CLI)
import os
from copy import copy
from logging import getLogger
import fledge.interface.workspace as workspace
import fledge.interface.aggregator as aggregator
import fledge.interface.collaborator as collaborator
import fledge.interface.plan as plan
from fledge.interface.cli_helper import *

from fledge.component import Aggregator
from fledge.transport import AggregatorGRPCServer
from fledge.component import Collaborator
from fledge.transport import CollaboratorGRPCClient
from fledge.federated import Plan

from fledge.protocols import dump_proto, construct_model_proto
from fledge.utilities import split_tensor_dict_for_holdouts

logger = getLogger(__name__)

WORKSPACE_PREFIX = os.path.join(os.path.expanduser('~'), '.local', 'workspace')

def setup_logging():
    #Setup logging
    from logging import basicConfig
    from rich.console   import Console
    from rich.logging   import RichHandler
    import pkgutil
    if (True if pkgutil.find_loader('tensorflow') else False):
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
    console = Console(width = 160)
    basicConfig(level = 'INFO', format = '%(message)s', datefmt = '[%X]', handlers = [RichHandler(console = console)])    

def init(workspace_template='default', agg_fqdn=None, col_names=['one', 'two']):
    """
    The initialization function for the fledge package. It performs the following tasks:
         
         1. Creates a workspace in ~/.local/workspace (Equivalent to `fx workspace create --prefix ~/.local/workspace --template $workspace_template)
         2. Setup certificate authority (equivalent to `fx workspace certify`)
         3. Setup aggregator PKI (equivalent to `fx aggregator generate-cert-request` followed by `fx aggregator certify`)
         4. Setup list of collaborators (col_names) and their PKI. (Equivalent to running `fx collaborator generate-cert-request` followed by `fx collaborator certify` for each of the collaborators in col_names)
         5. Setup logging

    Args:
        workspace_template : str (default='default')
            The template that should be used as the basis for the experiment. Other options include are any of the template names [keras_cnn_mnist,tf_2dunet,tf_cnn_histology,torch_cnn_histology,torch_cnn_mnist]
        agg_fqdn : str
           The local node's fully qualified domain name (if it can't be resolved automatically)
        col_names: list[str]
           The names of the collaborators that will be created. These collaborators will be set up to participate in the experiment, but are not required to
    
    Returns:
        None
    """

    workspace.create(WORKSPACE_PREFIX, workspace_template)
    os.chdir(WORKSPACE_PREFIX)
    workspace.certify()
    aggregator.generate_cert_request(agg_fqdn)
    aggregator.certify(agg_fqdn, silent=True)
    data_path = 1
    for col_name in col_names:
        collaborator.generate_cert_request(col_name, str(data_path), silent=True, skip_package=True)
        collaborator.certify(col_name, silent=True)
        data_path += 1

    setup_logging()


def create_collaborator(plan, name, model, aggregator):

    #Using the same plan object to create multiple collaborators leads to identical collaborator objects.
    #This function can be removed once collaborator generation is fixed in fledge/federated/plan/plan.py
    plan = copy(plan)

    return plan.get_collaborator(name,task_runner=model,client=aggregator)

def setup_pki(aggregator_fqdn, collaborator_names):
    """
    Params
    ------
        aggregator_fqdn: str   - aggregator fqdn (this can be resolved with socket.get_fqdn()
        collaborator_names: 
    """
    pass


def patch_plan(config,plan):
    """

    Strawman function (not used yet):

    The config is a dictionary of parameters that should be patched into the default plan

    Ideally, the parameter would be simplified to something like 'round_to_train' instead of the 
    triple nested dict config['aggregator']['settings']['rounds_to_train']

    This is not as trivial as it sounds, because there are special Plan (as in, the plan class) variables
    that need to be overwritten as well. One way simplifying this complexity is patching the plan, 
    saving the new plan, the reloading the plan again
    """

    for param in config:
      reference_to_plan_config = find_plan_param(param,plan)
      reference_to_plan_config[param] = config[param]

    #Save plan
    Dump('plan/plan_modified.yaml',plan)

    #Reinitialize plan
    plan = Plan.Parse(plan_config_path = Path(plan_config),
                      cols_config_path = Path(cols_config),
                      data_config_path = Path(data_config))

    return plan


def run_experiment(collaborator_dict,config={}):
    """
    Core function that executes the FL Plan. 

    Args:
        collaborator_dict : dict {collaborator_name(str): FederatedModel}
            This dictionary defines which collaborators will participate in the experiment, as well as a reference to that collaborator's federated model.
        override_config : dict {flplan.key : flplan.value}
            Override any of the plan parameters at runtime using this dictionary. To get a list of the available options, execute `fx.get_plan()`

    Returns:
        final_federated_model : FederatedModel
            The final model resulting from the federated learning experiment
    """

    from sys       import path

    file = Path(__file__).resolve()
    root = file.parent.resolve() # interface root, containing command modules
    work = Path.cwd().resolve()

    path.append(   str(root))
    path.insert(0, str(work))

    #TODO: Fix this implementation. The full plan parsing is reused here, 
    #but the model and data will be overwritten based on user specifications
    plan_config = 'plan/plan.yaml'
    cols_config = 'plan/cols.yaml'
    data_config = 'plan/data.yaml'

    plan = Plan.Parse(plan_config_path = Path(plan_config),
                      cols_config_path = Path(cols_config),
                      data_config_path = Path(data_config))

    if 'rounds_to_train' in config:
        plan.config['aggregator']['settings']['rounds_to_train'] = config['rounds_to_train']
        plan.rounds_to_train = config['rounds_to_train']
    rounds_to_train = plan.config['aggregator' ]['settings']['rounds_to_train'] 

    if 'tasks.locally_tuned_model_validation.aggregation_type' in config:
        plan.config['tasks']['locally_tuned_model_validation']['aggregation_type'] = config['tasks.locally_tuned_model_validation.aggregation_type']
        #logger.info('custom aggregation type set')
        #logger.info(f'{plan.config}')

    #Overwrite plan values
    plan.authorized_cols = list(collaborator_dict)
    tensor_pipe = plan.get_tensor_pipe() 

    #This must be set to the final index of the list (this is the last tensorflow session to get created)
    plan.runner_ = list(collaborator_dict.values())[-1]
    model = plan.runner_

    #Initialize model weights
    init_state_path = plan.config['aggregator' ]['settings']['init_state_path']
    tensor_dict, holdout_params = split_tensor_dict_for_holdouts(logger, 
                                                                 plan.runner_.get_tensor_dict(False))

    model_snap = construct_model_proto(tensor_dict  = tensor_dict,
                                       round_number = 0,
                                       tensor_pipe  = tensor_pipe)

    logger.info(f'Creating Initial Weights File    🠆 {init_state_path}' )

    dump_proto(model_proto = model_snap, fpath = init_state_path)

    logger.info('Starting Experiment...')
    
    aggregator = plan.get_aggregator()

    model_states = {collaborator: None for collaborator in collaborator_dict.keys()}

    #Create the collaborators
    collaborators = {collaborator: create_collaborator(plan,collaborator,model,aggregator) for collaborator in plan.authorized_cols}

    for round_num in range(rounds_to_train):
        for col in plan.authorized_cols:

            collaborator = collaborators[col]
            model.set_data_loader(collaborator_dict[col].data_loader)

            if round_num != 0:
                model.rebuild_model(round_num,model_states[col])

            collaborator.run_simulation()

            model_states[col] = model.get_tensor_dict(with_opt_vars=True)

    #Set the weights for the final model
    model.rebuild_model(rounds_to_train-1,aggregator.last_tensor_dict,validation=True)
    return model
