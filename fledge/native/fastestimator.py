from pathlib import Path
from logging import getLogger
from fledge.federated import Plan
from fledge.protocols import dump_proto, construct_model_proto
from fledge.utilities import split_tensor_dict_for_holdouts
import fledge.native as fx
from fledge.federated.data import FastEstimatorDataLoader
from fledge.federated.task import FastEstimatorTaskRunner

logger = getLogger(__name__)

class FederatedFastEstimator:
    def __init__(self, estimator, rounds=10, **kwargs):
        self.estimator = estimator
        self.rounds = rounds
        fx.init('fe', **kwargs)

    def fit(self):
        from sys       import path

        file = Path(__file__).resolve()
        root = file.parent.resolve() # interface root, containing command modules
        work = Path.cwd().resolve()

        path.append(   str(root))
        path.insert(0, str(work))

        #TODO: Fix this implementation. The full plan parsing is reused here, 
        #but the model and data will be overwritten based on user specifications
        plan_config = (Path(fx.WORKSPACE_PREFIX) / 'plan'/'plan.yaml')
        cols_config = (Path(fx.WORKSPACE_PREFIX) / 'plan'/'cols.yaml')
        data_config = (Path(fx.WORKSPACE_PREFIX) / 'plan'/'data.yaml')

        plan = Plan.Parse(plan_config_path = plan_config,
                        cols_config_path = cols_config,
                        data_config_path = data_config)

        plan.config['aggregator']['settings']['rounds_to_train'] = self.rounds
        plan.rounds_to_train = self.rounds
        self.rounds = plan.config['aggregator' ]['settings']['rounds_to_train'] 

        #Overwrite plan values
        tensor_pipe = plan.get_tensor_pipe() 
        data_loader = FastEstimatorDataLoader(self.estimator.pipeline)
        #This must be set to the final index of the list (this is the last tensorflow session to get created)
        plan.runner_ = FastEstimatorTaskRunner(self.estimator, data_loader=data_loader)
        runner = plan.runner_

        #Do PKI setup here 

        #setup_pki(aggregator_fqdn,collaborator_names)

        #Set rounds to train


        #Initialize model weights
        init_state_path = plan.config['aggregator' ]['settings']['init_state_path']
        tensor_dict, holdout_params = split_tensor_dict_for_holdouts(logger, 
                                                                    plan.runner_.get_tensor_dict(False),
                                                                    {})

        model_snap = construct_model_proto(tensor_dict  = tensor_dict,
                                        round_number = 0,
                                        tensor_pipe  = tensor_pipe)

        logger.info(f'Creating Initial Weights File    🠆 {init_state_path}' )

        dump_proto(model_proto = model_snap, fpath = init_state_path)

        logger.info('Starting Experiment...')
        
        aggregator = plan.get_aggregator()

        model_states = {collaborator: None for collaborator in plan.authorized_cols}

        #Create the collaborators
        collaborators = {collaborator: fx.create_collaborator(plan,collaborator,runner,aggregator) for collaborator in plan.authorized_cols}

        for round_num in range(self.rounds):
            for col in plan.authorized_cols:

                collaborator = collaborators[col]

                if round_num != 0:
                    runner.rebuild_model(round_num,model_states[col])

                collaborator.run_simulation()

                model_states[col] = runner.get_tensor_dict(with_opt_vars=True)

        #TODO This will return the model from the last collaborator, NOT the final aggregated model (though they should be similar). 
        #There should be a method added to the aggregator that will load the best model from disk and return it 
        return runner.model

        
def split_data(train,eva,test,rank,collaborator_count):
        """
        Split data into N parts, where N is the collaborator count
        """

        if collaborator_count == 1:
            return train,eva,test

        fraction = [ 1.0 / float(collaborator_count) ] 
        fraction *= (collaborator_count - 1)
        
        #Expand the split list into individual parameters
        train_split = train.split(*fraction)
        eva_split   = eva.split(*fraction)
        test_split  = test.split(*fraction)

        train = [train]
        eva = [eva]
        test = [test]

        if type(train_split) is not list:
            train.append(train_split)
            eva.append(eva_split)
            test.append(test_split)
        else:
            #Combine all partitions into a single list
            train = [train] + train_split
            eva   = [eva] + eva_split
            test  = [test] + test_split

        #Extract the right shard
        train = train[rank-1]
        eva   = eva[rank-1]
        test  = test[rank-1]

        return train,eva,test