import numpy as np
import tensorflow as tf
from datetime import datetime
import pickle
import uuid
from enum import Enum
import os

sess = tf.Session()

# TODO: Do we need this?
# model = None

# determines whether at each institution at the beggining of each round, 
# the first and second moments for the adam optimizer are:
# (RESET) reset to initial values
# (AGG)  set to an aggregation of last round final values from all institutions, or
# (EDGE) kept as the last round final values from that particular institution

class OptMode(Enum):
    RESET = 1
    AGG = 2
    EDGE = 3

def aggregate_tensordicts(td0, td1, n0, n1):
    return {key: np.average(np.array([td0[key], td1[key]]), axis=0, weights=[n0, n1]) \
            for key in td0}


def parse_tensor_dict(tensor_dict, tensor_dict_opt_keys):
    layer_dict = tensor_dict.copy()
    opt_dict = {key: layer_dict.pop(key) for key in tensor_dict_opt_keys}
    return layer_dict, opt_dict


def main(sess, 
         model, 
         inst_datasets,
         full_kwargs, 
         epochs, 
         rounds, 
         seed, 
         abort_patience, 
         iterations, 
         opt_mode, 
         minimum_accept, 
         training_dice_test):
    
    config = full_kwargs
    print('Running experiment with config {}'.format(config))    
    
    # TODO: Do I need this?
    # global model

    # save the tensor_dict optimizer keys
    tensor_dict_opt_keys = [v.name for v in model.opt_vars]

    # infer the number of institutions
    num_insts = len(inst_datasets)    
    
    # create our results dictionary
    results = {'config': config, 'by_seed': {}}
    
    # if the directory ./results does not exist, create it
    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    # create a unique name for the results file
    results_path = os.path.join('.', 'results', '{}.pkl'.format(uuid.uuid4()))
    
    # create the global validation set
    # each inst dataset is a 4-tuple: X_train, y_train, X_val, y_val
    X_global_val = np.concatenate([dataset[2] for dataset in inst_datasets], axis=0) 
    y_global_val = np.concatenate([dataset[3] for dataset in inst_datasets], axis=0)
    
    while iterations > 0:
        # note the seed coupling below. The random seed for tf effects the model weight initialization, whereas the 
        # ramdom seed for numpy effects the data set shuffling. If you want to explore these effects independently 
        # be sure to define an independent seed for numpy and track it in the config and results.
        tf.set_random_seed(seed)
        np.random.seed(seed+1)

        # init our parameters
        sess.run(tf.global_variables_initializer())
        
        # get our parameters
        layer_dict, opt_dict = parse_tensor_dict(model.get_tensor_dict(with_opt_vars=True), 
                                                tensor_dict_opt_keys)

        # save the initial optimizer state
        init_opt_dict = opt_dict

        # establish the first instances of roundstart tensor dicts
        roundstart_opt_dicts = []
        for _ in range(num_insts):
            roundstart_opt_dicts.append(init_opt_dict)
        roundstart_layer_dict = layer_dict
        
        results_by_round = []
        best_score = 0
        previous_score = 0
        rounds_score_flat = 0
        for r in range(rounds):
            round_results = {}
            round_results['training_results'] = []
            if training_dice_test:
                round_results['training_DCs'] = []
                
            agg_data_size = 0

            for inst in range(num_insts):

                # overwrite the model using current insitution
                model.X_train, model.y_train, _, _ = inst_datasets[inst]

                model.set_tensor_dict(dict(**roundstart_layer_dict, 
                                           **roundstart_opt_dicts[inst]), 
                                      with_opt_vars=True)

                inst_data_size = model.X_train.shape[0]
                training_epochs = epochs
                    
                # train the institution model
                training_results = [model.train_epoch() for e in range(training_epochs)]
                round_results['training_results'].append(training_results)
                
                # get the weights from the institution model
                inst_layer_dict, inst_opt_dict = parse_tensor_dict(model.get_tensor_dict(), 
                                                                  tensor_dict_opt_keys)
                
                if opt_mode == OptMode.EDGE:
                    roundstart_opt_dicts[inst] = inst_opt_dict
                    
                if agg_data_size == 0:
                    agg_layer_dict = inst_layer_dict
                    agg_opt_dict = inst_opt_dict
                else:
                    agg_layer_dict = aggregate_tensordicts(agg_layer_dict,
                                                          inst_layer_dict,
                                                          agg_data_size,
                                                          inst_data_size)
                    if opt_mode == OptMode.AGG:
                        agg_opt_dict = aggregate_tensordicts(agg_opt_dict,
                                                            inst_opt_dict,
                                                            agg_data_size,
                                                            inst_data_size)
                agg_data_size += inst_data_size

            # establish next roundstart tensor values
            # if opt_mode is EDGE, roundstart_opt_dict is already correct
            roundstart_layer_dict = agg_layer_dict
            if opt_mode == OptMode.RESET:
                for idx in range(num_insts):
                    roundstart_opt_dicts[idx] = init_opt_dict
            elif opt_mode == OptMode.AGG:
                for idx in range(num_insts):
                    roundstart_opt_dicts[idx] = agg_opt_dict
                    
            # overwrite model with global model for global validation
            # opt setting and train data not needed here (no training)
            model.set_tensor_dict(agg_layer_dict, with_opt_vars=False)
            model.X_val = X_global_val 
            model.y_val = y_global_val

            # test the model on the combined validation and store the results
            score = model.validate()
            round_results['val_score'] = score
            
            if training_dice_test:
                # overwrite validation sets with local training data and validate
                # each inst dataset is a 4-tuple: X_train, y_train, X_val, y_val
                for inst in range(num_insts):
                    model.X_val = inst_datasets[inst][2]
                    model.y_val = inst_datasets[inst][3]
                    round_results['training_DCs'].append(model.validate())
                print('training DCs: {}'.format(round_results["training_DCs"]))
                
            # update the 'best score'
            if score > best_score:
                best_score = score
                
            # save the results in the by-round list
            results_by_round.append(round_results)
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            # print the round results
            print('{} {} round {} complete with '.format(now, seed, r) \
              + 'val score {:0.04} (best so far {:0.04})'.format(round_results["val_score"], best_score))
              
            # if training is stuck, abort
            if score == previous_score:
                rounds_score_flat += 1
                if rounds_score_flat >= abort_patience:
                    print('Aborting due to abort patience {}'.format(abort_patience))
                    break
            else:
                rounds_score_flat = 0
                previous_score = score
                
        results['by_seed'][seed] = results_by_round
        
        seed += 1
        
        # ensure successful iteration before counting
        if best_score > minimum_accept:
            iterations -= 1
            
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)