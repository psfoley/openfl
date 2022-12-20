from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from federated_flow import FLSpec
from federated_flow import Aggregator, Collaborator, LocalRuntime
from placement import aggregator, collaborator
import torchvision.transforms as transforms
import pickle
from pathlib import Path

from privacy_meter.model import PytorchModelTensor
import copy
from auditor import ReferenceAuditor,plot_auc_history, plot_tpr_history, plot_roc_history,PM_report

import time
import os
import argparse
from cifar10_loader import CIFAR10
import warnings
 
warnings.filterwarnings("ignore")

batch_size_train = 32
batch_size_test = 1000
learning_rate = 0.005
momentum = 0.9 
log_interval = 10

# set the random seed for repeatable results
random_seed = 10
torch.manual_seed(random_seed)
np.random.seed(random_seed)

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

def default_optimizer(model,optimizer_type=None,optimizer_like=None):
    """
    Return a new optimizer based on the optimizer_type or the optimizer template
    """
    if optimizer_type == 'SGD' or isinstance(optimizer_like,optim.SGD):
        return optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=momentum)
    elif optimizer_type == 'Adam' or isinstance(optimizer_like,optim.Adam):
        return optim.Adam(model.parameters())

def FedAvg(models):
    new_model = models[0]    
    if len(models) >1:
        state_dicts = [model.state_dict() for model in models]
        state_dict = new_model.state_dict()
        for key in models[1].state_dict():
            state_dict[key] = np.sum([state[key] for state in state_dicts],axis=0) / len(models)
        new_model.load_state_dict(state_dict)
    return new_model

def inference(network,test_loader,device):
    network.eval()
    network.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = network(data)
        criterion =  nn.CrossEntropyLoss()
        test_loss +=  criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader)
    print('Dataset set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))  
    accuracy = float(correct.item() / len(test_loader.dataset))
    network.to('cpu')
    return accuracy

def optimizer_to_device(optimizer, device):

    if optimizer.state_dict()['state'] != {}:
        if isinstance(optimizer,optim.SGD):
            for param in optimizer.param_groups[0]['params']:
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        elif isinstance(optimizer,optim.Adam):
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    else:
        raise (ValueError(f'No dict keys in optimizer state: please check'))

def load_previous_round_model_and_optimizer_and_perform_testing(model, global_model, optimizer, collaborator_name, round_num, device):
    """
    # may be helpful for debugging
    if isinstance(optimizer, optim.SGD):
        print(f'Visualization of optimizer state dict (SGD)')
        print(f"optimizer.state_dict()[param_groups][0] keys: ", optimizer.state_dict()['param_groups'][0].keys())
        print(f"type of optimizer.state_dict()[param_groups][0][params]: ", type(optimizer.state_dict()['param_groups'][0]['params']))
        print(f"length of optimizer.state_dict()[param_groups][0][params]: ", len(optimizer.state_dict()['param_groups'][0]['params']))
        print(f"optimizer.state_dict()[state] keys: ", optimizer.state_dict()['state'].keys())
        print(f"optimizer.state_dict()[state][0] keys: ", optimizer.state_dict()['state'][0].keys())
        print(f"optimizer.state_dict()[state][1] keys: ", optimizer.state_dict()['state'][1].keys())
        print(f"length of optimizer.state_dict()[state]: ", len(optimizer.state_dict()['state']))
        print(f"type of optimizer.state_dict()[state][0][momentum_buffer]: ", type(optimizer.state_dict()['state'][0]['momentum_buffer']))            
    """
    print(f'Loading model and optimizer state dict for round {round_num-1}')
    model_prevround = Net()    # instanciate a new model
    model_prevround = model_prevround.to(device)
    optimizer_prevround = default_optimizer(model_prevround,optimizer_like=optimizer)
    if os.path.isfile(f'Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}.pickle'):
        with open(f'Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}.pickle', "rb") as f:
            model_prevround_config = pickle.load(f)
            model_prevround.load_state_dict(model_prevround_config["model_state_dict"])
            optimizer_prevround.load_state_dict(model_prevround_config["optim_state_dict"])
            
            for param_tensor in model.state_dict():                        
                for tensor_1, tensor_2 in zip(model.state_dict()[param_tensor], global_model.state_dict()[param_tensor]):
                    if torch.equal(tensor_1, tensor_2) is not True:
                            raise (ValueError(f'Before train model state and global state do not match for collaborator: {collaborator_name} at round {round_num-1}.'))

                if isinstance(optimizer, optim.SGD):
                    if optimizer.state_dict()['state'] != {}: 
                        for param_idx in optimizer.state_dict()['param_groups'][0]['params']:
                            for tensor_1, tensor_2 in zip(optimizer.state_dict()['state'][param_idx]['momentum_buffer'], optimizer_prevround.state_dict()['state'][param_idx]['momentum_buffer']):
                                if torch.equal(tensor_1.to(device), tensor_2.to(device)) is not True:
                                    raise (ValueError(f'Momentum buffer data is not the same between current round before train and previous round after train for collaborator: {collaborator_name} at round {round_num-1}'))
                    else:
                        raise(ValueError(f'Current optimizer state is empty'))

                model_params = [model.state_dict()[param_tensor] for param_tensor in model.state_dict()]
                for idx, param in enumerate(optimizer.param_groups[0]['params']):
                    for tensor_1, tensor_2 in zip(param.data, model_params[idx]):
                        if torch.equal(tensor_1, tensor_2) is not True:
                            raise (ValueError(f'Model and optimizer do not point to the same params for collaborator: {collaborator_name} at round {round_num-1}.'))
            
    else:
        raise(ValueError(f'No such name of pickle file exists'))



def load_previous_round_reference_model_and_optimizer_and_perform_testing(model, global_model, optimizer, collaborator_name, round_num, device,reference_idx):
    """
    # may be helpful for debugging
    if isinstance(optimizer, optim.SGD):
        print(f'Visualization of optimizer state dict (SGD)')
        print(f"optimizer.state_dict()[param_groups][0] keys: ", optimizer.state_dict()['param_groups'][0].keys())
        print(f"type of optimizer.state_dict()[param_groups][0][params]: ", type(optimizer.state_dict()['param_groups'][0]['params']))
        print(f"length of optimizer.state_dict()[param_groups][0][params]: ", len(optimizer.state_dict()['param_groups'][0]['params']))
        print(f"optimizer.state_dict()[state] keys: ", optimizer.state_dict()['state'].keys())
        print(f"optimizer.state_dict()[state][0] keys: ", optimizer.state_dict()['state'][0].keys())
        print(f"optimizer.state_dict()[state][1] keys: ", optimizer.state_dict()['state'][1].keys())
        print(f"length of optimizer.state_dict()[state]: ", len(optimizer.state_dict()['state']))
        print(f"type of optimizer.state_dict()[state][0][momentum_buffer]: ", type(optimizer.state_dict()['state'][0]['momentum_buffer']))            
    """
    print(f'Loading model and optimizer state dict for round {round_num-1}')
    model_prevround = Net()    # instanciate a new model
    model_prevround = model_prevround.to(device)
    optimizer_prevround = default_optimizer(model_prevround,optimizer_like=optimizer)
    if os.path.isfile(f'Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}_{reference_idx}.pickle'):
        with open(f'Collaborator_{collaborator_name}_model_config_roundnumber_{round_num-1}_{reference_idx}.pickle', "rb") as f:
            model_prevround_config = pickle.load(f)
            model_prevround.load_state_dict(model_prevround_config["model_state_dict"])
            optimizer_prevround.load_state_dict(model_prevround_config["optim_state_dict"])
            
            for param_tensor in model.state_dict():                        
                for tensor_1, tensor_2 in zip(model.state_dict()[param_tensor], global_model.state_dict()[param_tensor]):
                    if torch.equal(tensor_1, tensor_2) is not True:
                            raise (ValueError(f'Before train model state and global state do not match for collaborator: {collaborator_name} at round {round_num-1}.'))

                if isinstance(optimizer, optim.SGD):
                    if optimizer.state_dict()['state'] != {}: 
                        for param_idx in optimizer.state_dict()['param_groups'][0]['params']:
                            for tensor_1, tensor_2 in zip(optimizer.state_dict()['state'][param_idx]['momentum_buffer'], optimizer_prevround.state_dict()['state'][param_idx]['momentum_buffer']):
                                if torch.equal(tensor_1.to(device), tensor_2.to(device)) is not True:
                                    raise (ValueError(f'Momentum buffer data is not the same between current round before train and previous round after train for collaborator: {collaborator_name} at round {round_num-1}'))
                    else:
                        raise(ValueError(f'Current optimizer state is empty'))

                model_params = [model.state_dict()[param_tensor] for param_tensor in model.state_dict()]
                for idx, param in enumerate(optimizer.param_groups[0]['params']):
                    for tensor_1, tensor_2 in zip(param.data, model_params[idx]):
                        if torch.equal(tensor_1, tensor_2) is not True:
                            raise (ValueError(f'Model and optimizer do not point to the same params for collaborator: {collaborator_name} at round {round_num-1}.'))
            
    else:
        raise(ValueError(f'No such name of pickle file exists'))


def save_current_round_model_and_optimizer_for_next_round_testing(model, optimizer, collaborator_name, round_num): 
    model_config = {
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
    }        
    with open(f'Collaborator_{collaborator_name}_model_config_roundnumber_{round_num}.pickle', "wb") as f:
        pickle.dump(model_config, f)            

def save_current_round_reference_model_and_optimizer_for_next_round_testing(model, optimizer, collaborator_name, round_num, reference_idx): 
    model_config = {
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
    }        
    with open(f'Collaborator_{collaborator_name}_model_config_roundnumber_{round_num}_{reference_idx}.pickle', "wb") as f:
        pickle.dump(model_config, f)  

class FederatedFlow(FLSpec):

    def __init__(self, model, optimizers,reference_models,reference_optimizers, device='cpu', total_rounds=10, top_model_accuracy=0, flow_internal_loop_test=False,  **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.global_model = Net()
        self.optimizers = optimizers
        self.total_rounds = total_rounds 
        self.top_model_accuracy = top_model_accuracy
        self.device = device
        self.flow_internal_loop_test = flow_internal_loop_test
        self.round_num = 0     # starting round
        self.reference_optimizers = None # save the reference_optimizer state to disk for pass OOM
        self.reference_models = reference_models
        print(20*"#")
        print(f'Round {self.round_num}...')
        print(20*"#")

    @aggregator
    def start(self):
        self.start_time = time.time()
        print(f'Performing initialization for model')
        self.collaborators = self.runtime.collaborators
        self.private = 10
        self.next(self.aggregated_model_validation,foreach='collaborators',exclude=['private'])
    
    @collaborator(num_gpus=1)
    def aggregated_model_validation(self):
        print(f'Performing aggregated model validation for collaborator {self.input} in round {self.round_num}')
        
        # self.model = self.model.to(self.device)
        # self.global_model = self.global_model.to(self.device)

        # verifying that model went to the correct GPU device
        # print(next(self.model.parameters()).device)
        # print(self.device)

        # assert next(self.model.parameters()).device == self.device
        # assert next(self.global_model.parameters()).device == self.device

        print("Test dataset performance")
        self.agg_validation_score = inference(self.global_model, self.test_loader, self.device)
        print(f'{self.input} value of {self.agg_validation_score}')
        self.collaborator_name = self.input
        self.next(self.train)


    @collaborator(num_gpus=1)
    def train(self):
        print(20*"#")
        print(f'Performing model training for collaborator {self.input} in round {self.round_num}')
        print(self.device)
        self.model.to(self.device)
        self.optimizer = default_optimizer(self.model,optimizer_like=self.optimizers[self.input])
        optimizer_save_path  = self.local_pm_info.log_dir
        if self.round_num > 0:
            self.optimizer.load_state_dict(torch.load(f'{optimizer_save_path}/{self.input}_optimizer.pt'))
            optimizer_to_device(optimizer=self.optimizer, device=self.device)

            if self.flow_internal_loop_test:
                load_previous_round_model_and_optimizer_and_perform_testing(self.model, self.global_model, self.optimizer, self.collaborator_name, self.round_num, self.device)

        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
          data = data.to(self.device)
          target = target.to(self.device)
          self.optimizer.zero_grad()
          output = self.model(data)
          criterion =  nn.CrossEntropyLoss()
          loss = criterion(output, target).to(self.device)
          loss.backward()
          self.optimizer.step()
          if batch_idx % log_interval == 0:
            print('Train Epoch: 1 [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               batch_idx * len(data), len(self.train_loader.dataset),
              100. * batch_idx / len(self.train_loader), loss.item()))
            self.loss = loss.item()
        
       
        self.training_completed = True
        if self.flow_internal_loop_test:
            save_current_round_model_and_optimizer_for_next_round_testing(self.model, self.optimizer, self.collaborator_name, self.round_num)

        self.model.to('cpu')
        
        optimizer_to_device(optimizer=self.optimizer, device='cpu')
        torch.save(self.optimizer.state_dict(), f'{optimizer_save_path}/{self.input}_optimizer.pt')

        self.next(self.local_model_validation)

    @collaborator(num_gpus=1)
    def local_model_validation(self):
        print(f'Performing local model validation for collaborator {self.input} in round {self.round_num}')
        print(self.device)
        start_time = time.time()

        print('Test dataset performance')
        self.local_validation_score = inference(self.model,self.test_loader,self.device)
        print('Train dataset performance')
        self.local_validation_score_train = inference(self.model,self.train_loader,self.device)
        
        print(f'Doing local model validation for collaborator {self.input}: {self.local_validation_score}')
        print('local validation time cost {}'.format(time.time()-start_time))

    

        if self.round_num ==0 or self.round_num % self.local_pm_info.interval==0 or self.round_num == self.total_rounds:
            print('Performing Auditing')
            self.next(self.audit)        
        else:
            self.next(self.join, exclude=['training_completed'])

    @collaborator(num_gpus=1)
    def audit(self):
        print(f'Performing local and global model auditing for collaborator {self.input} in round {self.round_num}')
        begin_time = time.time()

        datasets = {
            'train': self.train_dataset,
            'test': self.test_dataset,
            'audit':self.reference_datasets
        }

        start_time = time.time()
        optimizer_save_path = self.local_pm_info.log_dir
        # batch_size for the PytorchModelTensor indicates batch size for computing the signals.
        # for computing loss and logits, it can be large, e.g., 1000.
        # for computing the signal_norm, it should be around 25. Otherwise, we will get OOM.
        target_model = PytorchModelTensor(copy.deepcopy(self.global_model),nn.CrossEntropyLoss(),self.device,batch_size=500)
        target_reference_models = [PytorchModelTensor(copy.deepcopy(model),nn.CrossEntropyLoss(),self.device,batch_size=500) for model in self.reference_models]
        self.global_pm_info = ReferenceAuditor(target_model,target_reference_models,datasets,self.global_pm_info)
        self.global_pm_info.update_history('round',self.round_num)
        target_model.model_obj.to('cpu')
        del target_model
        for model in target_reference_models:
            model.model_obj.to('cpu')
            del model
        
        print('population attack for the global model uses {}'.format(time.time()-start_time))


        # training for the reference models 
        start_time = time.time()
        train_reference_device = self.device
        for idx,reference_model_global in enumerate(self.reference_models):
            # add the following code in order to ensure each reference model updates is for one reference model
            reference_model = Net()
            reference_model.load_state_dict(reference_model_global.state_dict())
            reference_model.train()
            reference_model.to(train_reference_device) # otherwise, we encounter OOM
            reference_data_loader = torch.utils.data.DataLoader(self.reference_datasets[idx],batch_size=batch_size_train, shuffle=True)
        
            reference_optimizer = default_optimizer(reference_model,optimizer_like=self.optimizer) # assuming the adversary knows how we update the target model 
            if self.round_num > 0:
                reference_optimizer.load_state_dict(torch.load(f'{optimizer_save_path}/{self.input}_refer_{idx}_optimizer.pt')) # update to load and save state_dict directly to save memory
                optimizer_to_device(optimizer=reference_optimizer,device=train_reference_device)
                if self.flow_internal_loop_test:
                    load_previous_round_reference_model_and_optimizer_and_perform_testing(reference_model, self.reference_models[idx], reference_optimizer, self.collaborator_name, self.round_num, self.device,idx)

            
            for batch_idx, (data, target) in enumerate(reference_data_loader):
                data = data.to(train_reference_device)
                target = target.to(train_reference_device)
                reference_optimizer.zero_grad()
                output = reference_model(data)
                criterion =  nn.CrossEntropyLoss()
                loss = criterion(output, target).to(train_reference_device)
                loss.backward()
                reference_optimizer.step()
                if batch_idx % log_interval == 0:
                    print('{} Reference Train Epoch: 1 [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(idx,
                    batch_idx * len(data), len(reference_data_loader.dataset),
                    100. * batch_idx / len(reference_data_loader), loss.item()))
                    self.loss = loss.item()
            
            acc  = inference(reference_model, reference_data_loader, self.device)
            print(f'during the training reference model {idx} accuracy: {acc}')           
            reference_model.to('cpu')
            optimizer_to_device(reference_optimizer,'cpu')
            torch.save(reference_optimizer.state_dict(), f'{optimizer_save_path}/{self.input}_refer_{idx}_optimizer.pt')

            # self.reference_optimizers[self.input][idx] = copy.deepcopy(reference_optimizer.state_dict())
            
            self.reference_models[idx] = copy.deepcopy(reference_model)
            if self.flow_internal_loop_test:
                save_current_round_reference_model_and_optimizer_for_next_round_testing(reference_model, reference_optimizer, self.collaborator_name, self.round_num,idx)
            
        print(f'debug: training the reference models costs {time.time()-start_time} seconds')


        
        start_time = time.time()
        # move all the target models and reference models to the cpu
        target_model = PytorchModelTensor(copy.deepcopy(self.model),nn.CrossEntropyLoss(),self.device, batch_size=500)
        # test the performance on the reference_data_loader

        target_reference_models = [PytorchModelTensor(copy.deepcopy(model),nn.CrossEntropyLoss(),self.device, batch_size=500) for model in self.reference_models]
        self.local_pm_info =  ReferenceAuditor(target_model,target_reference_models,datasets,self.local_pm_info)
        self.local_pm_info.update_history('round',self.round_num)

        target_model.model_obj.to('cpu')
        del target_model
        for model in target_reference_models:
            model.model_obj.to('cpu')
            del model # the memory will not be increased as the training 


        self.global_model.to('cpu')
        self.model.to('cpu')
        torch.cuda.empty_cache()
        print('population attack for the local model uses {}'.format(time.time()-start_time))


        start_time = time.time()
        print('############Privacy Meter Result#########################')
        print('Local model tpr for the given FPR tolerance')
        for idx,fpr in enumerate(self.local_pm_info.fpr_tolerance):
            for sidx,signal in enumerate(self.local_pm_info.signals):
                print('{} - Actual FPR: {}, Actual TPR: {} '.format(signal,self.local_pm_info.history['fpr'][-1][sidx][idx],self.local_pm_info.history['tpr'][-1][sidx][idx]))
        
        print(40*'-')
        print('\n Global model tpr for the given FPR tolerance')
        for idx,fpr in enumerate(self.global_pm_info.fpr_tolerance):
            for sidx,signal in enumerate(self.global_pm_info.signals):
                print('{} - Actual FPR: {}, Actual TPR: {} '.format(signal,self.global_pm_info.history['fpr'][-1][sidx][idx],self.global_pm_info.history['tpr'][-1][sidx][idx]))

        print(40*"#")


        history_dict = {'PM Result (Local)': self.local_pm_info, 
                     'PM Result (Global)': self.global_pm_info}

        # # generate the plot for the privacy loss
        plot_tpr_history(history_dict,self.input,self.local_pm_info.fpr_tolerance)
        plot_auc_history(history_dict,self.input)
        plot_roc_history(history_dict,self.input)

        # save the privacy report
        saving_path = '{}/{}.pkl'.format(self.local_pm_info.log_dir, self.input)
        Path(self.local_pm_info.log_dir).mkdir(parents=True, exist_ok=True)
        with open(saving_path, 'wb') as handle:
            pickle.dump(history_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print('saving and ploting the attack for the both attacks uses {}'.format(time.time()-start_time))
        print('auditing time: {}'.format(time.time()-begin_time))
        #no objects on GPU.
        self.next(self.join, exclude=['training_completed'])

    @aggregator
    def join(self,inputs):
        self.average_loss = sum(input.loss for input in inputs)/len(inputs)
        self.aggregated_model_accuracy = sum(input.agg_validation_score for input in inputs)/len(inputs)
        self.local_model_accuracy = sum(input.local_validation_score for input in inputs)/len(inputs)       
        print(f'Average aggregated model validation values = {self.aggregated_model_accuracy}')
        print(f'Average training loss = {self.average_loss}')
        print(f'Average local model validation values = {self.local_model_accuracy}')
    
        self.model = FedAvg([input.model.cpu() for input in inputs])
        self.global_model.load_state_dict(deepcopy(self.model.state_dict()))
        # self.optimizers.update({input.collaborator_name: input.optimizer for input in inputs})

        self.reference_models = [FedAvg([input.reference_models[i].cpu() for input in inputs]) for i in range(len(inputs[0].reference_models))]
        # self.reference_optimizers.update({input.collaborator_name: input.reference_optimizers[input.collaborator_name] for input in inputs})
       

        self.next(self.check_round_completion)

    @aggregator
    def check_round_completion(self):
        if self.round_num != self.total_rounds:
            if self.aggregated_model_accuracy > self.top_model_accuracy:
                print(f'Accuracy improved to {self.aggregated_model_accuracy} for round {self.round_num}')
                self.top_model_accuracy = self.aggregated_model_accuracy
            self.round_num += 1
            print(20*"#")
            print(f'Round {self.round_num}...')
            print(20*"#")
            self.next(self.aggregated_model_validation, foreach='collaborators', exclude=['private'])
        else:
            self.next(self.end)

    @aggregator
    def end(self):
        print(20*"#")
        print(f'All rounds completed successfully')
        print(20*"#")
        print(f'This is the end of the flow')
        print(20*"#")

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description=__doc__)
    # argparser.add_argument(
    #     '--gpu',
    #     default="multi",
    #     help='single(collaborators running on single GPU) or multi(each collaborator is assigned to a GPU) (default:"single")')
    # argparser.add_argument(
    #     '--deviceid_single',
    #     default="0",
    #     help='Device id for single GPU (default:"0")')
    # argparser.add_argument(
    #     '--deviceid_multi',
    #     default="0000", 
    #     help='Device id for multiple GPU (default:"0123")')
    
    argparser.add_argument(
        '--audit_dataset_ratio',type=float,
        default=0.4, 
        help='Indicate the what fraction of the sample will be used for auditing')
    argparser.add_argument(
        '--test_dataset_ratio',type=float,
        default=0.3, 
        help='Indicate the what fraction of the sample will be used for testing')
    argparser.add_argument(
        '--train_dataset_ratio',type=float,
        default=0.3, 
        help='Indicate the what fraction of the sample will be used for training')
    argparser.add_argument(
        '--signals', nargs="*",  
        type=str,
        default=["loss", "logits"], 
        help='Indicate which signal to use for membership inference attack. Currently, we support: loss, gradient_norm, logits')
    argparser.add_argument(
        "--fpr_tolerance", 
        nargs="*",  
        type=float,
        default=[0.1,0.2,0.3],
        help='indicate false positive tolerance rates in which users are interested' ) 
    argparser.add_argument(
        "--log_dir",type=str, default='test_debug',
        help='Indicate where to save the privacy loss profile and log files during the training'
    )
    argparser.add_argument(
        "--comm_round",type=int,default=30,
        help='Indicate the communication round of FL'
    )    
    argparser.add_argument(
        '--auditing_interval',type=int,
        default=1, 
        help='Indicate auditing interval')
    argparser.add_argument(
        '--is_features',type=bool,
        default=True, 
        help='Indicate whether to use the gradient norm with respect to the features as a signal. It works when the gradient_norm is used as the signal')
    argparser.add_argument(
        '--layer_number',type=int,
        default=10, 
        help='Indicate whether layer to compute the gradient or gradient norm')
    argparser.add_argument(
        '--flow_internal_loop_test',type=str,
        default=False, 
        help='Indicate enabling of internal loop testing of Federated Flow')
    argparser.add_argument(
        '--optimizer_type',type=str,
        default='SGD', 
        help='Indicate optimizer to use for training')

    argparser.add_argument(
        '--num_reference_models',type=int,
        default=10, 
        help='Indicate the number of reference models for auditing')


    args = argparser.parse_args()
    # if args.deviceid_multi:
    #     args.deviceid_multi = list(args.deviceid_multi)

    # Setup participants
    aggregator = Aggregator()
    aggregator.private_attributes = {}

    # Setup collaborators with private attributes
    collaborator_names = ['Portland', 'Seattle']

    collaborators = [Collaborator(name=name) for name in collaborator_names]
    
    # if args.gpu == 'single':
    #     if torch.cuda.is_available():
    #         device = {collaborators[i].name: torch.device(f'cuda:{args.deviceid_single}') for i in range(len(collaborator_names))}
    #     else:
    #        device = {collaborators[i].name: torch.device('cpu') for i in range(len(collaborator_names))}
    # elif args.gpu == 'multi':
    #     if torch.cuda.is_available():
    #         device = {collaborators[i].name: torch.device(f'cuda:{args.deviceid_multi[i]}') for i in range(len(collaborator_names))}
    #     else:
    #        device = {collaborators[i].name: torch.device('cpu') for i in range(len(collaborator_names))}
    # else:
    #     raise Exception('input should be single or multi')
    if torch.cuda.is_available():
        device = torch.device(f'cuda:0')  # This will enable Ray library to reserve an available GPU for the task
    else:
        device = torch.device(f'cpu')
    # print(device)
    transform = transforms.Compose(
    [transforms.ToTensor()])

    cifar_train = CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    cifar_test = CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)


    
    # split the dataset
    N_total_samples = len(cifar_test) + len(cifar_train)
    train_dataset_size = int(N_total_samples*args.train_dataset_ratio)
    test_dataset_size = int(N_total_samples*args.test_dataset_ratio)
    audit_dataset_size = min(int(N_total_samples*args.audit_dataset_ratio), N_total_samples - test_dataset_size - train_dataset_size)


    X = np.concatenate([cifar_test.data, cifar_train.data])
    Y = np.concatenate([cifar_test.targets, cifar_train.targets]).tolist()

    train_dataset = deepcopy(cifar_train)
    train_dataset.data = X[:train_dataset_size]
    train_dataset.targets = Y[:train_dataset_size]

    test_dataset = deepcopy(cifar_test)
    test_dataset.data = X[train_dataset_size:train_dataset_size+test_dataset_size]
    test_dataset.targets = Y[train_dataset_size:train_dataset_size+test_dataset_size]


    population_dataset = deepcopy(cifar_test)
    population_dataset.data = X[-audit_dataset_size:]
    population_dataset.targets = Y[-audit_dataset_size:]

    print('Dataset info (total {}): train - {}, test - {}, audit - {}'.format(N_total_samples,len(train_dataset),len(test_dataset),len(population_dataset)))

    # partition the dataset for clients
    for idx, collaborator in enumerate(collaborators):

        # construct the training and test and population dataset
        local_train = deepcopy(train_dataset)
        local_test = deepcopy(test_dataset)
        local_population = deepcopy(population_dataset)


        # train, test and population dataset for each collaborator 
        local_population.data = population_dataset.data[idx::len(collaborators)]
        local_population.targets = population_dataset.targets[idx::len(collaborators)]
        population_data_index  = np.arange(0, len(local_population), 1, dtype=int)

        local_train.data = train_dataset.data[idx::len(collaborators)]
        local_train.targets = train_dataset.targets[idx::len(collaborators)]
    
        local_test.data = test_dataset.data[idx::len(collaborators)]
        local_test.targets = test_dataset.targets[idx::len(collaborators)]
        
        local_reference_datasets = []

        for reference_idx in range(args.num_reference_models):
            # get the index from the population dataset with replacement 
            selected_index = np.random.choice(population_data_index, size=len(local_train), replace=False, p=None)
            local_reference_data = deepcopy(local_population)
            local_reference_data.data = local_population.data[selected_index]
            local_reference_data.targets = list(np.array(local_population.targets)[selected_index])
            local_reference_datasets.append(local_reference_data)



        

        # initialize pm report to track the privacy loss during the training
        local_pm_info = PM_report(  fpr_tolerance_list=args.fpr_tolerance, 
                                    is_report_roc=True,
                                    level='local',
                                    signals=args.signals,
                                    log_dir = args.log_dir,
                                    interval=args.auditing_interval,
                                    other_info={'is_features':args.is_features,'layer_number':args.layer_number})
        global_pm_info = PM_report( fpr_tolerance_list=args.fpr_tolerance, 
                                    is_report_roc=True,
                                    level='global',
                                    signals=args.signals,
                                    log_dir = args.log_dir,
                                    interval=args.auditing_interval,
                                    other_info={'is_features':args.is_features,'layer_number':args.layer_number})

        Path(local_pm_info.log_dir).mkdir(parents=True, exist_ok=True)
        Path(global_pm_info.log_dir).mkdir(parents=True, exist_ok=True)

        collaborator.private_attributes = {
                'local_pm_info': local_pm_info,
                'global_pm_info': global_pm_info,
                'train_dataset': local_train, 
                'test_dataset': local_test, # provide the dataset obj for the auditing purpose
                'reference_datasets':  local_reference_datasets,
                'train_loader': torch.utils.data.DataLoader(local_train,batch_size=batch_size_train, shuffle=True),
                'test_loader': torch.utils.data.DataLoader(local_test,batch_size=batch_size_test, shuffle=False)
                
        }

    local_runtime = LocalRuntime(aggregator=aggregator, collaborators=collaborators)
    print(f'Local runtime collaborators = {local_runtime._collaborators}')

    ## change to the internal flow loop
    model = Net()    
    top_model_accuracy = 0
    optimizers = {collaborator.name: default_optimizer(model,optimizer_type=args.optimizer_type) for collaborator in collaborators}
    
    reference_models = [Net()]*args.num_reference_models
    reference_optimizer = {collaborator.name: [default_optimizer(reference_model,optimizer_type=args.optimizer_type) for reference_model in reference_models] for collaborator in collaborators}
    flflow = FederatedFlow(model, optimizers, reference_models,reference_optimizer,device, args.comm_round, top_model_accuracy, args.flow_internal_loop_test)

    flflow.runtime = local_runtime
    flflow.run() 