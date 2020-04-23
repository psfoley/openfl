import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import math
from torch.nn.parameter import Parameter
import copy



class RepresentationMatchingWrapper(nn.Module):
    r"""
    Wraps a pytorch model in order to implement the representation matching scheme described in (https://arxiv.org/abs/1912.13075). 
    After wrapping, do not use the model's forward function but use this wrapper's forward function instead during training in order
    to calculate the matching loss. The wrapper uses a lot of heuristics to automatically generate the matching networks. One caveat is 
    that if you have maxpooling followed by a non-linearity, you should specify the maxpooling followed by the non-linearity in the layer stack, even though
    the order should not matter as far as the model behavior is concerned. 

    Args:
        model: Iterable of nn.Module representing the model's layer stack. The stack of layers should fully describe the model. 

        input_shape: Shape of the model input (not including the batch dimension)

        layers_of_interest: iterable of nonlinearity layer types. In the collaborator copy of the model, the activations of layers with a type present in 'layers_of_interest'
        will form the input to the matching network. 
        In the aggregator(fixed) copy of the model, the activations of these layers will be the reconstruction targets. Note that we reconstruct the activations of a 
        layer of interest in the aggregator copy from the activations of the next layer of interest above it in the collaborator copy. If there is an intervening
        max pooling layer between two layers of interest, the activations of the collaborator layer of interest will be unpooled using the same pooling positions used 
        in the intervening pooling layer in the collaborator.
        The virtual input layer (whose output is simply the input) and the top layer are automatically layers of interest. 
           Default : [nn.ReLU,nn.Sigmoid,nn.Tanh]

        matching_kernel_size: kernel size of convolutional matching layers
           Default : 3

    """

    def __init__(self,model,input_shape,layers_of_interest = [nn.ReLU,nn.Sigmoid,nn.Tanh],matching_kernel_size = 3):
        super(RepresentationMatchingWrapper,self).__init__()
        self.collab_model = list(model)
        self.update_aggregator_model()

        
        self.match_data = [[0,nn.Identity,input_shape]]
        x = torch.randn(1,*input_shape)
        for layer_idx,layer in enumerate(model):
            x = layer(x)
            matched_layer_type = [x for x in layers_of_interest if isinstance(layer,x)]
            if matched_layer_type or layer_idx == len(model)-1:
                self.match_data.append([layer_idx,matched_layer_type[0] if matched_layer_type else None,x.size()[1:]])

        self.matching_layers = nn.ModuleList()
        for src_layer,tgt_layer in zip(self.match_data[1:],self.match_data[:-1]):
            src_size,tgt_size = src_layer[-1],tgt_layer[-1]
            tgt_activation = tgt_layer[1]
            if len(src_size) == 1: #Fully connected src layer
               self.matching_layers.append(nn.Sequential(nn.Linear(src_size[0],np.prod(tgt_size)),tgt_activation()))
            elif len(src_size) == 3: #Conv src layer
               self.matching_layers.append(nn.Sequential(nn.Conv2d(src_size[0],tgt_size[0],matching_kernel_size,padding = matching_kernel_size //2),tgt_activation()))

        self.total_matching_loss = 0
    def update_aggregator_model(self):
        r"""
        Copies the collaborator model to the internal aggregator model. Call this function after the collaborator replaces
        its model by the aggregator-provided model at the beginning of the round
        """
        self.aggregator_model = [copy.deepcopy(model_layer) for model_layer in self.collab_model]


    def pad_to_match(self,A,B,dim):
        '''
        Returns copies of A and B that are zero padded so that they have the same size in dimension dim. A and B should have the same number of dimensions
        Args:
            A : pytorch tensor
            B : pytorch tensor
            dim : integer specifying the dimension along which to pad. 
        '''    

        initial_pads = [0,0] * (A.ndim - 1 - dim)
        dim_diff = A.size(dim) - B.size(dim)
        #print('padding ',repr(dim_diff))
        if dim_diff > 0:
           B = F.pad(B,initial_pads + [dim_diff//2,dim_diff - dim_diff//2])
        elif dim_diff < 0:
           A = F.pad(A,initial_pads + [-dim_diff//2,-dim_diff - (-dim_diff)//2])
        return A,B

    def calculate_matching_loss(self,src_activations,target_activations,matching_layer,pooling_config):
        '''
        Uses a matching layer to map the src_activations to the target_activations, and calculate the L2 difference. For convolutional (n_dim=4)
        activations, first unpools the src_activations if pooling_config is not None. Then uses zero padding to match the X and Y sizes of 
        src_activations and target_activations
        '''    


        #print('matching {} to {} using {}'.format(src_activations.size(),target_activations.size(),matching_layer))
        if src_activations.ndim == 2 : #Non-convolutional source activation
           reconstruction_error = matching_layer(src_activations) - target_activations.reshape(target_activations.size(0),-1)
        elif src_activations.ndim == 4 : #Convolutional source activation
           assert target_activations.ndim == 4, 'can not match a convolutional source activation to a non-convolutional target activation'
           if pooling_config is not None:
               pooling_indices,kernel_size = pooling_config
               src_activations = F.max_unpool2d(src_activations,pooling_indices,kernel_size)

           src_activations,target_activations = self.pad_to_match(src_activations,target_activations,2)
           src_activations,target_activations = self.pad_to_match(src_activations,target_activations,3)       

           reconstruction_error = matching_layer(src_activations) - target_activations

        matching_loss = (reconstruction_error.view(reconstruction_error.size(0),-1)**2).sum(-1).mean()
        return matching_loss
        

    def get_matching_loss(self):
        return self.total_matching_loss
    
    def forward(self,x):
        '''
        Runs a forward pass on the collaborator model. If the collaborator model is in training mode, also does a forward pass in 
        the aggregator model to calculate the matching loss
        '''    
        self.total_matching_loss = 0
        if self.collab_model[0].training:
            x_aggregator = x_collaborator = x
            matching_idx = 1
            target_activations = x_aggregator
            pooling_config = None
            for idx,(collab_layer,aggregator_layer) in enumerate(zip(self.collab_model,self.aggregator_model)):
                if isinstance(collab_layer,nn.MaxPool2d):
                   #Patch the pooling layer so that it returns the pooling indices
                   orig_return_indices = collab_layer.return_indices
                   collab_layer.return_indices = True
                   x_collaborator,pooling_indices = collab_layer(x_collaborator)
                   collab_layer.return_indices = orig_return_indices
                   pooling_config = (pooling_indices,collab_layer.kernel_size)
                else:
                   x_collaborator = collab_layer(x_collaborator)   

                with torch.no_grad():
                     x_aggregator = aggregator_layer(x_aggregator)

                if idx== self.match_data[matching_idx][0]:
                   self.total_matching_loss += self.calculate_matching_loss(x_collaborator,target_activations,self.matching_layers[matching_idx - 1],pooling_config)
                   target_activations = x_aggregator
                   pooling_config = None
                   matching_idx += 1
            result = x_collaborator
        else:
            for layer in self.collab_model:
                x = layer(x)
            result = x
        return result
