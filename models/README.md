# Copyright (C) 2020 Intel Corporation
# Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.

# Models for Federated Learning

The folder contains runnable code for different models (and a generic test module to validate the code) in the FL framework. 

As our tool develops, the collaborators will eventually download the code to join federated learning. 


## Interface
The current examples are classes, whose constructors take a data object along with other key word arguments. In order to run federations (or single process simulations), the model class should have the methods:
> TODO: we should implement a base class for all models

* get_data()
* set_data(data_object)
* train_epoch()
* get_training_data_size()
* validate()
* get_validation_data_size()
* get_tensor_dict(Boolean with_opt_vars)
* set_tensor_dict(tensor_dict)
* reset_opt_vars()

We may also add this for the aggregator:

* export_initial_weights(fpath)


And for local test:
* load_weights(weight_fpath)
    to use the trained weights.


## Proposal of Revised Interface

* train_epoch() --> train(iterations=5)
> so that we can support flexible training iterations.
* validate() returns a dictionary of {metric: value} so that we can report and aggregate multiple metircs.


## TODO

* multiple metrics to calculate in validate()
* get_*_data_size(): 
* validate() returns dict()

