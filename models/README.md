# Models for Federated Learning

The folder contains runnable code for different models (and a generic test module to validate the code) in the FL framework. 

The collaborators will download the code to join federated learning. 


## Interface
The current example is a class with these methods:
> TODO: we should implement a base class for all models

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

