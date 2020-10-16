.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _running_baremetal:

Creating the Federation
#######################

TL;DR
~~~~~

Here's the :download:`"Hello Federation" bash script <../tests/gitlab/test_hello_federation.sh>` used for testing the project pipeline.

.. literalinclude:: ../tests/gitlab/test_hello_federation.sh
  :language: bash


Hello Federation - Your First Federated Learning Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will show you how to set up |productName|. 
Before you run the federation make sure you have installed |productName| 
:ref:`using these instructions <install_initial_steps>` on every node (i.e. aggregator and collaborators).

.. _creating_workspaces:

On the Aggregator
~~~~~~~~~~~~~~~~~

1. Make sure you have initialized the virtual environment and can run the :code:`fx` command.

2. Choose a workspace template:

    - :code:`keras_cnn_mnist`: workspace with a simple `Keras <http://keras.io/>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.
    - :code:`tf_2dunet`: workspace with a simple `TensorFlow <http://tensorflow.org>`_ CNN model that will use the `BraTS <https://www.med.upenn.edu/sbia/brats2017/data.html>`_ dataset and train in a federation.
    - :code:`tf_cnn_histology`: workspace with a simple `TensorFlow <http://tensorflow.org>`_ CNN model that will download the `Colorectal Histology <https://zenodo.org/record/53169#.XGZemKwzbmG>`_ dataset and train in a federation.
    - :code:`torch_cnn_histology`: workspace with a simple `PyTorch <http://pytorch.org/>`_ CNN model that will download the `Colorectal Histology <https://zenodo.org/record/53169#.XGZemKwzbmG>`_ dataset and train in a federation.
    - :code:`torch_cnn_mnist`: workspace with a simple `PyTorch <http://pytorch.org>`_ CNN model that will download the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset and train in a federation.

3. For example, we will use :code:`keras_cnn_mnist`:
    .. code-block:: console
        $ export WORKSPACE_TEMPLATE = keras_cnn_mnist
    
2. Create a workspace for the new federation project.

    .. code-block:: console
    
       $ fx workspace create --prefix WORKSPACE.PATH --template ${WORKSPACE_TEMPLATE}
       
    where **WORKSPACE.PATH** is the directory to create your workspace.  A list of
    pre-created templates can be found by simply running the command:

    .. code-block:: console
    
       $ fx workspace create --prefix WORKSPACE.PATH 
       
    .. note::
    
    Existing models can either be copied into the :code:`code` subdirectory
    in the workspace.

3. Change to the workspace directory.

    .. code-block:: console
    
        $ cd WORKSPACE.PATH
     
        
4.  Although it is possible to train models from scratch, it is assumed that in many cases the federation may perform fine-tuning of a previously-trained model. For this reason, the pre-trained weights for the model will be stored within protobuf files on the aggregator and passed to the collaborators during initialization. As seen in the YAML file, the protobuf file with the initial weights is expected to be found in the file **${WORKSPACE_TEMPLATE}_init.pbuf**. For this example, however, we’ll just create an initial set of random model weights and putting it into that file by running the command:

    .. code-block:: console
    
       $ fx plan initialize -a AFQDN

   where *AFQDN** is the actual `fully qualified domain name (FQDN) <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`_ of the aggregator node. If you omit this parameter, :code:`fx` will automatically use the FQDN of the current node assuming the node has been correctly set with a static address. 
   .. note::

   Each workspace may have multiple Federated Learning plans and multiple collaborator lists associated with it.
   Therefore, the Aggregator has the following optional parameters.

   +-------------------------+---------------------------------------------------------+
   | Optional Parameters     | Description                                             |
   +=========================+=========================================================+
   | -p, --plan_config PATH  | Federated Learning plan [default = plan/plan.yaml]      |
   +-------------------------+---------------------------------------------------------+
   | -c, --cols_config PATH  | Authorized collaborator list [default = plan/cols.yaml] |
   +-------------------------+---------------------------------------------------------+
   | -d, --data_config PATH  | The data set/shard configuration file                   |
   +-------------------------+---------------------------------------------------------+    
