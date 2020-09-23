.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

***********************
Starting the Federation
***********************

On the Aggregator
~~~~~~~~~~~~~~~~~

#.	Now we’re ready to start the aggregator by running the Python script. 

    .. code-block:: console
    
       $ fx aggregator start

    At this point, the aggregator is running and waiting
    for the collaborators to connect. When all of the collaborators
    connect, the aggregator starts training. When the last round of
    training is complete, the aggregator stores the final weights in
    the protobuf file that was specified in the YAML file
    (in this case *save/keras_cnn_mnist_latest.pbuf*).

.. _running_collaborators:

On the Collaborator
~~~~~~~~~~~~~~~~~~~

1.	Open a new terminal, change the directory to the workspace, and activate the virtual environment.

2.	Now run the collaborator that was labelled *one* using the :code:`fx` command.

    .. code-block:: console

       $ fx collaborator start -n one

    .. note::

       Each workspace may have multiple Federated Learning plans and multiple collaborator lists associated with it.
       Therefore, the Collaborator has the following optional parameters.
       
           +-------------------------+---------------------------------------------------------+
           | Optional Parameters     | Description                                             |
           +=========================+=========================================================+
           | -p, --plan_config PATH  | Federated Learning plan [default = plan/plan.yaml]      |
           +-------------------------+---------------------------------------------------------+
           | -d, --data_config PATH  | The data set/shard configuration file                   |
           +-------------------------+---------------------------------------------------------+

3.	Repeat this for each collaborator in the federation. Once all collaborators have joined,  the aggregator will start and you will see log messages describing the progress of the federated training.

On Remote Collaborators
~~~~~~~~~~~~~~~~~~~~~~~

1. First you'll need to export the workspace and copy it to the remote collaborator. On any node that contains the workspace (e.g. on the Aggregator), run the following command:

    .. code-block:: console

       $ fx workspace export --include_certificates

    .. warning::
       The :code:`--include_certificates` should not be used other than for demonstration purposes
       as it includes the certificates for all nodes of the federation. Instead, users should
       manually copy of the necessary certificates and keys to protect the security of the PKI.
       We include the parameter in this demo simply to make it easier for demonstration purposes.

2. If the :code:`export` command is successful, you will have a :code:`.zip` file with the workspace name in the directory. Copy this archive to the remote collaborator node.

3. Make sure you have installed |productName| on the remote collaborator node :ref:`using these instructions <install_initial_steps>` and have activated the virtual environment.

4. Run the command to import the workspace archive:

    .. code-block:: console
    
       $ fx workspace import --file WORKSPACE.zip
       
    This command will unzip the workspace archive to the current directory and install the Python dependencies that were 
    recorded on the original workspace (via :code:`requirements.txt`).
    
    .. warning::
       Make sure you are within a Python virtual environment when importing the workspace as the :code:`requirements.txt` file will overwrite your Python packages.

5. Change the current directory to the new workspace directory. This node can now be used as a collaborator in the federation as described :ref:`previously <running_collaborators>`.


