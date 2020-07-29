.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.


How to set up Federated Learning on Docker
-------------------------------------------

We will show you how to set up federated learning on Docker
using the simplest MNIST dataset as an example.

Before we start the tutorial, please make sure you have Docker
installed and confugured properly. Here is a easy test to run in order to test some basic functionality:

.. code-block:: console

  $ docker run hello-world
  Hello from Docker!
  This message shows that your installation appears to be working correctly.
  ...
  ...
  ...


Federated Training of an MNIST Classifier
-------------------------------------------

Configure The Federation
^^^^^^^^^^^^^^^^^^^^^^^^

(**Every machine needs configured. We recommend creating/editing config files on a single machine,
then copying files to the others, as indicated in the table at the end. Eventually, the Governor
will handle most of this**)

1. Enter the project folder.
Note that "spr_secure_intelligence-trusted_federated_learning"
is the folder name we chose for the local repository.
It can be anything of your choice on your machine.

.. code-block:: console

  $ cd spr_secure_intelligence-trusted_federated_learning

2. Edit the FL plan file to specify the correct addresses for your machines.
Open bin/federations/plans/keras_cnn_mnist_2.yaml:

.. code-block:: console

  $ vi bin/federations/plans/keras_cnn_mnist_2.yaml


Find the keys in the federation config for the address ("agg_addr") and port ("agg_port"):

.. code-block:: console

  ...
  federation:
    fed_id: &fed_id 'fed_0'
    opt_treatment: &opt_treatment 'RESET'
    polling_interval: &polling_interval 4
    rounds_to_train: &rounds_to_train 16
    agg_id: &agg_id 'agg_0'
    agg_addr: &agg_addr "agg.domain.com"   # CHANGE THIS STRING
    agg_port: &agg_port <some_port>        # CHANGE THIS INT
...


Next find the hostnames key and set the machine names for each collaborator.
You may use the same machine as the aggregator. Any collaborator not explicitly set
will use the "__DEFAULT_HOSTNAME__".

.. code-block:: console

  ...
  hostnames:
    __DEFAULT_HOSTNAME__: defaultcol.doman.com  # CHANGE THIS
    col_0: col0.doman.com                       # CHANGE THIS
    col_1: col1.doman.com                       # CHANGE THIS
  ...


3. Build the pki using our plan file. For some pki details, see :ref:`tutorial-tls` (requires pyyaml be installed). 


.. code-block:: console

  $ bin/create_pki_for_flplan.py -p keras_cnn_mnist_2.yaml


  Generating RSA private key, 3072 bit long modulus (2 primes)
  created /home/msheller/git/tfl_upenn/bin/federations/certs/test/ca.key
  created /home/msheller/git/tfl_upenn/bin/federations/certs/test/ca.crt
  Generating RSA private key, 3072 bit long modulus (2 primes)
  created /home/msheller/git/tfl_upenn/bin/federations/certs/test/agg_0.key
  ...


4. Copy files to each machine as needed:

.. list-table:: Files to copy
   :widths: 25 25
   :header-rows: 1

   * - Filename
     - Needed By
   * - ca.crt
     - All
   * - keras_cnn_mnist_2.yaml
     - All
   * - docker_data_config.yaml
     - all collaborators
   * - agg_0.key
     - aggregator machine
   * - col_*.key
     - collaborator machine for col_*
   * - col_*.crt
     - collaborator machine for col_*

Start an Aggregator
^^^^^^^^^^^^^^^^^^^^

1. Build the docker images "tfl_agg_<model_name>_<username>:0.1" and 
"tfl_col_<model_name>_<username>:0.1" using project folder Makefile targets.
This uses the project folder "Dockerfile".
We only build them once, unless we change `Dockerfile`.
We pass along the proxy configuration from the host machine
to the docker container, so that your container would be
able to access the Internet from typical corporate networks.
We also create a container user with the same UID so that it is easier
to access the mapped local volume from the docker container.
Note that we include the username to avoid development-time collisions
on shared develpment servers.
We build the collaborator Docker image upon the aggregator image, 
adding necessary dependencies such as the mainstream deep learning 
frameworks. You may modify `./models/<model_name>/Dockerfile` to install
the needed packages.


.. code-block:: console

  $ make build_containers model_name=keras_cnn
    docker build \
    --build-arg BASE_IMAGE=ubuntu:18.04 \
    --build-arg http_proxy \
    --build-arg https_proxy \
    --build-arg socks_proxy \
    --build-arg ftp_proxy \
    --build-arg no_proxy \
    --build-arg UID=11632344 \
    --build-arg GID=2222 \
    --build-arg UNAME=edwardsb \
    -t tfl_agg_keras_cnn_edwardsb:0.1 \
    -f Dockerfile \
    .
    Sending build context to Docker daemon   3.25GB
    Step 1/29 : ARG BASE_IMAGE=ubuntu:18.04
    Step 2/29 : FROM $BASE_IMAGE
     ---> ccc6e87d482b
    Step 3/29 : LABEL maintainer "Weilin Xu <weilin.xu@intel.com>"
     ---> Using cache
     ---> 7850bfc2c817
    
       ...
       ...
       ...
       
    Step 29/29 : ENV PATH=/home/${UNAME}/tfl/venv/bin:$PATH
     ---> Running in 5d41487d94f4
    Removing intermediate container 5d41487d94f4
     ---> 1e71e09a4a5a
    Successfully built 1e71e09a4a5a
    Successfully tagged tfl_agg_keras_cnn_edwardsb:0.1
    docker build --build-arg whoami=edwardsb \
    --build-arg use_gpu=false \
    -t tfl_col_cpu_keras_cnn_edwardsb:0.1 \
    -f ./models/tensorflow/keras_cnn/cpu.dockerfile \
    .
    Sending build context to Docker daemon  3.251GB
    Step 1/7 : ARG whoami
    
      ...
      ...
      ...
    
    
    
    Step 7/7 : RUN pip3 install intel-tensorflow==1.14.0;
     ---> Using cache
     ---> 7d1b3ef6fb8c
    [Warning] One or more build-args [use_gpu] were not consumed
    Successfully built 7d1b3ef6fb8c
    Successfully tagged tfl_col_cpu_keras_cnn_edwardsb:0.1

2. Run the aggregator container (entering a bash shell inside the container), 
again using the Makefile. Note that we map the local volumes `./bin/federations` to the container

.. code-block:: console

  $ make run_agg_container model_name=keras_cnn
    docker run \
    --net=host \
    -it --name=tfl_agg_keras_cnn_edwardsb \
    --rm \
    -w /home/edwardsb/tfl/bin \
    -v /home/edwardsb/repositories/gitlab_tfedlearn/bin/federations:/home/edwardsb/tfl/bin/federations:rw \
    tfl_agg_keras_cnn_edwardsb:0.1 \
    bash 

3. In the aggregator container shell, build the initial weights files providing the global model initialization 
that will be sent from the aggregator out to all collaborators.

.. code-block:: console

  $ ./create_initial_weights_file_from_flplan.py -p keras_cnn_mnist_2.yaml -dc docker_data_config.yaml

  ...
  ...
  ...

created /home/edwardsb/tfl/bin/federations/weights/keras_cnn_mnist_init.pbuf

4. In the aggregator container shell, run the aggregator, using
a shell script provided in the project.

.. code-block:: console

  $ ./run_mnist_aggregator.sh 
  Loaded logging configuration: logging.yaml
  2020-01-15 23:17:18,143 - tfedlrn.aggregator.aggregatorgrpcserver - DEBUG - Starting aggregator.


Start Collaborators
^^^^^^^^^^^^^^^^^^^^

Note: the collaborator machines can be the same as the aggregator machine.

1. (**On each collaborator machine**) Enter the project folder and build the containers as above.

.. code-block:: console

  $ make build_containers model_name=keras_cnn


2. (**On the first collaborator machine**)
Run the first collaborator container (entering a bash shell inside the container) 
using the project folder Makefile. Note that we map the local volumes `./bin/federations` 
to the docker container, and that we set different names for the two 
collaborator containers (hence the argument 'col_num'), though they share the same 
docker image.

.. code-block:: console

  $ make run_col_container model_name=keras_cnn col_num=0
  docker run \
  ...
  bash 

5. In this first collaborator shell, run the collabotor using the provided shell script.

.. code-block:: console

  $ ./run_mnist_collaborator.sh 0 
  

6. (**On the second collaborator machine, which could be a second terminal on the first machine**)
Run the second collaborator container (entering a bash shell inside the container).

.. code-block:: console

  $ make run_col_container model_name=keras_cnn col_num=1
  docker run \
  ...
  bash


7. In the second collaborator container shell, run the second collaborator.

.. code-block:: console

  $ ./run_mnist_collaborator.sh 1 


Federated Training of the 2D UNet (Brain Tumor Segmentation)
-----------------------------------------------------------------

This tutorial assumes that you've run the MNIST example above in that less details are provided.


1. Unlike the MNIST toy example, in this example we are allocating data correctly. To make this work,
we create a <Brats Symlinks Dir>, which is has directories of symlinks to the data for each institution
number. Setting this up is out-of-scope for this code at the moment, so we leave this to the reader. In
the end, our directory looks like below. Note that "0-9" allows us to do data-sharing training.

.. code-block:: console

  $ ll <Brats Symlinks Dir>

  ...
    drwxr-xr-x  90 <user> <group> 4.0K Nov 25 22:14 0
    drwxr-xr-x 212 <user> <group>  12K Nov  2 16:38 0-9
    drwxr-xr-x  24 <user> <group> 4.0K Nov 25 22:14 1
    drwxr-xr-x  36 <user> <group> 4.0K Nov 25 22:14 2
    drwxr-xr-x  14 <user> <group> 4.0K Nov 25 22:14 3
    drwxr-xr-x  10 <user> <group> 4.0K Nov 25 22:14 4
    drwxr-xr-x   6 <user> <group> 4.0K Nov 25 22:14 5
    drwxr-xr-x  10 <user> <group> 4.0K Nov 25 22:14 6
    drwxr-xr-x  16 <user> <group> 4.0K Nov 25 22:14 7
    drwxr-xr-x  17 <user> <group> 4.0K Nov 25 22:14 8
    drwxr-xr-x   7 <user> <group> 4.0K Nov 25 22:14 9
  ...


2. (**We start with just a two collaborator example.**)
Edit the FL plan file to specify the correct addresses for your machines.
Open bin/federations/plans/brats17_insts2_3.yaml.

.. code-block:: console

  $ vi bin/federations/plans/tf_2dunet_brats_insts2_3.yaml


Find the keys in the federation config for the address ("agg_addr") and port ("agg_port"):

.. code-block:: console

  ...
  federation:
    fed_id: &fed_id 'fed_0'
    opt_treatment: &opt_treatment 'CONTINUE_GLOBAL'
    polling_interval: &polling_interval 4
    rounds_to_train: &rounds_to_train 50
    agg_id: &agg_id 'agg_0'
    agg_addr: &agg_addr "agg.domain.com"   # CHANGE THIS STRING
    agg_port: &agg_port <some_port>        # CHANGE THIS INT
...


Next find the hostnames key and set the machine names for each collaborator.
You may use the same machine as the aggregator. Any collaborator not explicitly set
will use the "__DEFAULT_HOSTNAME__".

.. code-block:: console

  ...
  hostnames:
    __DEFAULT_HOSTNAME__: defaultcol.doman.com  # CHANGE THIS
    col_0: col0.doman.com                       # CHANGE THIS
    col_1: col1.doman.com                       # CHANGE THIS
  ...


3. Build the pki using our plan file. For some pki details, see :ref:`tutorial-tls`. 


.. code-block:: console

  $ bin/create_pki_for_flplan.py -p tf_2dunet_brats_insts2_3.yaml


  Generating RSA private key, 3072 bit long modulus (2 primes)
  created /home/msheller/git/tfl_upenn/bin/federations/certs/test/ca.key
  created /home/msheller/git/tfl_upenn/bin/federations/certs/test/ca.crt
  Generating RSA private key, 3072 bit long modulus (2 primes)
  created /home/msheller/git/tfl_upenn/bin/federations/certs/test/agg_0.key
  ...


4. Edit the docker data config file to refer to the correct username (the name of the account
you are using. Open bin/federations/docker_data_config.yaml and replace the username with your username

.. code-block:: console

  $ vi bin/federations/docker_data_config.yaml



collaborators:
  col_one_big:
    brats: &brats_data_path '/home/<USERNAME>/tfl/datasets/brats'                # replace with your username
  col_0:
    brats: *brats_data_path   
    mnist_shard: 0
  col_1:
    brats: *brats_data_path
    mnist_shard: 1
...


5. Copy files to each machine as needed:

.. list-table:: Files to copy
   :widths: 25 25
   :header-rows: 1

   * - Filename
     - Needed By
   * - ca.crt
     - All
   * - tf_2dunet_brats_insts2_3.yaml
     - All
   * - docker_data_config.yaml
     - all collaborators
   * - agg_0.key
     - aggregator machine
   * - col_*.key
     - collaborator machine for col_*
   * - col_*.crt
     - collaborator machine for col_*

Start an Aggregator
^^^^^^^^^^^^^^^^^^^^

1. Build the docker images "tfl_agg_<model_name>_<username>:0.1" and 
"tfl_col_<model_name>_<username>:0.1" using project folder Makefile targets.
This uses the project folder "Dockerfile".
We only build them once, unless we change `Dockerfile`.
We pass along the proxy configuration from the host machine
to the docker container, so that your container would be
able to access the Internet from typical corporate networks.
We also create a container user with the same UID so that it is easier
to access the mapped local volume from the docker container.
Note that we include the username to avoid development-time collisions
on shared develpment servers.
We build the collaborator Docker image upon the aggregator image, 
adding necessary dependencies such as the mainstream deep learning 
frameworks. You may modify `./models/<model_name>/Dockerfile` to install
the needed packages.


.. code-block:: console

  $ make build_containers model_name=tf_2dunet
 

2. Run the aggregator container (entering a bash shell inside the container), 
again using the Makefile. Note that we map the local volumes `./bin/federations` to the container

.. code-block:: console

  $ make run_agg_container model_name=tf_2dunet dataset=brats

3. In the aggregator container shell, build the initial weights files providing the global model initialization 
that will be sent from the aggregator out to all collaborators.

.. code-block:: console

  $ ./create_initial_weights_file_from_flplan.py -p tf_2dunet_brats_insts2_3.yaml -dc docker_data_config.yaml



4. In the aggregator container shell, run the aggregator, using
a shell script provided in the project.

.. code-block:: console

  $ ./run_brats_aggregator.sh 
  Loaded logging configuration: logging.yaml
  2020-01-15 23:17:18,143 - tfedlrn.aggregator.aggregatorgrpcserver - DEBUG - Starting aggregator.


Start Collaborators
^^^^^^^^^^^^^^^^^^^^

Note: the collaborator machines can be the same as the aggregator machine.

1. (**On each collaborator machine**) Enter the project folder and build the containers as above.

.. code-block:: console

  $ make build_containers model_name=tf_2dunet


2. (**On the first collaborator machine**)
Run the first collaborator container. Note we are using collaborators 2 and 3.

.. code-block:: console

  $ make run_col_container model_name=tf_2dunet dataset=brats col_num=2

5. In this first collaborator shell, run the collabotor using the provided shell script.

.. code-block:: console

  $ ./run_brats_collaborator.sh 2 

6. (**On the second collaborator machine, which could be a second terminal on the first machine**)
Run the second collaborator container (entering a bash shell inside the container).

.. code-block:: console

  $ make run_col_container model_name=tf_2dunet dataset=brats col_num=3
  docker run \
  ...
  bash


7. In the second collaborator container shell, run the second collaborator.

.. code-block:: console

  $ ./run_brats_collaborator.sh 3

  ...
  ...
  ...


