
How to set up Federated Learning on Docker
-------------------------------------------

We will show you how to set up federated learning on Docker
using the simplest MNIST dataset as an example.

Before we start the tutorial, please make sure you have Docker
installed and confugured properly. Here is a easy test to run:

.. code-block:: console

  $ docker run hello-world
  Hello from Docker!
  This message shows that your installation appears to be working correctly.
  ...
  ...
  ...


Federated Training of an MNIST Classifier
-------------------------------------------


Start an Aggregator
^^^^^^^^^^^^^^^^^^^^
1. Enter the project folder and clean the build folder.
Note that "spr_secure_intelligence-trusted_federated_learning"
is the folder name we chose for the local repository.
It can be anything of your choice on your machine.

.. code-block:: console

  $ cd spr_secure_intelligence-trusted_federated_learning
  $ make clean
  rm -r -f venv
  rm -r -f dist
  rm -r -f build
  rm -r -f tfedlrn.egg-info
  rm -r -f bin/federations/certs/test/*

2. Build the docker images "tfl_agg_<model_name>_<username>:0.1" and 
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

  $ make build_containers model_name=mnist_cnn_keras
  --build-arg http_proxy \
  --build-arg https_proxy \
  --build-arg socks_proxy \
  --build-arg ftp_proxy \
  --build-arg no_proxy \
  --build-arg UID=11632344 \
  --build-arg GID=2222 \
  --build-arg UNAME=edwardsb \
  -t tfl_agg_mnist_cnn_keras_edwardsb:0.1 \
  -f Dockerfile \
  .
  Sending build context to Docker daemon  12.95MB
  Step 1/28 : FROM ubuntu:18.04
   ---> 775349758637
  Step 2/28 : LABEL maintainer "Weilin Xu <weilin.xu@intel.com>"
   ---> Using cache
   ---> fae6ee6bdabf

   ...
   ...
   ...
   
   Step 7/7 : RUN pip3 install intel-tensorflow==1.14.0;
   ---> Using cache
   ---> 54ac91a69eb1
  Successfully built 54ac91a69eb1
  Successfully tagged tfl_col_mnist_cnn_keras_edwardsb:0.1

3. Run the aggregator container (entering a bash shell inside the container), 
again using the Makefile.

.. code-block:: console

  $ make run_agg_container model_name=mnist_cnn_keras
  docker run \
  --net=host \
  -it --name=tfl_agg_mnist_cnn_keras_edwardsb \
  --rm \
  -v /home/edwardsb/repositories/gitlab_tfedlearn/bin:/home/edwardsb/tfl/bin:rw \
  -w /home/edwardsb/tfl/bin \
  tfl_agg_mnist_cnn_keras_edwardsb:0.1 \
  bash

4. In this container shell, generate the files for TLS communication.
The folder is initially empty.
We will generate the files using a script (via the makefile).
The details of TLS, see :ref:`tutorial-tls-pki`.

.. code-block:: console

  $ cd ../
  $ make local_certs
  openssl genrsa -out bin/federations/certs/test/local.key 3072
  Generating RSA private key, 3072 bit long modulus (2 primes)
  ...................................................................................................................++++
  ..........................................................++++
  e is 65537 (0x010001)
  openssl req -new -key bin/federations/certs/test/local.key -out bin/federations/certs/test/local.csr -subj /CN=spr-gpu01.jf.intel.com
  Can't load /home/edwardsb/.rnd into RNG
  140391364972992:error:2406F079:random number generator:RAND_load_file:Cannot open file:../crypto/rand/randfile.c:88:Filename=/home/edwardsb/.rnd
  openssl genrsa -out bin/federations/certs/test/ca.key 3072
  Generating RSA private key, 3072 bit long modulus (2 primes)
  ..............................................++++
  ....................++++
  e is 65537 (0x010001)
  openssl req -new -x509 -key bin/federations/certs/test/ca.key -out bin/federations/certs/test/ca.crt -subj "/CN=Trusted Federated Learning Test Cert Authority"
  Can't load /home/edwardsb/.rnd into RNG
  140520576963008:error:2406F079:random number generator:RAND_load_file:Cannot open file:../crypto/rand/randfile.c:88:Filename=/home/edwardsb/.rnd
  openssl x509 -req -in bin/federations/certs/test/local.csr -CA bin/federations/certs/test/ca.crt -CAkey bin/federations/certs/test/ca.key -CAcreateserial -out bin/federations/certs/test/local.crt
  Signature ok
  subject=CN = spr-gpu01.jf.intel.com
  Getting CA Private Key

Navigate back to the bin directory, and see that the relevant files are now present.

.. code-block:: console

  $ cd bin/
  $ ls federations/certs/test/
  ca.crt  ca.key  ca.srl  local.crt  local.csr  local.key



5. Still in the aggregator container shell, run the aggregator, using
a shell script provided in the project.

.. code-block:: console

  $ ./run_mnist_aggregator.sh 
  Loaded logging configuration: logging.yaml
  2020-01-15 23:17:18,143 - tfedlrn.aggregator.aggregatorgrpcserver - DEBUG - Starting aggregator.


Start Collaborators
^^^^^^^^^^^^^^^^^^^^
You should **skip the first two steps** if you are running
the collaborators on the same machine as the aggregator.

1. (**Only if not on the aggregator machine**) Enter the project folder, clean the build folder, 
and build the containers as above.

.. code-block:: console

  $ cd spr_secure_intelligence-trusted_federated_learning
  $ make clean
  $ make build_containers model_name=mnist_cnn_keras


2. (**Only if not on the aggregator machine**) Copy over authentication files. 
Create the directory 'bin/federations/certs/test/' if it does not already exist, 
then copy the files: ca.cert local.cert and local.key 
(from the machine running the aggregator and created during step 4 of 
'Start an Aggregator' above) into this directory. Of course this is not standard 
practice, but is for tutorial purposes only.

.. code-block:: console  

  $ mkdir -p bin/federations/certs/test/
  $ scp <agg machine hostname>:<appropriate dirctory>/\{ca.crt,local.crt,local.key\} bin/federations/certs/test/

3. Run the first collaborator container (entering a bash shell inside the container) 
using the project folder Makefile. Note that we map the local volumes `./models/` 
and `./bin/` to the docker container, and that we set different names for the two 
collaborator containers (hence the argument 'col_num'), though they share the same 
docker image.

.. code-block:: console

  $ make run_col_container model_name=mnist_cnn_keras col_num=0
  docker run \
  --net=host \
  -it --name=tfl_col_mnist_cnn_keras_edwardsb_0 \
  --rm \
  -v /home/edwardsb/repositories/gitlab_tfedlearn/models:/home/edwardsb/tfl/models:ro \
  -v /home/edwardsb/repositories/gitlab_tfedlearn/bin:/home/edwardsb/tfl/bin:rw \
   \
  -w /home/edwardsb/tfl/bin \
  tfl_col_mnist_cnn_keras_edwardsb:0.1 \
  bash 

4. In this first collaborator shell, run the collabotor using the provided shell script.

.. code-block:: console

  $ ./run_mnist_collaborator.sh 0 
  /home/edwardsb/tfl/venv/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:516: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])

  ...
  ...
  ...

  Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
  11493376/11490434 [==============================] - 0s 0us/step
  Loaded logging configuration: logging.yaml

  ...
  ...
  ...

  x_train shape: (6000, 28, 28, 1)
  y_train shape: (6000,)
  6000 train samples
  1000 test samples

  ...
  ...
  ...

  Training set size: 6000; Validation set size: 1000
  2020-01-24 19:19:40,684 - tfedlrn.collaborator.collaboratorgpcclient - DEBUG - Connecting to gRPC at spr-gpu01.jf.intel.com:8844
  2020-01-24 19:19:40,684 - tfedlrn.collaborator.collaborator - INFO - Collaborator [col_0] connects to federation [fl_mnist_conv2fc2] and aggegator [agg_mnist].
  2020-01-24 19:19:40 spr-gpu01 tfedlrn.collaborator.collaborator[18] INFO Collaborator [col_0] connects to federation [fl_mnist_conv2fc2] and aggegator [agg_mnist].
  2020-01-24 19:19:40,685 - tfedlrn.collaborator.collaborator - DEBUG - The optimizer variable treatment is [OptTreatment.RESET].
  2020-01-24 19:19:40,747 - tfedlrn.collaborator.collaborator - DEBUG - Got a job JOB_DOWNLOAD_MODEL
  2020-01-24 19:19:40,761 - tfedlrn.collaborator.collaborator - INFO - Completed the model downloading job.

  ...
  ...
  ...

5. In a second shell on the same machine that you ran the first collaborator container, run 
the second collaborator container (entering a bash shell inside the container). Note that the
two collaborator containers can run on separate machines as well, all that is needed is to 
build the containers on the new machine and copy over the authentication files as
was done above.

.. code-block:: console

  $ make run_col_container model_name=mnist_cnn_keras col_num=1
  docker run \
  --net=host \
  -it --name=tfl_col_mnist_cnn_keras_edwardsb_1 \
  --rm \
  -v /home/edwardsb/repositories/gitlab_tfedlearn/models:/home/edwardsb/tfl/models:ro \
  -v /home/edwardsb/repositories/gitlab_tfedlearn/bin:/home/edwardsb/tfl/bin:rw \
   \
  -w /home/edwardsb/tfl/bin \
  tfl_col_mnist_cnn_keras_edwardsb:0.1 \
  bash


6. In the second collaborator container shell, run the second collaborator.

.. code-block:: console

  $ ./run_mnist_collaborator.sh 1 

  ...
  ...
  ...



Understand federated learning using Tensorboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The aggregator collects performace readings from the
collaborators and the federation, and outputs to
Tensorboard checkpoints. You can start a separate Tensorboard
program from the project folder to visualize the learning process.

.. code-block:: console

  $ tensorboard --logdir ./bin/logs

Federated Training of the BraTS 2D UNet (Brain Tumor Segmentation)
-----------------------------------------------------------------

This tutorial assumes that you've run the MNIST example above in that less details are provided.

BraTS Federation with One Collaborator
----------------------------------------

We'll start the tutorial by training with a single collaborator. Then, we'll edit the FLPlan to include more collaborators and run multiple.

Start an Aggregator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. (**On the aggregator machine**) Build the brats aggregator and collaborator containers. 

.. code-block:: console

  $ make build_containers model_name=brats_2dunet_tensorflow

2. Run the aggregator container, then inside the shell create the files for TLS and run the aggregator.

.. code-block:: console

  $ make run_agg_container model_name=brats_2dunet_tensorflow

(inside the aggregator container shell)

.. code-block:: console

  $ cd ../
  $ make local_certs
  $ cd bin/
  $ ./run_brats_aggregator.sh

Start Collaborator
^^^^^^^^^^^^^^^^^^^^

3. Create the symlinks for the per-institution datasets. 

We host the entire brats 17 dataset on a single volume that the collaborators can all reach and 
provide directories with symlinks for each insitution, such that each institution then only sees its own data.
To create these symlinks, we provide a simple script in bin/create_brats_symlinks.py. It takes two parameters, one
for the path to the brats17 HGG data, and another for the symlinks path to create the institutional subdirs
in. The command is then:

.. code-block:: console

  $ bin/create_brats_symlinks.py -s=<symlink_path> -b=<brats_hgg_path>

So in our case, the command is:

.. code-block:: console

  $ bin/create_brats_symlinks.py -s= '/raid/datasets/BraTS17/symlinks/' -b='/raid/datasets/BraTS17/MICCAI_BraTS17_Data_Training/HGG/'

Note: to remove the links, we recommend using find <symlink_path> -type l -exec unlink {} \; to avoid deleting the actual files.

4. (**On a collaborator machine**) Run the collaborator container (entering a bash shell inside the container).

.. code-block:: console

  $ make run_col_container model_name=brats_2dunet_tensorflow col_num=0
  
  
5. (**On a collaborator machine**) Run the collaborator inside the collaborator container.

.. code-block:: console

  $ ./run_brats_collaborator.sh 0

The model will now train with a single insitution. To stop the training, CTRL-C on each process will suffice.

BraTS Federation with Two or More Collaborators
--------------------------------------------

6. (**On the aggregator machine**) Edit the FLPlan to run with up to 10 collaborators. In bin/federations/plans/brats17_a.yaml, you'll change the "collaborators" value in the "aggregator" block:

.. code-block:: console
  aggregator:
    ...
    collaborators  : 1

becomes

.. code-block:: console
  aggregator:
    ...
    collaborators  : 10

(or less than 10).

Note: Typically, you would want to change the FLPlan file on each machine, but it isn't strictly necessary, since the collaborators will ignore that value anyway. Eventually, the collaborators and aggregators will all kepe their files in sync via the Governor.


Start the Aggregator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


7. (**On the aggregator machine**) Run the aggregator container, then inside the shell run the aggregator.

.. code-block:: console

  $ make run_agg_container model_name=brats_2dunet_tensorflow

(inside the aggregator container shell)

.. code-block:: console

  $ ./run_brats_aggregator.sh

Start the Collaborators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have to repeat our earlier steps for each collaborator:

8. (**On each new collaborator machine**) Build the brats containers, as before.

9. (**On each new collaborator machine**) Copy the certs over, as before. (**This is incorrect for use over an unsecured network! Real cases require unique certs!!!**)

10. (**For each collaborator**) On the given collaborator machine, run the collaborator conainer and run the collaborator inside the container shell(replacing #### with the collaborator number, starting with 0). 

.. code-block:: console

  $ make run_col_container model_name=brats_2dunet_tensorflow col_num=####
  
  
(inside the collaborator container shell)

.. code-block:: console

  $ ./run_brats_collaborator.sh ####
  
  
  
  


