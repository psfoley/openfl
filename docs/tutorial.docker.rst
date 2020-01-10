
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


Start an Aggregator
^^^^^^^^^^^^^^^^^^^^
1. Enter the project folder and clean the build folder.
Note that "spr_secure_intelligence-trusted_federated_learning"
is the folder name we chose for the local repository.
It can be anything of your choice on your machine.

.. code-block:: console

  $ cd spr_secure_intelligence-trusted_federated_learning
  $ make clean


2. Build the docker images "tfl_agg_<username>:0.1" and 
"tfl_col_<username>:0.1" using project folder Makefile targets.
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

  $ make build_containers


3. Run the aggregator container (entering a bash shell inside the container), 
again using the Makefile.

.. code-block:: console

  $ make run_agg_container


4. In this container shell, generate the files for TLS communication.
The folder is initially empty.
We will generate the files using a script (via another makefile).
The details of TLS, see :ref:`tutorial-tls-pki`.

.. code-block:: console

  $ cd ../
  $ make local_certs


The files should now be present.

.. code-block:: console

  $ cd bin/
  $ ls federations/certs/test/
  ca.crt  ca.key  ca.srl  local.crt  local.csr  local.key



5. Still in the aggregator container shell, start the aggregator, using
a shell script provided in the project.

.. code-block:: console

  $ chmod +x start_mnist_aggregator.sh
  $ ./start_mnist_aggregator.sh 
  


Start Collaborators
^^^^^^^^^^^^^^^^^^^^
You should **skip the first two steps** if you are running
the collaborators on the same machine as the aggregator.

1. (**Only if not on the aggregator machine**) Enter the project folder, clean the build folder, 
and build the containers as above.

.. code-block:: console

  $ cd spr_secure_intelligence-trusted_federated_learning
  $ make clean
  $ make build_containers


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

  $ make run_col_container col_num=0


4. In this first collaborator shell, start the collabotor using the provided shell script.

.. code-block:: console

  $ chmod +x start_mnist_collaborator.sh
  $ ./start_mnist_collaborator.sh 0 


5. In a second shell on the same machine that you ran the first collaborator, run 
the second collaborator (entering a bash shell inside the container). Note that the
two collaborators can run on separate machines as well, all that is needed is to 
build the containers on the new machine and copy over the authentication files as
was done above.

.. code-block:: console

  $ make run_col_container col_num=1


6. In the second collaborator container shell, start the second collaborator.

.. code-block:: console

  $ ./start_mnist_collaborator.sh 1 


Understand federated learning using Tensorboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The aggregator collects performace readings from the
collaborators and the federation, and outputs to
Tensorboard checkpoints. You can start a separate Tensorboard
program from the project folder to visualize the learning process.

.. code-block:: console

  $ tensorboard --logdir ./bin/logs

Running the BraTS 2D UNet
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(**This tutorial assumes that you've run the MNIST example above.**)

1. Start an aggregator. 

.. code-block:: console

  $ tfl-agg-docker python3 run_aggregator_from_flplan.py -p brats17_a.yaml
  Loaded logging configuration: logging.yaml


1. Create the symlinks for the per-institution datasets. 

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

2. (**On each collaborator machine**) Create the collaborator image that includes the 2d unet:

.. code-block:: console

  $ docker build --build-arg whoami=$(whoami) \
  -t tfl_unet_col_$(whoami):0.1 \
  -f ./models/brats_2dunet_tensorflow/Dockerfile \
  .

3. (**On each collaborator machine**) Create the alias for the specific collaborator. Replace 'col0' with 'col1', 'col2', etc... as appropriate.
Also, replace 'symlinks/0' with 'symlinks/1', 'symlinks/2', etc... as appropriate.

.. code-block:: console

  $ alias tfl-docker-col0='docker run \
  --net=host \
  -it --name=tfl_$(whoami)_col_0 \
  --rm \
  -v "$PWD"/models:/home/$(whoami)/tfl/models:ro \
  -v "$PWD"/bin:/home/$(whoami)/tfl/bin:rw \
  -v "/raid/datasets/BraTS17/symlinks/0":/home/$(whoami)/tfl/datasets/brats:ro \
  -v "/raid/datasets/BraTS17/MICCAI_BraTS17_Data_Training/HGG":/raid/datasets/BraTS17/MICCAI_BraTS17_Data_Training/HGG:ro \
  -w /home/$(whoami)/tfl/bin \
  tfl_unet_col_$(whoami):0.1'

4. (**On each collaborator machine**) Run the collaborator, once again replacing 'col0' with 'col1', 'col2', 'col3' as appropriate.

.. code-block:: console

  $ tfl-docker-col0 bash -c "../venv/bin/python3 run_collaborator_from_flplan.py -p brats17_a.yaml -col col_0;"
