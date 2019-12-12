
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
  rm -r -f venv
  rm -r -f dist
  rm -r -f build
  rm -r -f tfedlrn.egg-info
  rm -r -f bin/federations/certs/test/*

2. Build a docker image "tfl_agg:0.1" from "Dockerfile".
We only build it once unless we change `Dockerfile`.
We pass along the proxy configuration from the host machine
to the docker container, so that your container would be
able to access the Internet from typical corporate networks.
We also create a user with the same UID so that it is easier
to access the mapped local volume from the docker container.


.. code-block:: console

  $ docker build \
  --build-arg http_proxy \
  --build-arg https_proxy \
  --build-arg socks_proxy \
  --build-arg ftp_proxy \
  --build-arg no_proxy \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg UNAME=$(whoami) \
  -t tfl_agg:0.1 \
  -f Dockerfile \
  .

  Sending build context to Docker daemon  14.89MB
  Step 1/27 : FROM ubuntu:18.04
  ---> 775349758637
  Step 2/27 : LABEL maintainer "Weilin Xu <weilin.xu@intel.com>"
  ---> fae6ee6bdabf
  ...
  ...
  ...
  Successfully installed coloredlogs-10.0 grpcio-1.25.0 humanfriendly-4.18 nibabel-2.5.1 numpy-1.17.4 protobuf-3.11.1 pyyaml-5.2 six-1.13.0 tensorboardX-1.9 tfedlrn-0.0.0
  Removing intermediate container 787283c5807f
  ---> c3f4a70f4117
  Successfully built c3f4a70f4117
  Successfully tagged tfl_agg:0.1
  $

3. Create several aliases to simplify the docker usage.
First, we create an alias to run the docker container.
We map the local volume `./bin/` to the docker container.

.. code-block:: console

  $ alias tfl-agg-docker='docker run \
  --net=host \
  -it --name=tfl_agg \
  --rm \
  -v "$PWD"/bin:/home/$(whoami)/tfl/bin:rw \
  -w /home/$(whoami)/tfl/bin \
  tfl_agg:0.1'

Second, we create an alias to make the certificates that are required by TLS.

.. code-block:: console

  $ alias tfl-make-local-certs='tfl-agg-docker bash -c "cd ..; make local_certs"'

Third, we create an alias to run aggregators.

.. code-block:: console

  $ alias tfl-aggregator='tfl-agg-docker \
  ../venv/bin/python3 run_aggregator_from_flplan.py'


4. Generate the certificates for TLS communication.
The folder of certificates is initially empty.
We will generate the certificates using a script.
The details of TLS, see :ref:`tutorial-tls-pki`.

.. code-block:: console

  $ ls bin/federations/certs/test/
  $
  $ tfl-make-local-certs
  openssl genrsa -out bin/federations/certs/test/local.key 3072
  Generating RSA private key, 3072 bit long modulus (2 primes)
  ............++++
  ..++++
  e is 65537 (0x010001)
  openssl req -new -key bin/federations/certs/test/local.key -out bin/federations/certs/test/local.csr -subj /CN=spr-gpu02.jf.intel.com
  Can't load /home/weilinxu/.rnd into RNG
  140265634959808:error:2406F079:random number generator:RAND_load_file:Cannot open file:../crypto/rand/randfile.c:88:Filename=/home/weilinxu/.rnd
  openssl genrsa -out bin/federations/certs/test/ca.key 3072
  Generating RSA private key, 3072 bit long modulus (2 primes)
  ..........................................................................................................................................++++
  ....................++++
  e is 65537 (0x010001)
  openssl req -new -x509 -key bin/federations/certs/test/ca.key -out bin/federations/certs/test/ca.crt -subj "/CN=Trusted Federated Learning Test Cert Authority"
  Can't load /home/weilinxu/.rnd into RNG
  140015244689856:error:2406F079:random number generator:RAND_load_file:Cannot open file:../crypto/rand/randfile.c:88:Filename=/home/weilinxu/.rnd
  openssl x509 -req -in bin/federations/certs/test/local.csr -CA bin/federations/certs/test/ca.crt -CAkey bin/federations/certs/test/ca.key -CAcreateserial -out bin/federations/certs/test/local.crt
  Signature ok
  subject=CN = spr-gpu02.jf.intel.com
  Getting CA Private Key
  $ ls bin/federations/certs/test/
  ca.crt  ca.key  ca.srl  local.crt  local.csr  local.key


5. Start an aggregator.
(TODO: We need to print some information about starting an aggregator.)

.. code-block:: console

  $ tfl-aggregator -p mnist_a.yaml
  Loaded logging configuration: logging.yaml

In case anytime you need to examine the docker container
with a shell, just type

.. code-block:: console

  $ tfl-agg-docker bash


Start Collaborators
^^^^^^^^^^^^^^^^^^^^

We build the Docker image for collaborators upon the
aggregator image, adding necessary dependencies such as
the mainstream deep learning frameworks.
You may modify `./models/<model_name>/Dockerfile` to install
the needed packages.

You should **skip the first two steps** if you are building
the collaborator image on the same machine as the aggregator.

1. (Optional) Enter the project folder and clean the build folder.

.. code-block:: console

  $ cd spr_secure_intelligence-trusted_federated_learning
  $ make clean


2. (Optional) Build the aggregator image, which is the parent of the
collaborator image (`Dockerfile.agg`).

.. code-block:: console

  $ docker build \
  --build-arg http_proxy \
  --build-arg https_proxy \
  --build-arg socks_proxy \
  --build-arg ftp_proxy \
  --build-arg no_proxy \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  --build-arg UNAME=$(whoami) \
  -t tfl_agg:0.1 \
  -f Dockerfile \
  .


3. Build a docker image from `Dockerfile` provided by the model.
We only build it once unless we change `Dockerfile` or the base image.

.. code-block:: console

  $ docker build \
  -t tfl_col:0.1 \
  -f ./models/mnist_cnn_keras/Dockerfile \
  .


4. Create alias to run the docker container.
We set a different name for different collaborators,
while they share the same docker image.
We map the local volumes `./models/` and `./bin/` to the docker container.

.. code-block:: console

  $ alias tfl-docker-col0='docker run \
  --net=host \
  -it --name=tfl_col_0 \
  --rm \
  -v "$PWD"/models:/home/$(whoami)/tfl/models:ro \
  -v "$PWD"/bin:/home/$(whoami)/tfl/bin:rw \
  -w /home/$(whoami)/tfl/bin \
  tfl_col:0.1'

  $ alias tfl-docker-col1='docker run \
  --net=host \
  -it --name=tfl_col_1 \
  --rm \
  -v "$PWD"/models:/home/$(whoami)/tfl/models:ro \
  -v "$PWD"/bin:/home/$(whoami)/tfl/bin:rw \
  -w /home/$(whoami)/tfl/bin \
  tfl_col:0.1'

5. Set up TLS certificates.
Copy the CA certificate and the local certificates signed by the CA.
We just let the two collaborators share the same private key for simplicity
in this example. But it will be very WRONG in practice.

.. code-block:: console

  $ copy ca.crt local.crt local.key bin/certs/test/

6. Start collaborators.
A collaborator needs to prepare a dataset that meets the requirement
of a federated learning plan.
As an example, we perform dataset preparation and start the collaborator
in one line of command:

.. code-block:: console

  $ tfl-docker-col0 bash -c "mkdir -p ../datasets/mnist_batch; \
  ../venv/bin/python3 \
  ../models/mnist_cnn_keras/prepare_dataset.py \
  -ts=0 \
  -te=6000 \
  -vs=0 \
  -ve=1000 \
  --output_path=../datasets/mnist_batch/mnist_batch.npz; \
  ../venv/bin/python3 run_collaborator_from_flplan.py -p mnist_a.yaml -col col_0;"

  $ tfl-docker-col1 bash -c "mkdir -p ../datasets/mnist_batch; \
  ../venv/bin/python3 \
  ../models/mnist_cnn_keras/prepare_dataset.py \
  -ts=6000 \
  -te=12000 \
  -vs=1000 \
  -ve=2000 \
  --output_path=../datasets/mnist_batch/mnist_batch.npz; \
  ../venv/bin/python3 run_collaborator_from_flplan.py -p mnist_a.yaml -col col_1;"


In case anytime you need to examine the docker container
with a shell, just type

.. code-block:: console

  $ tfl-docker-col0 bash
  $ tfl-docker-col1 bash


Understand federated learning using Tensorboard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The aggregator collects performace readings from the
collaborators and the federation, and outputs to
Tensorboard checkpoints. You can start a separate Tensorboard
program to visualize the learning process.

.. code-block:: console

  $ tensorboard --logdir ./federation/logs

