
Docker Examples
----------------



Aggregator
^^^^^^^^^^^^
1. Enter the project folder and clean the build folder

.. code-block:: console

    $ cd spr_secure_intelligence-trusted_federated_learning
    $ make clean

2. Build a docker image "tfl_agg:0.1" from "Dockerfile".
We only build it once unless we change `Dockerfile`.
We create a user with the same UID so that it is easier
to access local volume from the docker container.

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

.. code-block:: console

  $ tfl-make-local-certs


5. Start an aggregator.

.. code-block:: console

  $ tfl-aggregator -p mnist_a.yaml


In case anytime you need to examine the docker container
with a shell, just type

.. code-block:: console

  $ tfl-agg-docker bash


Collaborators
^^^^^^^^^^^^^

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


3. Build a docker image from `Dockerfile`.
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