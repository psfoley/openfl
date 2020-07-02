.. raw:: html

   <div class="WordSection1">

 

Intel® Federated Learning

 

 

A screenshot of a cell phone Description automatically generated

*Secure, Privacy-Preserving Machine Learning*

 

 

 

 

**Technical Manual
**

 

Table of Contents

`Overview..3 <#_Toc43273078>`__

`What is Federated Learning?.3 <#_Toc43273079>`__

`How can Intel® SGX Protect Federated Learning?.3 <#_Toc43273080>`__

`What is Intel® Federated Learning?.5 <#_Toc43273081>`__

`Installing and Running the Software.7 <#_Toc43273082>`__

`Accessing the Source Code Repository.7 <#_Toc43273083>`__

`Design Philosophy.7 <#_Toc43273084>`__

`Configure The Federation.7 <#_Toc43273085>`__

`Baremetal Installation.11 <#_Toc43273086>`__

`Installation Steps.11 <#_Toc43273087>`__

`Running the Baremetal Demo – “Hello Federation”.13 <#_Toc43273088>`__

`Steps to Run the Baremetal Federation.13 <#_Toc43273089>`__

`Docker Installation.16 <#_Toc43273090>`__

`Installation Steps.16 <#_Toc43273091>`__

`Running the Docker Demo – “Hello Federation”.18 <#_Toc43273092>`__

`Steps to Run the Docker Federation.18 <#_Toc43273093>`__

`Federated Training of the 2D U-Net (Brain Tumor
Segmentation)21 <#_Toc43273094>`__

`Start an Aggregator.22 <#_Toc43273095>`__

`Start Collaborators.23 <#_Toc43273096>`__

`Porting your Experiments to Intel® Federated
Learning.25 <#_Toc43273097>`__

`Design Philosophy.25 <#_Toc43273098>`__

`Repository Structure.26 <#_Toc43273099>`__

`The Federation Plan.27 <#_Toc43273100>`__

`Models28 <#_Toc43273101>`__

`TensorFlow / Keras / PyTorch.28 <#_Toc43273102>`__

`Data.31 <#_Toc43273103>`__

`Local Data Config.33 <#_Toc43273104>`__

`Running the simulator34 <#_Toc43273105>`__

`Simulated Federated Training of an MNIST Classifier across 10
Collaborators34 <#_Toc43273106>`__

`Create the project virtual environment34 <#_Toc43273107>`__

`Bibliography.37 <#_Toc43273108>`__

 

.. rubric:: Overview
   :name: overview

 

.. rubric:: What is Federated Learning?
   :name: what-is-federated-learning

 

Federated learning is a distributed machine learning approach that
enables organizations to collaborate on machine learning projects
without sharing sensitive data, such as, patient records, financial
data, or classified secrets (McMahan, 2016; Sheller, Reina, Edwards,
Martin, & Bakas, 2019; Yang, Liu, Chen, & Tong, 2019). The basic premise
behind federated learning is that the model moves to meet the data
rather than the data moving to meet the model (Figure 1). Therefore, the
minimum data movement needed across the federation is solely the model
parameters and their updates.

 

A close up of a logo Description automatically generated

Figure 1 Diagram of Federated Learning. The data (yellow, red, and blue
disks) does not leave the original owner (a, b, c). Instead the model
(t) and model updates (t+1,a; t+1,b; t+1,c) are passed and each owner
performs training locally. The parameter server sends the model (t) to
each owner and the aggregator combines the model updates (t+1,a; t+1,b;
t+1,c). The aggregator sends this combined model (t+1) back to the
parameter server for another round of training (as the new model t).

 

.. rubric:: How can Intel® SGX Protect Federated Learning?
   :name: how-can-intel-sgx-protect-federated-learning

 

           Intel® Software Guard Extensions (SGX) are a set of CPU
instructions that can be used by developers to set aside private regions
of code and data (Bahmani, et al., 2017). These private regions, called
enclaves, are isolated sections of memory and compute that cannot be
accessed without a cryptographic key. Even users with root access or
physical access to the CPU cannot access the enclave without the
authorized key (Figure 2). This allows for developers to deploy their
code and data on untrusted machines in a secure manner. In 2015, Intel®
SGX was launched as the `first commercial
implementation <https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions/details.html>`__
of what is more formally called a trusted execution environment
(`TEE <https://en.wikipedia.org/wiki/Trusted_execution_environment>`__).

 

A black and blue text Description automatically generated

Figure 2 Intel® Software Guard Extensions (SGX) allow developers to
create secure enclaves that are not accessible by the OS or VM without
the proper security keys. This allows for developers to protect code
during use on the CPU.

 

One path to enable Intel® SGX in an application is to refactor the
application code to use the `Intel SDK for
SGX <https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions/sdk.html>`__.
However, many developers are reluctant to change their existing code.
`Graphene <https://grapheneproject.io/>`__ is an open-source library OS
that was created by Intel and its partners to provide developers an easy
way to leverage SGX without the need to change their existing
applications (Figure 3). Several commercial implementations based on
Graphene have been created by our partners, including
`Fortanix <https://fortanix.com/>`__ and
`SContain <https://scontain.com/index.html?lang=en>`__.

 

With Graphene, the developer simply defines a manifest file that
describes which code and data is allowed within the enclave. This
manifest file is used to automatically create the enclave on an
SGX-compatible CPU. For example, once Graphene is installed and the
manifest file is specified, the command:

 

+-----------------------------------------------------------------------+
| $ SGX=1 ./pal_loader httpd                                            |
+-----------------------------------------------------------------------+

 

will use the pal_loader command to create the enclave from the manifest
and run the web server (http) within the enclave. No other modifications
are needed for the httpd application.

A screenshot of a cell phone Description automatically generated

Figure 3 Graphene is an open-sourced project maintained by Intel that
allows developers to run their code within a secure enclave without
needing to modify the code.

.. rubric:: What is Intel® Federated Learning?
   :name: what-is-intel-federated-learning

 

A picture containing clock Description automatically generated

Figure 4 Intel® Federated learning with Intel® SGX allows federated
learning while protecting the model from theft/tampering on the remote
collaborator node.

By leveraging the security provided by Intel® SGX and the ease of
deployment provided by Graphene, Federated Learning can be protected
from adversarial attacks that are well documented in the literature.
With Intel SGX on every node in the federation, risks are mitigated even
if the nodes are not fully-controlled by the federation owner (Figure
4). Previous attacks have shown that adversaries may be able to steal
the model, reconstruct data based on the model updates, and/or prevent
convergence of the training when using untrusted nodes (Bagdasaryan,
Veit, Hua, Estrin, & Shmatikov, 2018; Bhagoji, Chakraborty, Supriyo, &
Calo, 2018). With Intel® Federated Learning protected via Intel® SGX,
adversaries are unable to use the model and unable to adapt their
attacks because the actual training is only visible to those with an
approved key (Figure 5). Additionally, Intel® SGX allows developers to
require
`attestation <https://software.intel.com/content/www/us/en/develop/articles/code-sample-intel-software-guard-extensions-remote-attestation-end-to-end-example.html>`__
from collaborators which proves that the collaborator actually ran the
expected code within the enclave. Attestation can either be done via a
trusted Intel server or by the developer’s own server. This stops
attackers from injecting their own code into the federated training.

 

A screenshot of a cell phone Description automatically generated

Figure 5 Secure Federated Learning with Intel SGX allows researchers to
leverage the benefits of federated learning while mitigating the risks.

 

.. rubric:: Installing and Running the Software
   :name: installing-and-running-the-software

 

.. rubric:: Accessing the Source Code Repository
   :name: accessing-the-source-code-repository

.. rubric:: 
   :name: section

 

The source code described in this manual should be open-sourced by Intel
for a future public release. Until then, it can only be accessed via a
legal agreement between Intel and the requestor. The development code
currently lives at https://github.com/IntelLabs/OpenFederatedLearning.
It is expected to be continually developed and improved. Changes to this
manual, the project code, the project design should be expected.

 

.. rubric:: Design Philosophy
   :name: design-philosophy

 

The overall design is that all of the scripts are built off of the
**federation plan**. The plan is just a YAML file that defines the
collaborators, aggregator, connections, models, data, and any other
parameters that describes how the training will evolve. In the “Hello
Federation” demos, the plan will be located in the YAML file:
bin/federations/plans/keras_cnn_mnist_2.yaml. As you modify the demo to
meet your needs, you’ll effectively just be modifying the plan along
with the Python code defining the model and the data loader in order to
meet your requirements. Otherwise, the same scripts will apply. When in
doubt, look at the FL plan’s YAML file.

 

.. rubric:: Configure The Federation
   :name: configure-the-federation

 

TLS encryption is used for the network connections. Therefore, security
keys and certificates will need to be created for the aggregator and
collaborators to negotiate the connection securely. For the “Hello
Federation” demo we will run the aggregator and collaborators on the
same localhost server so these configuration steps just need to be done
once on that machine.

 

.. rubric:: Steps:
   :name: steps

 

.. rubric:: All Nodes
   :name: all-nodes

 

1.     Unzip the source code

 

+-----------------------------------------------------------------------+
| unzip OpenFederatedLearning-master.zip                                |
+-----------------------------------------------------------------------+

 

2.    Change into the OpenFederatedLearning-master subdirectory.

 

+-----------------------------------------------------------------------+
| cd OpenFederatedLearning-master                                       |
+-----------------------------------------------------------------------+

 

.. rubric:: On the Aggregator Node
   :name: on-the-aggregator-node

 

1.    Change the directory to bin/federations/pki:

 

+-----------------------------------------------------------------------+
| cd bin/federations/pki                                                |
+-----------------------------------------------------------------------+

 

2.    Run the Certificate Authorityscript. This will setup the
Aggregator node as the Certificate Authority for the Federation. All
certificates will be signed by the aggregator. Follow the command-line
instructions and enter in the information as prompted. The script will
create a simple database file to keep track of all issued certificates.

 

+-----------------------------------------------------------------------+
| bash setup_ca.sh                                                      |
+-----------------------------------------------------------------------+

 

A screenshot of a social media post Description automatically generated

 

3.    Run the aggregator cert script, replacing
AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME with the
actual\ `FQDN <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`__\ for
the aggregator machine. You may optionally include the IP address for
the aggregator, replacing [IP_ADDRESS].

 

+-----------------------------------------------------------------------+
| bash create-aggregator.sh AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME      |
+-----------------------------------------------------------------------+

 

*Tip: You can discover
the*\ `FQDN <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`__\ *with
the Linux command: *\ hostname –fqdn

 

A screenshot of a social media post Description automatically generated

 

 

4.    \ **For each test machine you want to run collaborators on**\ , we
create a collaborator certificate, replacing TEST.MACHINE.NAME with the
actual test machine name. Note that this does not have to be the FQDN.
Also, note that this script is run on the Aggregator node because it is
the Aggregator that signs the certificate. Only Collaborators with valid
certificates signed by the Aggregator can join the federation.

 

+-----------------------------------------------------------------------+
| bash create-collaborator.sh TEST.MACHINE.NAME                         |
+-----------------------------------------------------------------------+

 

A screenshot of a social media post Description automatically generated

 

5.    Once you have the certificates created, you need to move the
certificates to the correct machines and ensure each machine has the
cert_chain.crt needed to verify cert signatures. For example, on a test
machine named TEST_MACHINE that you want to be able to run as a
collaborator, you should have:

 

+-----------------------------------------------------------------------+
| ·     bin/federations/pki/cert_chain.crt                              |
|                                                                       |
| ·     bin/federations/pki/col_TEST_MACHINE/col_TEST_MACHINE.crt       |
|                                                                       |
| ·     bin/federations/pki/col_TEST_MACHINE/col_TEST_MACHINE.key       |
+-----------------------------------------------------------------------+

 

           Note that once the certificates are transferred to the
collaborator, it is now possible to participate in any future
federations run by this aggregator. (The aggregator can revoke this
privilege.)

 

6.    On the aggregator machine you should have the files:

 

 

+-----------------------------------------------------------------------+
| ·      bin/federations/pki/cert_chain.crt                             |
|                                                                       |
| ·      bin/federations/pki/agg_AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME |
| /agg_AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME.crt                       |
|                                                                       |
| ·      bin/federations/pki/agg_AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME |
| /agg_AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME.key                       |
+-----------------------------------------------------------------------+

 

 

.. rubric:: Baremetal Installation
   :name: baremetal-installation

.. rubric:: 
   :name: section-1

 

Intel has tested the installation on Ubuntu 18.04 and Centos 7.6
systems. A Python 3.6 virtual environment
(`venv <https://docs.python.org/3/library/venv.html>`__) is used to
isolate the packages. The basic installation is via the Makefile
included in the root directory of the repository.

 

.. rubric:: Installation Steps
   :name: installation-steps

 

**NOTE: Steps 1-2 may have already been completed.**

 

1.    Unzip the source code

 

+-----------------------------------------------------------------------+
| unzip OpenFederatedLearning-master.zip                                |
+-----------------------------------------------------------------------+

 

2.    Change into the OpenFederatedLearning-master subdirectory.

 

+-----------------------------------------------------------------------+
| cd OpenFederatedLearning-master                                       |
+-----------------------------------------------------------------------+

 

3.    Use a text editor to open the YAML file for the federation plan.

 

+-----------------------------------------------------------------------+
| vi bin/federations/plans/keras_cnn_mnist_2.yaml                       |
+-----------------------------------------------------------------------+

 

This YAML file defines the IP addresses for the
aggregator\ `[EB1] <#_msocom_1>`__\  . It is the main file that controls
all of the execution of the federation. By default, the YAML file is
defining a federation where the aggregator runs on the localhost at port
5050(it us up to the developer to make sure that the port chosen is open
and accessible to all participants). For this demo, we’ll just focus on
running everything on the same server. You’ll need to edit the YAML file
and replace localhost with the aggregator address. Please make sure you
specify the fully-qualified domain name
(`FQDN <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`__)
address (required for security). For example:

 

A screenshot of a cell phone Description automatically generated

You can discover the FQDN by running the Linux command: hostname --fqdn

 

4.    If pyyaml is not installed, then use pip to install it:

 

+-----------------------------------------------------------------------+
| pip3 install pyyaml                                                   |
+-----------------------------------------------------------------------+

 

5.    Make sure that you followed the steps in **Configure the
Federation**\ and have copied the keys and certificates onto the
federation nodes.

 

6.    Build the virtual environment using the command:

 

+-----------------------------------------------------------------------+
| make install                                                          |
+-----------------------------------------------------------------------+

 

This should create a Python 3 virtual environment with the required
packages (e.g. TensorFlow, PyTorch, nibabel) that are used by the
aggregator and the collaborators. Note that you can add custom Python
packages by editing this section in the Makefile:

 

A screenshot of a cell phone Description automatically generated

 

| Just add your own line. For example, venv/bin/pip3 install my_package
| 

.. rubric:: Running the Baremetal Demo – “Hello Federation”
   :name: running-the-baremetal-demo-hello-federation

 

This is a quick tutorial on how to run a default federation using the
Intel® Federated Learning solution. The default federation will simply
train a TF/Keras CNN model on the MNIST dataset. We’ll define one
aggregator and two collaborator nodes. We’ve tested this tutorial on
both Ubuntu 18.04 and CentOS 7.6 Linux machines with a Python 3.6
virtual environment
(`venv <https://docs.python.org/3/library/venv.html>`__). For
demonstration purposes, we’ll run the aggregator and the 2 collaborators
on the same server; however, we point out in this tutorial how to easily
change the code to run the aggregator and collaborators on separate
nodes.

 

.. rubric:: Steps to Run the Baremetal Federation
   :name: steps-to-run-the-baremetal-federation

 

.. rubric:: On All Nodes
   :name: on-all-nodes

 

1.    Install Python 3.6 and the Python virtual environment. You can
find the instructions on the official `Python
website <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv>`__.
You may need to log out and back in for the changes to take effect.

 

+-----------------------------------------------------------------------+
| python3 -m pip install --user virtualenv                              |
+-----------------------------------------------------------------------+

 

.. rubric:: On the Aggregator
   :name: on-the-aggregator

 

1.    Follow the Baremetal Installation steps as described previously.

 

2.    It is assumed that the federation may be fine-tuning a previously
trained model. For this reason, the pre-trained weights for the model
will be stored within protobuf files on the aggregator and passed to the
collaborators during initialization. As seen in the YAML file, the
protobuf file with the initial weights is expected to be found in the
file keras_cnn_mnist\_init.pbuf. For this example, however, we’ll just
create an initial set of random model weights and putting it into that
file by running the command:

 

+-----------------------------------------------------------------------+
| ./venv/bin/python3 ./bin/create_initial_weights_file_from_flplan.py   |
| -p keras_cnn_mnist_2.yaml -dc local_data_config.yaml                  |
+-----------------------------------------------------------------------+

 

3.    Now we’re ready to start the aggregator by running the Python
script. Note that we will need to pass in the fully-qualified domain
name (FQDN) for the aggregator node address in order to present the
correct certificate.

 

+-----------------------------------------------------------------------+
| ./venv/bin/python3 ./bin/run_aggregator_from_flplan.py -p             |
| keras_cnn_mnist_2.yaml -ccnAGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME     |
+-----------------------------------------------------------------------+

 

At this point, the aggregator is running and waiting for the
collaborators to connect. When all of the collaborators connect, the
aggregator starts training. When the last round of training is complete,
the aggregator stores the final weights in the protobuf file that was
specified in the YAML file (in this case keras_cnn_mnist\_latest.pbuf).

 

.. rubric:: On the Collaborator
   :name: on-the-collaborator

 

**NOTE: If the demo is performed using the same node for the aggregator
and the collaborators, then steps 1-5 are already completed. You may
skip directly to step 6.**

 

1.    Unzip the source code

 

+-----------------------------------------------------------------------+
| unzip OpenFederatedLearning-master.zip                                |
+-----------------------------------------------------------------------+

 

2.    Change into the OpenFederatedLearning-master subdirectory.

 

+-----------------------------------------------------------------------+
| cd OpenFederatedLearning-master                                       |
+-----------------------------------------------------------------------+

 

3.    Make sure that you followed the steps in **Configure the
Federation** and have copied the keys and certificates onto the
federation nodes.

::

    

::

   4.     Copy the plan file (keras_cnn_mnist_2.yaml) from the aggregator over to the collaborator to the plan subdirectory (bin/federations/plans)

::

    

5.    Build the virtual environment using the command:

 

+-----------------------------------------------------------------------+
| make install                                                          |
+-----------------------------------------------------------------------+

 

6.    Now run the collaborator col_1using the Python script. Again, you
will need to pass in the fully qualified domain name in order to present
the correct certificate.

 

+-----------------------------------------------------------------------+
| ./venv/bin/python3 ./bin/run_collaborator_from_flplan.py -p           |
| keras_cnn_mnist_2.yaml -col col_1                                     |
| -ccnCOLLABORATOR.FULLY.QUALIFIED.DOMAIN.NAME                          |
+-----------------------------------------------------------------------+

 

7.    Repeat this for each collaborator in the federation. Once all
collaborators have joined, the aggregator will start and you will see
log messages describing the progress of the federated training.

A screenshot of a cell phone Description automatically generated

 

.. rubric:: Docker Installation
   :name: docker-installation

We will show you how to set up Intel\ ® Federated Learning on Docker
using a simple MNIST dataset and a TensorFlow/Keras CNN model as an
example. You will note that this is literally the same code as the
Baremetal Installation, but we are simply wrapping the venv within a
Docker container.

Before we start the tutorial, please make sure you have Docker installed
and configured properly. Here is an easy test to run in order to test
some basic functionality:

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ docker run hello-world                                           |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    Hello from Docker!                                                 |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    This message shows that your installation appears to be working co |
| rrectly.                                                              |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    ...                                                                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    ...                                                                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    ...                                                                |
+-----------------------------------------------------------------------+

::

    

.. rubric:: Installation Steps
   :name: installation-steps-1

::

    

::

   NOTE: Steps 1-2 may have already been performed.

::

    

1.    Unzip the source code

 

+-----------------------------------------------------------------------+
| unzip OpenFederatedLearning-master.zip                                |
+-----------------------------------------------------------------------+

 

2.    Change into the OpenFederatedLearning-master subdirectory.

 

+-----------------------------------------------------------------------+
| cd OpenFederatedLearning-master                                       |
+-----------------------------------------------------------------------+

 

3.    Use a text editor to open the YAML file for the federation plan.

 

+-----------------------------------------------------------------------+
| vi bin/federations/plans/keras_cnn_mnist_2.yaml                       |
+-----------------------------------------------------------------------+

 

This YAML file defines the IP addresses for the aggregator and
collaborators. It is the main file that controls all of the execution of
the federation. By default, the YAML file is defining a federation where
the aggregator runs on the localhost at port 5050 and there are two
collaborators(it us up to the developer to make sure that the port
chosen is open and accessible to all participants). For this demo, we’ll
just focus on running everything on the same server. You’ll need to edit
the YAML file and replace localhost with the aggregator address. Please
make sure you specify the fully-qualified domain name
(`FQDN <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`__)
address (required for security). For example:

 

 

 

A screenshot of a cell phone Description automatically generated

 

You can discover the FQDN by running the Linux command: hostname –fqdn

 

4.    If pyyaml is not installed, then use pip to install it:

 

+-----------------------------------------------------------------------+
| pip3 install pyyaml                                                   |
+-----------------------------------------------------------------------+

 

5.    Make sure that you followed the steps in **Configure the
Federation** and have copied the keys and certificates onto the
federation nodes.

 

6.    Build the Docker containers using the command:

 

+-----------------------------------------------------------------------+
| make build_containers model_name=keras_cnn                            |
+-----------------------------------------------------------------------+

 

           This should create the Docker containers that are used by the
aggregator and the collaborators.

+-----------------------------------------------------------------------+
| Successfully tagged tfl_agg_keras_cnn_bduser:0.1                      |
|                                                                       |
| Successfully tagged tfl_col_cpu_keras_cnn_bduser:0.1                  |
+-----------------------------------------------------------------------+

 

 

.. rubric:: Running the Docker Demo – “Hello Federation”
   :name: running-the-docker-demo-hello-federation

 

This is a quick tutorial on how to run a default federation using the
Intel® Federated Learning solution. The default federation will simply
train a TF/Keras CNN model on the MNIST dataset. We’ll define one
aggregator and two collaborator nodes. All 3 nodes will use Docker to
run their code. We’ve tested this tutorial on Ubuntu 18.04 and CentOS
7.6 Linux machines with Docker 18.06.1 and Python 3.6. For demonstration
purposes, we’ll run the aggregator and the 2 collaborators on the same
server; however, we point out in this tutorial how to easily change the
code to run the aggregator and collaborators on separate nodes.

 

.. rubric:: Steps to Run the Docker Federation
   :name: steps-to-run-the-docker-federation

 

.. rubric:: On the Aggregator
   :name: on-the-aggregator-1

 

1.    Follow the Docker Installation steps as described previously.

2.    Run the Docker container for the aggregator:

 

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    make run_agg_container model_name=keras_cnn                        |
+-----------------------------------------------------------------------+

 

When the Docker container for the aggregator begins you’ll see the
prompt above. This means you are within the running Docker container.
You can always exit back to the original Linux shell by typing \`exit\`.

 

3.    It is assumed that the federation may be fine-tuning a previously
trained model. For this reason, the pre-trained weights for the model
will be stored within protobuf files on the aggregator and passed to the
collaborators during initialization. As seen in the YAML file, the
protobuf file with the initial weights is expected to be found in the
file keras_cnn_mnist\_init.pbuf. For this example, however, we’ll just
create an initial set of random model weights and putting it into that
file by running the command:

 

+-----------------------------------------------------------------------+
| ./create_initial_weights_file_from_flplan.py -p                       |
| keras_cnn_mnist_2.yaml -dc docker_data_config.yaml                    |
+-----------------------------------------------------------------------+

 

4.    Now we’re ready to start the aggregator by running the Python
script:

 

+-----------------------------------------------------------------------+
| python3 run_aggregator_from_flplan.py -p keras_cnn_mnist_2.yaml -ccn  |
| AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME                                |
+-----------------------------------------------------------------------+

 

Notice we have to pass the fully qualified domain name (FQDN) so that
the correct certificate can be presented. At this point, the aggregator
is running and waiting for the collaborators to connect. When all of the
collaborators connect, the aggregator starts training. When the last
round of training is complete, the aggregator stores the final weights
in the protobuf file that was specified in the YAML file (in this case
keras_cnn_mnist\_latest.pbuf).

 

.. rubric:: On the Collaborators
   :name: on-the-collaborators

 

**NOTE: If the demo is performed using the same node for the aggregator
and the collaborators, then steps 1-5 are already completed. You may
skip directly to step 6.**

 

1.    Unzip the source code

 

+-----------------------------------------------------------------------+
| unzip OpenFederatedLearning-master.zip                                |
+-----------------------------------------------------------------------+

 

2.    Change into the OpenFederatedLearning-master subdirectory.

 

+-----------------------------------------------------------------------+
| cd OpenFederatedLearning-master                                       |
+-----------------------------------------------------------------------+

 

3.    Make sure that you followed the steps in **Configure the
Federation** and have copied the keys and certificates onto the
federation nodes.

::

    

::

   4.     Copy the plan file (keras_cnn_mnist_2.yaml) from the aggregator over to the collaborator to the plan subdirectory (bin/federations/plans)

::

    

::

   5.     Either transfer the Docker image over from the aggregator or build the Docker container using the command:

 

+-----------------------------------------------------------------------+
| make build_containers model_name=keras_cnn                            |
+-----------------------------------------------------------------------+

 

6.    Now run the Docker on the collaborator. For collaborator #1, run
this command:

::

    

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    make run_col_container model_name=keras_cnn col_name=col_1         |
+-----------------------------------------------------------------------+

 

7.    Now run the collaborator Python script to start the collaborator.
Notice that you’ll need to specify the fully qualified domain name
(FQDN) for the collaborator node to present the correct certificate.

 

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    python3 run_collaborator_from_flplan.py -p keras_cnn_mnist_2.yaml  |
| -col col_1 -dc docker_data_config.yaml -ccn COLLABORATOR.FULLY.QUALIF |
| IED.DOMAIN.NAME                                                       |
+-----------------------------------------------------------------------+

 

8.    Repeat this for each collaborator in the federation. Once all
collaborators have joined, the aggregator will start and you will see
log messages describing the progress of the federated training.

 

 

.. rubric:: Federated Training of the 2D U-Net (Brain Tumor
   Segmentation)
   :name: federated-training-of-the-2d-u-net-brain-tumor-segmentation

This tutorial assumes that you've run the previous MNIST demos. We’ll
provide fewer details about the installation and highlight the specific
differences needed for training a 2D U-Net. We also assume that you’ve
downloaded the Brain Tumor Segmentation
(\ `BraTS <http://braintumorsegmentation.org/>`__\ ) dataset and have
the data accessible to the training nodes.

1.    In the “Hello Federation” demo the MNIST data was downloaded
during runtime from an online repository. However, in this example we
are allocating data directly. To make this work, we create a <Brats
Symlinks Dir>, which is has directories of symlinks to the data for each
institution number. Setting this up is out-of-scope for this code at the
moment, so we leave this to the reader. In the end, our directory looks
like below. Note that "0-9" allows us to do data-sharing training (i.e.
as if the data were centrally-pooled rather than federated at different
collaborator sites).

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ ll <Brats Symlinks Dir>                                          |
|                                                                       |
| ::                                                                    |
|                                                                       |
|                                                                       |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    ...                                                                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      drwxr-xr-x  90 <user> <group> 4.0K Nov 25 22:14 0                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      drwxr-xr-x 212 <user> <group>  12K Nov  2 16:38 0-9              |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      drwxr-xr-x  24 <user> <group> 4.0K Nov 25 22:14 1                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      drwxr-xr-x  36 <user> <group> 4.0K Nov 25 22:14 2                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      drwxr-xr-x  14 <user> <group> 4.0K Nov 25 22:14 3                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      drwxr-xr-x  10 <user> <group> 4.0K Nov 25 22:14 4                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      drwxr-xr-x   6 <user> <group> 4.0K Nov 25 22:14 5                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      drwxr-xr-x  10 <user> <group> 4.0K Nov 25 22:14 6                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      drwxr-xr-x  16 <user> <group> 4.0K Nov 25 22:14 7                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      drwxr-xr-x  17 <user> <group> 4.0K Nov 25 22:14 8                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      drwxr-xr-x   7 <user> <group> 4.0K Nov 25 22:14 9                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    ...                                                                |
+-----------------------------------------------------------------------+

2.    For this demo, we’ll again consider only 2 collaborators.Recall
that the design philosophy states that everything gets specified in the
federation plan. For the BraTS demo, the FL plan is called
bin/federations/plans/brats17_insts2_3.yaml. Edit that to file specify
the correct addresses for your machines.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ vi bin/federations/plans/tf_2dunet_brats_insts2_3.yaml           |
+-----------------------------------------------------------------------+

Find thekeys for the address ("agg_addr") and port ("agg_port") and
replace them with your values.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    federation:                                                        |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      fed_id: &fed_id 'fed_0'                                          |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      opt_treatment: &opt_treatment 'AGG'                              |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      polling_interval: &polling_interval 4                            |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      rounds_to_train: &rounds_to_train 50                             |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      agg_id: &agg_id 'agg_0'                                          |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      agg_addr: &agg_addr "agg.domain.com"   # CHANGE THIS STRING      |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      agg_port: &agg_port <some_port>        # CHANGE THIS INT         |
+-----------------------------------------------------------------------+

::

    

3.    Edit the docker data config file to refer to the correct username
(the name of the account you are using.
Openbin/federations/docker_data_config.yamland replace the username with
your username.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ vi bin/federations/docker_data_config.yaml                       |
|                                                                       |
| collaborators:                                                        |
|                                                                       |
| col_one_big:                                                          |
|                                                                       |
| brats: &brats_data_path '/home/<USERNAME>/tfl/datasets/brats' #       |
| replace with your username                                            |
|                                                                       |
| col_0:                                                                |
|                                                                       |
| brats:\ `\* <#id5>`__\ brats_data_path mnist_shard: 0                 |
|                                                                       |
| col_1:                                                                |
|                                                                       |
| brats:\ `\* <#id7>`__\ brats_data_path mnist_shard: 1                 |
|                                                                       |
| ...                                                                   |
+-----------------------------------------------------------------------+

 

.. rubric:: Start an Aggregator
   :name: start-an-aggregator

1.    Build the docker images "tfl_agg_<model_name>_<username>:0.1" and
"tfl_col_<model_name>_<username>:0.1" using project folder Makefile
targets. This uses the project folder "Dockerfile". We only build them
once, unless we changeDockerfile. We pass along the proxy configuration
from the host machine to the docker container, so that your container
would be able to access the Internet from typical corporate networks. We
also create a container user with the same UID so that it is easier to
access the mapped local volume from the docker container. Note that we
include the username to avoid development-time collisions on shared
development servers. We build the collaborator Docker image upon the
aggregator image, adding necessary dependencies such as the mainstream
deep learning frameworks. You may
modify./models/<model_name>/Dockerfileto install the needed packages.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ make build_containers model_name=tf_2dunet                       |
+-----------------------------------------------------------------------+

2.    Run the aggregator container (entering a bash shell inside the
container), again using the Makefile. Note that we map the local
volumes./bin/federationsto the container.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ make run_agg_container model_name=tf_2dunet dataset=brats        |
+-----------------------------------------------------------------------+

 

 

3.    In the aggregator container shell, build the initial weights files
providing the global model initialization that will be sent from the
aggregator out to all collaborators.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ ./create_initial_weights_file_from_flplan.py -p tf_2dunet_brats_ |
| insts2_3.yaml -dc docker_data_config.yaml                             |
+-----------------------------------------------------------------------+

4.    In the aggregator container shell, run the aggregator, using a
shell script provided in the project.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ ./run_brats_aggregator.sh                                        |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    Loaded logging configuration: logging.yaml                         |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    2020-01-15 23:17:18,143 - tfedlrn.aggregator.aggregatorgrpcserver  |
| - DEBUG - Starting aggregator.                                        |
+-----------------------------------------------------------------------+

::

    

::

    

.. rubric:: Start Collaborators
   :name: start-collaborators

Note: the collaborator machines can be the same as the aggregator
machine.

1. (**On each collaborator machine**) Enter the project folder and build
   the containers as above.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ make build_containers model_name=tf_2dunet                       |
+-----------------------------------------------------------------------+

::

    

2.    (On the first collaborator machine) Run the first collaborator
container. Note we are using collaborators 2 and 3.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ make run_col_container model_name=tf_2dunet dataset=brats col_na |
| me=2                                                                  |
+-----------------------------------------------------------------------+

::

    

3.    In this first collaborator shell, run the collaborator using the
provided shell script.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ ./run_brats_collaborator.sh 2                                    |
+-----------------------------------------------------------------------+

::

    

4.    (On the second collaborator machine, which could be a second
terminal on the first machine) Run the second collaborator container
(entering a bash shell inside the container).

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ make run_col_container model_name=tf_2dunet dataset=brats col_na |
| me=col_3                                                              |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    docker run \                                                       |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    ...                                                                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    bash                                                               |
+-----------------------------------------------------------------------+

::

    

5.    In the second collaborator container shell, run the second
collaborator.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ ./run_brats_collaborator.sh 3                                    |
|                                                                       |
| ::                                                                    |
|                                                                       |
|                                                                       |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    ...                                                                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    ...                                                                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    ...                                                                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|                                                                       |
+-----------------------------------------------------------------------+

 

 

.. rubric:: Porting your Experiments to Intel® Federated Learning
   :name: porting-your-experiments-to-intel-federated-learning

 

.. rubric:: Design Philosophy
   :name: design-philosophy-1

 

Intel® Federated Learning is completely agnostic to deep learning
frameworks. The code works equally-well with TensorFlow, PyTorch, or
other frameworks because the code only actually passes the model and
optimizerweight values around the network. The graph and framework code
is never passed.

A close up of a device Description automatically generated

Figure 5 The code remains framework agnostic because it only passes
protobufs across the federation.

 

To accomplish this, we’ll define methodsforeach model’s Python class, to
define how to take the weights from the computational graph and
convertthe set of weights into anumpy array valued dictionary(and
vise-versa).As we demonstrate in our example model code,these methods
can be inherited from framework specific model classesto avoid
duplicationfor each model utilizing the same ML framework.The
collaborator object classes in our code(who have the model class as an
attribute)take care of further converting these arrays toa generic,
protobuf object (and vice versa, Figure 5). Once these methods of
themodel classes are defined, the collaborators know how to take their
framework-specific model, convert the weights to the generic protobuf,
transfer the protobuf to the aggregator, receive the consensus protobuf
results back from the aggregator, and finally load new weights into the
local model. Figure 6 shows howthe existingKeras model class examplecode
takesa dictionary ofnumpyweights (tensor_dict) and writesthe weights
into themodelcompute graph. The tensor_dict containsthe names and values
of the model and optimizerweights. The methods set_tensor_dict and
get_tensor_dict are used to pass tensors to and from the model’s compute
graph. These methods have been pre-defined for TensorFlow, Keras, and
PyTorch. Support for new frameworks can be created similarly, and
theIntel® Federated Learning code enables conversion of these
dictionaries to and from protobuf independent of framework. The
Federation Plan’s YAML file defines the protobuf filenames for the
initial, latest, and best models (e.g. keras_cnn_mnist\_best.pbuf).

 

A screenshot of a social media post Description automatically generated

Figure 6 Method defined to write the values of a dictionary of model and
optimizer weights into the compute graph of a Keras model.. Similar
methodsexist for TensorFlow and PyTorch. An analogous method exists to
read the weights off of the graph and createthe tensor_dict of weight
names to numpy array values.

.. rubric:: Repository Structure
   :name: repository-structure

 

|Text Box: Figure 7 Intel® Federated Learning repository
structure.|\ |Text Box: ── bin │ ├── federations │ │ ├── plans │ │ └──
weights ├── data │ ├── dummy │ ├── pytorch │ └── tensorflow ├── docs ├──
models │ ├── dummy │ ├── pytorch │ │ ├── pt_2dunet │ │ ├── pt_cnn │ │
└── pt_resnet │ └── tensorflow │ ├── keras_cnn │ ├── keras_resnet │ └──
tf_2dunet ├── tests └── tfedlrn|\ Figure 7 shows that the Intel®
Federated Learning code repository is divided into 4 major sections:
bin,data,models,andtfedlrn. 

 

The tfedlrn subdirectory contains the code that orchestrates the
aggregator and collaborator nodes. It is not anticipated that the
developer will need to modify this code unless the aggregation algorithm
or communication protocols need to be modified.

 

In the previous “Hello Federation” demos, we showed how the plans and
weights files contained within the bin/federations subdirectory were
used to describe the federation architecture. Specifically, the
federation plan is a YAML file located in the bin/federations/plans
subdirectory.

 

 

 

 

 

 

.. rubric:: The Federation Plan
   :name: the-federation-plan

|Text Box: Figure 8 The YAML file of an example Federation
Plan.|\ |image3|

Figure 8 shows the Federation Plan from the “Hello Federation” demo. The
first section of the YAML file (federation, line 37) defines the
federation label (line 38), number of training rounds (line 41), the
aggregator’s fully defined domain name
(`FQDN <https://en.wikipedia.org/wiki/Fully_qualified_domain_name>`__)
address and port (lines 43-44). As noted in the documentation for these
demos, this FQDN should exactly match the security certificate (cf.
section **Configuring the Federation**). Beginning on line 45, the
collaborator names should be specified. Note that like the federation
and aggregator names (lines 38 and 42), these may be any label chosen by
the developer. Lines 48-50 define the filenames for the initial, latest,
and best model protobuf files. Finally, beginning at line 57, the
developer can describe the network addresses of the approved
collaboratorsfor testing on this specific federation (aka the
whitelist). Note that these addresses must exactly match the common name
provided in thesecurity certificates that were generated forthese
collaborators (cf. section **Configuring the Federation**). By
specifically whitelisting collaborators, the Federation can include
collaborators within some federations, but easily exclude them from
other federations. This granularity of access is important for securing
access to the
federation.\ `[EB2] <#_msocom_2>`__\ \  \ `[RGA3] <#_msocom_3>`__\ \ \  
Lines 64-65 provide switches to disable secure connections and client
authentication. These should only be changed when debugging network
connections (and only then with extreme caution as they eliminate any
security in the networking).

 

|Text Box: Figure 9 The Model and Data class sections of the FL
Plan.|\ |A close up of text on a black background Description
automatically generated|\ The remaining two subdirectories—models and
data—are the most important parts of the repository to the individual
developer. These two subdirectories contain the Python scripts that
define the modeland datapre-processing/loading (in whatever deep
learning framework is desired). The Federation Plan (Figure 9) has
sections to define which specific Python Model and Data classto use for
thefederation. Note that a subset ofuser-defined classparameters (e.g.
nb_collaborators, batch_size) may be defined here and will be passed to
the class constructors at runtime (in addition to other parameters- for
example the data_paththat is brought in from the local data config)via
\**kwargs parameters in the class initializer.

 

.. rubric:: Models
   :name: models

 

           Because the code is agnostic to the framework, the developer
needs to define what to do when the federation starts training. At the
collaborator nodes, the training is expected to be using a deep learning
framework to train
locally.\ `[EB4] <#_msocom_4>`__\ \  \ `[RGA5] <#_msocom_5>`__\ \ \  The
FLModel class defines the interface between the modelobject and the rest
of the code via thecollaborator object(Figure 10). This parent class is
ultimately inherited by all models.

 

A screenshot of a cell phone Description automatically generated

Figure 10 The pre-defined class FLModel.

           Figure 11 shows an example DummyModel class that inherits the
FLModel parent. This class simply waits for a random number of seconds
when the train\_batches method is called and then returns a random value
for the loss metric. Similar code is used for the validate method. This
Dummy class is useful for sanity-checking the FL pipeline without having
to define a model or dataset.

 

Figure 11 A Dummy Model class. This inherits the FLModel class, but the
train\_batches method simply waits a random number of seconds and then
returns a random loss value.

.. rubric:: TensorFlow / Keras / PyTorch
   :name: tensorflow-keras-pytorch

 

           There are pre-defined model classes for TensorFlow, Keras,
and PyTorch in the repository under the models subdirectory. Note that
commonly-used, framework-specific methods, such as,
train_batches,validate,get_tensor_dict,
set\_tensor_dict,reset_opt_vars,andinitialize_globals methods are
already definedin framework dependent base classes(or examples
provided)forallofthese frameworks. The only attributes that remainto be
developedwhen inheriting from theseclasses arethose providing specific
model architecture(self.model) andthe data objectforservingup batchesfor
trainingand validation.

For example, Figure 12 shows the class that should be inherited by all
models using the Keras DL framework. The developer onlyneeds toinherit
fromKerasFLModel, andoverride theself.model attribute, replacing with a
specifickeras.Modelobject that defines their specific model
topology(line 21).

 

 

Figure 12 The base class for all Keras Models. New Keras models can be
created by inheriting the KerasFLModel class and overriding the
self.model instantiation with the custom Keras model.

In most cases, all other framework-specific methods should be correctly
defined. For example, Figure 13 demonstrates that the train\_batches
method for Keras simply calls the Keras fit method for a single epoch
and returns the loss value as a float.

 

Figure 13 The pre-defined train_epoch method for Keras models.

 

           The FL Plan defines where to find the model-specific code for
the federation. In Figure 9, the FL plan specifies that the model
definition exists in the subdirectory models/tensorflow/keras_cnn and
the class name is KerasCNN. Additional parameters to the Model class are
passed at runtime via the \**kwargsparameter. Figure 14 shows the code
for the custom KerasCNN Model class. This class inherits the parent
class KerasFLModel(line 13) and defines theself.modelobject using the
custom code in the methodbuild_model(line 20). Other than the initial
model definition, all other methods are pre-defined by the parent class.
Similar Keras models can be created by inheriting theKerasFLModelclass
and defining the model topology using the Keras Model API.

.. rubric:: 
   :name: section-2

 

A screenshot of a cell phone Description automatically generated

Figure 14 A user-defined Keras model. Note that the only change the
developer needs to make is to define the Keras model. All other methods
are pre-defined and inherited from the KerasFLModel parent class.

.. rubric:: 
   :name: section-3

 

           Similar templates and demos exist for PyTorch and generic
(i.e. non-Keras API) TensorFlow code. Figure 15 shows the pre-defined
code for the PyTorch train\_batches method. For these frameworks, some
customization of the train_epoch method may be needed because PyTorch
(and generic TensorFlow) are not as rigidly templated as the Keras API.
Nevertheless, all FLModel methods assume that train_epoch will perform
one epochs of training and return the value of the loss for the model.

 

 

Figure 15 A PyTorch method for train\_batches from the pt_cnn.py custom
model class.

 

.. rubric:: Data
   :name: data

 

|Text Box: Figure 16 The FLData class.|\ |A screen shot of a computer
Description automatically generated|\ The Data subdirectory defines the
Python classes for loading the data and serving it to the model (Figure
16). The primary methods for this class are get_train_loader and
get_val_loader. These Python methods should be modified by the developer
to define data loaderobjects that can be used as iterators to serve up
batches of data to the model for training and validation.

 

           Figure 9 shows the Data definitions within the FL Plan for
the “Hello Federation” demo. The **Data** object within the YAML file
defines the path to the Python class code (*code.path*), the class name
to use for the data loader (*class_name*), and some user-defined
parameters such as the number of collaborators (*nb_collaborators*) and
batch size (*batch_size*). Number of collaborators is used in this case
sincetheMNIST datasetis downloadedandsharded (partitioned)according to
how many collaborators there are. The specific shard number to be used
for each collaborator is looked up(using the collaborator name)in the
localdata config and passedto the data object constructorat runtime.

 

A screenshot of a cell phone Description automatically generated

Figure 17 The custom Data class used for the "Hello Federation" demo.

           Figure 17 shows the custom data loader defined by the Data
class for the “Hello Federation” demo. It simply loads the particular
partition of MNIST,and splits it into a training and validation set
topopulatethe in-memory variables self.X_train, self.y_train,
self.X_val, and self.y_val. Out-of-memory data loaders (e.g.
`tf.data <https://www.tensorflow.org/guide/data>`__,
`torch.utils.data <https://pytorch.org/docs/stable/data.html>`__) may be
used but are not currently implemented in the code and are out of the
scope for this manual. Nevertheless, any Python method that returns
NumPyarrayproducing data loaders is allowed.

 

| `[EB6] <#_msocom_6>`__\ \  \ `[RGA7] <#_msocom_7>`__\ \ \  
| 

.. rubric:: Local Data Config
   :name: local-data-config

 

Like the model code, thecode for thedata objectmay be shared among many
collaborators in a federation.Wethereforeallowthedata file locationsto
be independent of the data object code, instead passing this
informationto the data object constructor at runtime. In order to do
this,we utilize a local data configuration file. ThisYAMLfile(see Figure
18 for an example)is located in the bin/federations directory, andis
used byexecutables tolook up the data path(or other relevant
information)for a particular collaboratorand data type. At present there
is only one top level key, “collaborators”, for this file.In deployment
we expect that possibly each collaborator will only have their own
collaborator name populatedin their file, but for experimentation we
have many(simulated)collaborators listed.For each collaborator name
listed under this key, there are multiple dataset names as keys, the
value for whichisa datapath (or other information)to be usedfor that
dataset.Forlarge unpartitionedpublic datasets used for experimentation,
itmay make sense to hard code the locationof the datain the data
object,in which case the value under a specific data name can be used
tocommunicate the shard number being used for that collaborator. This is
what we do for MNIST and CIFAR10 in our example code.

 

Figure 18 The local data configuration YAML file containing information
needed when looking up data for a particular collaborator.

 

.. rubric:: Running the simulator
   :name: running-the-simulator

The Federated Learning simulator allows data scientists to test models
and federation plans without having to run them across the network. In
the simulator, the aggregator and all of the collaborators are located
on the same node. This allows for basic experiments with respect to the
learning rate, number of rounds, model topology, and other
hyperparameters before deloying the full federated learning plan.

Note that much of the code used for simulation (ex. collaborator and
aggregator objects) is the same as for the multiprocess solution with
gRPC. Since the collaborator calls aggregator object methods via the
gRPC channel object, simulation is performed by simply replacing the
channel object provided to each collaborator with the aggregator object.

A picture containing screenshot Description automatically generated

.. rubric:: 
   :name: section-4

 

.. rubric:: Simulated Federated Training of an MNIST Classifier across
   10 Collaborators
   :name: simulated-federated-training-of-an-mnist-classifier-across-10-collaborators

 

The plan we will use for this tutorial iskeras_cnn_mnist_10.yaml.

 

.. rubric:: Create the project virtual environment
   :name: create-the-project-virtual-environment

 

::

   NOTE: Steps 1-2 may have already been performed.

1. To prepare, make sure you have python 3.6 (or higher) with virtualenv
   installed.

::

    

2.    Unzip the source code

 

+-----------------------------------------------------------------------+
| unzip OpenFederatedLearning-master.zip                                |
+-----------------------------------------------------------------------+

 

3.    Change into the OpenFederatedLearning-master subdirectory.

 

+-----------------------------------------------------------------------+
| cd OpenFederatedLearning-master                                       |
+-----------------------------------------------------------------------+

4.    Create the virtual environment, and change to the bin directory.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ make clean                                                       |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    $ make install                                                     |
|                                                                       |
| ::                                                                    |
|                                                                       |
|    $ cd bin                                                           |
+-----------------------------------------------------------------------+

::

    

5.    Create the initial weights file for the model to be trained.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ ../venv/bin/python create_initial_weights_file_from_flplan.py -p |
|  keras_cnn_mnist_10.yaml                                              |
+-----------------------------------------------------------------------+

::

    

4. Start the simulation.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ ../venv/bin/python run_simulation_from_flplan.py -p keras_cnn_mn |
| ist_10.yaml                                                           |
+-----------------------------------------------------------------------+

::

    

5. You'll find the output from the aggregator in
   bin/logs/aggregator.log. Grep this file to see results (one example
   below). You can check the progress as the simulation runs, if
   desired.

+-----------------------------------------------------------------------+
| ::                                                                    |
|                                                                       |
|    $ grep -A 2 "round results" logs/aggregator.log                    |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      2020-03-30 13:45:33,404 - tfedlrn.aggregator.aggregator - INFO - |
|  round results for model id/version KerasCNN/1                        |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      2020-03-30 13:45:33,404 - tfedlrn.aggregator.aggregator - INFO - |
|         validation: 0.4465000107884407                                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      2020-03-30 13:45:33,404 - tfedlrn.aggregator.aggregator - INFO - |
|         loss: 1.0632034242153168                                      |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      --                                                               |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      2020-03-30 13:45:35,127 - tfedlrn.aggregator.aggregator - INFO - |
|  round results for model id/version KerasCNN/2                        |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      2020-03-30 13:45:35,127 - tfedlrn.aggregator.aggregator - INFO - |
|         validation: 0.8630000054836273                                |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      2020-03-30 13:45:35,127 - tfedlrn.aggregator.aggregator - INFO - |
|         loss: 0.41314733028411865                                     |
|                                                                       |
| ::                                                                    |
|                                                                       |
|      --                                                               |
+-----------------------------------------------------------------------+

Note thataggregator.log is always appended to, so will include results
from previous runs.

6. Edit the plan to train for more rounds, etc.

 

 

 

.. rubric:: Bibliography
   :name: bibliography

 

Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., & Shmatikov, V. (2018).
How To Backdoor Federated Learning. arXiv,
https://arxiv.org/abs/1807.00459.

Bahmani, R., Barbosa, M., Brasser, F., Portela, B., Sadeghi, A.-R.,
Scerri, G., & Warinschi, B. (2017). Secure multiparty computation from
SGX. https://hal.archives-ouvertes.fr/hal-01898742/file/2016-1057.pdf.

Bhagoji, A. N., Chakraborty, Supriyo, M. P., & Calo, S. (2018).
Analyzing Federated Learning through an Adversarial Lens. arXiv,
https://arxiv.org/abs/1811.12470.

Bonawitz, K., Eichner, H., Grieskamp, W., Huba, D., Ingerman, A.,
Ivanov, V., . . . Van Overveldt, T. (2019). Towards federated learning
at scale: System design.

McMahan, H. B. (2016). Communication-efficient learning of deep networks
from decentralized data.

Sheller, M., Reina, G. A., Edwards, B., Martin, J., & Bakas, S. (2019).
Multi-institutional Deep Learning Modeling Without Sharing Patient Data:
A Feasibility Study on Brain Tumor Segmentation. Lecture Notes in
Computer Science.

Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). Federated Machine
Learning: Concept and Applications. ACM Transactions on Intelligent
Systems and Technology.

 

 

.. raw:: html

   </div>

.. raw:: html

   <div style="mso-element:comment-list">

--------------

.. raw:: html

   <div style="mso-element:comment">

.. raw:: html

   <div id="_com_1" class="msocomtxt" language="JavaScript"
   onmouseover="msoCommentShow('_anchor_1','_com_1')"
   onmouseout="msoCommentHide('_com_1')">

collaborators poll the aggregator, but not vise-versa. Location of the
collaborators is never needed. \ `[EB1] <#_msoanchor_1>`__

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div style="mso-element:comment">

.. raw:: html

   <div id="_com_2" class="msocomtxt" language="JavaScript"
   onmouseover="msoCommentShow('_anchor_2','_com_2')"
   onmouseout="msoCommentHide('_com_2')">

This makes it sound to me like whitelisting is a feature that is
relevant outside of development, when we want to make it clear it is for
testing only. \ `[EB2] <#_msoanchor_2>`__

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div style="mso-element:comment">

.. raw:: html

   <div id="_com_3" class="msocomtxt" language="JavaScript"
   onmouseover="msoCommentShow('_anchor_3','_com_3')"
   onmouseout="msoCommentHide('_com_3')">

 \ \ \ `[RGA3] <#_msoanchor_3>`__\ Ok. I think I probably need to talk
about this, then. I thought whitelisting was always needed.

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div style="mso-element:comment">

.. raw:: html

   <div id="_com_4" class="msocomtxt" language="JavaScript"
   onmouseover="msoCommentShow('_anchor_4','_com_4')"
   onmouseout="msoCommentHide('_com_4')">

Not sure what the focus should be now, but just wanted to comment
that: \ `[EB4] <#_msoanchor_4>`__

1) Local training does not need to be deep learning, any model training
based on batched gradient descent (including SVM for example) could plug
into our tfedlrn code.

2) In the future the model library code may be distributed by a
governing entity (or pulled from a common source).

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div style="mso-element:comment">

.. raw:: html

   <div id="_com_5" class="msocomtxt" language="JavaScript"
   onmouseover="msoCommentShow('_anchor_5','_com_5')"
   onmouseout="msoCommentHide('_com_5')">

 \ \ \ `[RGA5] <#_msoanchor_5>`__\ True. I think we can expand this in
the future to talk about other ML methods.

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div style="mso-element:comment">

.. raw:: html

   <div id="_com_6" class="msocomtxt" language="JavaScript"
   onmouseover="msoCommentShow('_anchor_6','_com_6')"
   onmouseout="msoCommentHide('_com_6')">

At this point we cannot say this. Even the collaborator object in our
code gets the batch size information by accessing model.data.batch_size.
Further, our models use the data attributes a lot (independent of
get_data) like data.get_feature_shape to learn the input shape,
data.get_training(validation)_data_size, and of course
data.get_train(val)_loader. \ `[EB6] <#_msoanchor_6>`__

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   <div style="mso-element:comment">

.. raw:: html

   <div id="_com_7" class="msocomtxt" language="JavaScript"
   onmouseover="msoCommentShow('_anchor_7','_com_7')"
   onmouseout="msoCommentHide('_com_7')">

 \ \ \ `[RGA7] <#_msoanchor_7>`__\ Ok.

 

.. raw:: html

   </div>

.. raw:: html

   </div>

.. raw:: html

   </div>

.. |Text Box: Figure 7 Intel® Federated Learning repository structure.| image:: Intel%20Federated%20Learning%20Manual.fld/image019.png
   :width: 239px
   :height: 26px
.. |Text Box: ── bin │ ├── federations │ │ ├── plans │ │ └── weights ├── data │ ├── dummy │ ├── pytorch │ └── tensorflow ├── docs ├── models │ ├── dummy │ ├── pytorch │ │ ├── pt_2dunet │ │ ├── pt_cnn │ │ └── pt_resnet │ └── tensorflow │ ├── keras_cnn │ ├── keras_resnet │ └── tf_2dunet ├── tests └── tfedlrn| image:: Intel%20Federated%20Learning%20Manual.fld/image020.png
   :width: 240px
   :height: 365px
.. |Text Box: Figure 8 The YAML file of an example Federation Plan.| image:: Intel%20Federated%20Learning%20Manual.fld/image021.png
   :width: 221px
   :height: 26px
.. |image3| image:: Intel%20Federated%20Learning%20Manual.fld/image022.png
   :width: 218px
   :height: 289px
.. |Text Box: Figure 9 The Model and Data class sections of the FL Plan.| image:: Intel%20Federated%20Learning%20Manual.fld/image023.png
   :width: 185px
   :height: 38px
.. |A close up of text on a black background Description automatically generated| image:: Intel%20Federated%20Learning%20Manual.fld/image024.png
   :width: 181px
   :height: 103px
.. |Text Box: Figure 16 The FLData class.| image:: Intel%20Federated%20Learning%20Manual.fld/image033.png
   :width: 250px
   :height: 26px
.. |A screen shot of a computer Description automatically generated| image:: Intel%20Federated%20Learning%20Manual.fld/image034.png
   :width: 246px
   :height: 94px
