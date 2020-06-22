Intel® Federated Learning

![A screenshot of a cell phone Description automatically
generated](media/image1.png){width="6.5in" height="4.320138888888889in"}

*Secure, Privacy-Preserving Machine Learning*

**Technical Manual\
**

Table of Contents {#table-of-contents .TOCHeading}
=================

[Overview 3](#overview)

[What is Federated Learning? 3](#what-is-federated-learning)

[How can Intel® SGX Protect Federated Learning?
3](#how-can-intel-sgx-protect-federated-learning)

[What is Intel® Federated Learning?
5](#what-is-intel-federated-learning)

[Installing and Running the Software
7](#installing-and-running-the-software)

[Accessing the Source Code Repository
7](#accessing-the-source-code-repository)

[Design Philosophy 7](#design-philosophy)

[Configure The Federation 7](#configure-the-federation)

[Baremetal Installation 11](#baremetal-installation)

[Installation Steps 11](#installation-steps)

[Running the Baremetal Demo -- "Hello Federation"
13](#running-the-baremetal-demo-hello-federation)

[Steps to Run the Baremetal Federation
13](#steps-to-run-the-baremetal-federation)

[Docker Installation 16](#docker-installation)

[Installation Steps 16](#installation-steps-1)

[Running the Docker Demo -- "Hello Federation"
18](#running-the-docker-demo-hello-federation)

[Steps to Run the Docker Federation
18](#steps-to-run-the-docker-federation)

[Federated Training of the 2D U-Net (Brain Tumor Segmentation)
21](#federated-training-of-the-2d-u-net-brain-tumor-segmentation)

[Start an Aggregator 22](#start-an-aggregator)

[Start Collaborators 23](#start-collaborators)

[Porting your Experiments to Intel® Federated Learning
25](#porting-your-experiments-to-intel-federated-learning)

[Design Philosophy 25](#design-philosophy-1)

[Repository Structure 26](#repository-structure)

[The Federation Plan 27](#the-federation-plan)

[Models 28](#models)

[TensorFlow / Keras / PyTorch 28](#tensorflow-keras-pytorch)

[Data 31](#data)

[Local Data Config 33](#local-data-config)

[Running the simulator 34](#running-the-simulator)

[Simulated Federated Training of an MNIST Classifier across 10
Collaborators
34](#simulated-federated-training-of-an-mnist-classifier-across-10-collaborators)

[Create the project virtual environment
34](#create-the-project-virtual-environment)

[Bibliography 37](#_Toc43273108)

Overview
========

What is Federated Learning?
---------------------------

Federated learning is a distributed machine learning approach that
enables organizations to collaborate on machine learning projects
without sharing sensitive data, such as, patient records, financial
data, or classified secrets (McMahan, 2016; Sheller, Reina, Edwards,
Martin, & Bakas, 2019; Yang, Liu, Chen, & Tong, 2019). The basic premise
behind federated learning is that the model moves to meet the data
rather than the data moving to meet the model (Figure 1). Therefore, the
minimum data movement needed across the federation is solely the model
parameters and their updates.

![A close up of a logo Description automatically
generated](media/image2.png){width="5.664704724409448in"
height="2.8517180664916886in"}

Figure Diagram of Federated Learning. The data (yellow, red, and blue
disks) does not leave the original owner (a, b, c). Instead the model
(t) and model updates (t+1,a; t+1,b; t+1,c) are passed and each owner
performs training locally. The parameter server sends the model (t) to
each owner and the aggregator combines the model updates (t+1,a; t+1,b;
t+1,c). The aggregator sends this combined model (t+1) back to the
parameter server for another round of training (as the new model t).

How can Intel® SGX Protect Federated Learning?
----------------------------------------------

Intel® Software Guard Extensions (SGX) are a set of CPU instructions
that can be used by developers to set aside private regions of code and
data (Bahmani, et al., 2017). These private regions, called enclaves,
are isolated sections of memory and compute that cannot be accessed
without a cryptographic key. Even users with root access or physical
access to the CPU cannot access the enclave without the authorized key
(Figure 2). This allows for developers to deploy their code and data on
untrusted machines in a secure manner. In 2015, Intel® SGX was launched
as the [first commercial
implementation](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions/details.html)
of what is more formally called a trusted execution environment
([TEE](https://en.wikipedia.org/wiki/Trusted_execution_environment)).

![A black and blue text Description automatically
generated](media/image3.tiff){width="2.80050634295713in"
height="2.7470581802274716in"}

Figure Intel® Software Guard Extensions (SGX) allow developers to create
secure enclaves that are not accessible by the OS or VM without the
proper security keys. This allows for developers to protect code during
use on the CPU.

One path to enable Intel® SGX in an application is to refactor the
application code to use the [Intel SDK for
SGX](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions/sdk.html).
However, many developers are reluctant to change their existing code.
[Graphene](https://grapheneproject.io/) is an open-source library OS
that was created by Intel and its partners to provide developers an easy
way to leverage SGX without the need to change their existing
applications (Figure 3). Several commercial implementations based on
Graphene have been created by our partners, including
[Fortanix](https://fortanix.com/) and
[SContain](https://scontain.com/index.html?lang=en).

With Graphene, the developer simply defines a manifest file that
describes which code and data is allowed within the enclave. This
manifest file is used to automatically create the enclave on an
SGX-compatible CPU. For example, once Graphene is installed and the
manifest file is specified, the command:

  ------------------------------
  \$ SGX=1 ./pal\_loader httpd
  ------------------------------

will use the pal\_loader command to create the enclave from the manifest
and run the web server (http) within the enclave. No other modifications
are needed for the httpd application.

![A screenshot of a cell phone Description automatically
generated](media/image4.tiff){width="2.9865824584426948in"
height="2.6941174540682415in"}

Figure Graphene is an open-sourced project maintained by Intel that
allows developers to run their code within a secure enclave without
needing to modify the code.

What is Intel® Federated Learning?
----------------------------------

![A picture containing clock Description automatically
generated](media/image5.png){width="5.9in"
height="2.9745833333333334in"}

Figure Intel® Federated learning with Intel® SGX allows federated
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
[attestation](https://software.intel.com/content/www/us/en/develop/articles/code-sample-intel-software-guard-extensions-remote-attestation-end-to-end-example.html)
from collaborators which proves that the collaborator actually ran the
expected code within the enclave. Attestation can either be done via a
trusted Intel server or by the developer's own server. This stops
attackers from injecting their own code into the federated training.

![A screenshot of a cell phone Description automatically
generated](media/image6.png){width="6.311765091863517in"
height="3.2192694663167103in"}

Figure 5 Secure Federated Learning with Intel SGX allows researchers to
leverage the benefits of federated learning while mitigating the risks.

Installing and Running the Software
===================================

Accessing the Source Code Repository
------------------------------------

The source code described in this manual should be open-sourced by Intel
for a future public release. Until then, it can only be accessed via a
legal agreement between Intel and the requestor. The development code
currently lives at <https://github.com/IntelLabs/OpenFederatedLearning>.
It is expected to be continually developed and improved. Changes to this
manual, the project code, the project design should be expected.

Design Philosophy
-----------------

The overall design is that all of the scripts are built off of the
**federation plan**. The plan is just a YAML file that defines the
collaborators, aggregator, connections, models, data, and any other
parameters that describes how the training will evolve. In the "Hello
Federation" demos, the plan will be located in the YAML file:
bin/federations/plans/keras\_cnn\_mnist\_2.yaml. As you modify the demo
to meet your needs, you'll effectively just be modifying the plan along
with the Python code defining the **model** and the **data** loader in
order to meet your requirements. Otherwise, the same scripts will apply.
When in doubt, look at the FL plan's YAML file.

### Configure The Federation

TLS encryption is used for the network connections. Therefore, security
keys and certificates will need to be created for the aggregator and
collaborators to negotiate the connection securely. For the "Hello
Federation" demo we will run the aggregator and collaborators on the
same localhost server so these configuration steps just need to be done
once on that machine.

#### Steps:

###### All Nodes

1.  Unzip the source code

  ----------------------------------------
  unzip OpenFederatedLearning-master.zip
  ----------------------------------------

2.  Change into the OpenFederatedLearning-master subdirectory.

  ---------------------------------
  cd OpenFederatedLearning-master
  ---------------------------------

##### On the Aggregator Node

1.  Change the directory to bin/federations/pki:

  ------------------------
  cd bin/federations/pki
  ------------------------

2.  Run the Certificate Authority script. This will setup the Aggregator
    node as the Certificate Authority for the Federation. All
    certificates will be signed by the aggregator. Follow the
    command-line instructions and enter in the information as prompted.
    The script will create a simple database file to keep track of all
    issued certificates.

  -------------------
  bash setup\_ca.sh
  -------------------

![A screenshot of a social media post Description automatically
generated](media/image7.png){width="4.402643263342083in"
height="5.0235301837270345in"}

3.  Run the aggregator cert script, replacing
    AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME with the actual
    [FQDN](https://en.wikipedia.org/wiki/Fully_qualified_domain_name)
    for the aggregator machine. You may optionally include the IP
    address for the aggregator, replacing \[IP\_ADDRESS\].

  ------------------------------------------------------------------
  bash create-aggregator.sh AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME
  ------------------------------------------------------------------

*Tip: You can discover the
[FQDN](https://en.wikipedia.org/wiki/Fully_qualified_domain_name) with
the Linux command:* hostname --fqdn

![A screenshot of a social media post Description automatically
generated](media/image8.png){width="5.394114173228346in"
height="4.33200678040245in"}

4.  **For each test machine you want to run collaborators on**, we
    create a collaborator certificate, replacing TEST.MACHINE.NAME with
    the actual test machine name. Note that this does not have to be the
    FQDN. Also, note that this script is run on the Aggregator node
    because it is the Aggregator that signs the certificate. Only
    Collaborators with valid certificates signed by the Aggregator can
    join the federation.

  -----------------------------------------------
  bash create-collaborator.sh TEST.MACHINE.NAME
  -----------------------------------------------

![A screenshot of a social media post Description automatically
generated](media/image9.png){width="4.436975065616798in"
height="3.4941174540682414in"}

5.  Once you have the certificates created, you need to move the
    certificates to the correct machines and ensure each machine has the
    cert\_chain.crt needed to verify cert signatures. For example, on a
    test machine named TEST\_MACHINE that you want to be able to run as
    a collaborator, you should have:

+-------------------------------------------------------------------+
| -   bin/federations/pki/cert\_chain.crt                           |
|                                                                   |
| -   bin/federations/pki/col\_TEST\_MACHINE/col\_TEST\_MACHINE.crt |
|                                                                   |
| -   bin/federations/pki/col\_TEST\_MACHINE/col\_TEST\_MACHINE.key |
+-------------------------------------------------------------------+

Note that once the certificates are transferred to the collaborator, it
is now possible to participate in any future federations run by this
aggregator. (The aggregator can revoke this privilege.)

6.  On the aggregator machine you should have the files:

+-----------------------------------------------------------------------+
| -   bin/federations/pki/cert\_chain.crt                               |
|                                                                       |
| -   bin/federations/pki/                                              |
|     agg\_AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME/agg\_AGGREGATOR.FULLY |
| .QUALIFIED.DOMAIN.NAME.crt                                            |
|                                                                       |
| -   bin/federations/pki/                                              |
|     agg\_AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME/agg\_AGGREGATOR.FULLY |
| .QUALIFIED.DOMAIN.NAME.key                                            |
+-----------------------------------------------------------------------+

Baremetal Installation
----------------------

Intel has tested the installation on Ubuntu 18.04 and Centos 7.6
systems. A Python 3.6 virtual environment
([venv](https://docs.python.org/3/library/venv.html)) is used to isolate
the packages. The basic installation is via the Makefile included in the
root directory of the repository.

### Installation Steps

**NOTE: Steps 1-2 may have already been completed.**

1.  Unzip the source code

  ----------------------------------------
  unzip OpenFederatedLearning-master.zip
  ----------------------------------------

2.  Change into the OpenFederatedLearning-master subdirectory.

  ---------------------------------
  cd OpenFederatedLearning-master
  ---------------------------------

3.  Use a text editor to open the YAML file for the federation plan.

  ----------------------------------------------------
  vi bin/federations/plans/keras\_cnn\_mnist\_2.yaml
  ----------------------------------------------------

This YAML file defines the IP addresses for the aggregator. It is the
main file that controls all of the execution of the federation. By
default, the YAML file is defining a federation where the aggregator
runs on the localhost at port 5050(it us up to the developer to make
sure that the port chosen is open and accessible to all participants).
For this demo, we'll just focus on running everything on the same
server. You'll need to edit the YAML file and replace localhost with the
aggregator address. Please make sure you specify the fully-qualified
domain name
([FQDN](https://en.wikipedia.org/wiki/Fully_qualified_domain_name))
address (required for security). For example:

![A screenshot of a cell phone Description automatically
generated](media/image10.png){width="6.017299868766404in"
height="2.152985564304462in"}

You can discover the FQDN by running the Linux command: hostname \--fqdn

4.  If pyyaml is not installed, then use pip to install it:

  ---------------------
  pip3 install pyyaml
  ---------------------

5.  Make sure that you followed the steps in **Configure the
    Federation** and have copied the keys and certificates onto the
    federation nodes.

6.  Build the virtual environment using the command:

  --------------
  make install
  --------------

This should create a Python 3 virtual environment with the required
packages (e.g. TensorFlow, PyTorch, nibabel) that are used by the
aggregator and the collaborators. Note that you can add custom Python
packages by editing this section in the Makefile:

![A screenshot of a cell phone Description automatically
generated](media/image11.png){width="2.8581310148731407in"
height="1.6635947069116361in"}

Just add your own line. For example, venv/bin/pip3 install my\_package

### Running the Baremetal Demo -- "Hello Federation"

This is a quick tutorial on how to run a default federation using the
Intel® Federated Learning solution. The default federation will simply
train a TF/Keras CNN model on the MNIST dataset. We'll define one
aggregator and two collaborator nodes. We've tested this tutorial on
both Ubuntu 18.04 and CentOS 7.6 Linux machines with a Python 3.6
virtual environment
([venv](https://docs.python.org/3/library/venv.html)). For demonstration
purposes, we'll run the aggregator and the 2 collaborators on the same
server; however, we point out in this tutorial how to easily change the
code to run the aggregator and collaborators on separate nodes.

### Steps to Run the Baremetal Federation

#### On All Nodes

1.  Install Python 3.6 and the Python virtual environment. You can find
    the instructions on the official [Python
    website](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv).
    You may need to log out and back in for the changes to take effect.

  -------------------------------------------
  python3 -m pip install \--user virtualenv
  -------------------------------------------

#### On the Aggregator

1.  Follow the Baremetal Installation steps as described previously.

2.  It is assumed that the federation may be fine-tuning a previously
    trained model. For this reason, the pre-trained weights for the
    model will be stored within protobuf files on the aggregator and
    passed to the collaborators during initialization. As seen in the
    YAML file, the protobuf file with the initial weights is expected to
    be found in the file keras\_cnn\_mnist\_init.pbuf. For this example,
    however, we'll just create an initial set of random model weights
    and putting it into that file by running the command:

  ------------------------------------------------------------------------------------------------------------------------------------
  ./venv/bin/python3 ./bin/create\_initial\_weights\_file\_from\_flplan.py -p keras\_cnn\_mnist\_2.yaml -dc local\_data\_config.yaml
  ------------------------------------------------------------------------------------------------------------------------------------

3.  Now we're ready to start the aggregator by running the Python
    script. Note that we will need to pass in the fully-qualified domain
    name (FQDN) for the aggregator node address in order to present the
    correct certificate.

  ------------------------------------------------------------------------------------------------------------------------------------
  ./venv/bin/python3 ./bin/run\_aggregator\_from\_flplan.py -p keras\_cnn\_mnist\_2.yaml -ccn AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME
  ------------------------------------------------------------------------------------------------------------------------------------

At this point, the aggregator is running and waiting for the
collaborators to connect. When all of the collaborators connect, the
aggregator starts training. When the last round of training is complete,
the aggregator stores the final weights in the protobuf file that was
specified in the YAML file (in this case
keras\_cnn\_mnist\_latest.pbuf).

#### On the Collaborator

**NOTE: If the demo is performed using the same node for the aggregator
and the collaborators, then steps 1-5 are already completed. You may
skip directly to step 6. **

1.  Unzip the source code

  ----------------------------------------
  unzip OpenFederatedLearning-master.zip
  ----------------------------------------

2.  Change into the OpenFederatedLearning-master subdirectory.

  ---------------------------------
  cd OpenFederatedLearning-master
  ---------------------------------

3.  Make sure that you followed the steps in **Configure the
    Federation** and have copied the keys and certificates onto the
    federation nodes.

4.  Copy the plan file (keras\_cnn\_mnist\_2.yaml) from the aggregator
    over to the collaborator to the plan subdirectory
    (bin/federations/plans)

5.  Build the virtual environment using the command:

  --------------
  make install
  --------------

6.  Now run the collaborator col\_1 using the Python script. Again, you
    will need to pass in the fully qualified domain name in order to
    present the correct certificate.

  ----------------------------------------------------------------------------------------------------------------------------------------------------
  ./venv/bin/python3 ./bin/run\_collaborator\_from\_flplan.py -p keras\_cnn\_mnist\_2.yaml -col col\_1 -ccn COLLABORATOR.FULLY.QUALIFIED.DOMAIN.NAME
  ----------------------------------------------------------------------------------------------------------------------------------------------------

7.  Repeat this for each collaborator in the federation. Once all
    collaborators have joined, the aggregator will start and you will
    see log messages describing the progress of the federated training.

![A screenshot of a cell phone Description automatically
generated](media/image12.png){width="5.776469816272966in"
height="2.1513648293963255in"}

Docker Installation
-------------------

We will show you how to set up Intel^®^ Federated Learning on Docker
using a simple MNIST dataset and a TensorFlow/Keras CNN model as an
example. You will note that this is literally the same code as the
Baremetal Installation, but we are simply wrapping the venv within a
Docker container.

Before we start the tutorial, please make sure you have Docker installed
and configured properly. Here is an easy test to run in order to test
some basic functionality:

+-----------------------------------------------------------------------+
| \$ docker run hello-world                                             |
|                                                                       |
| Hello from Docker!                                                    |
|                                                                       |
| This message shows that your installation appears to be working       |
| correctly.                                                            |
|                                                                       |
| \...                                                                  |
|                                                                       |
| \...                                                                  |
|                                                                       |
| \...                                                                  |
+-----------------------------------------------------------------------+

### Installation Steps

**NOTE: Steps 1-2 may have already been performed.**

1.  Unzip the source code

  ----------------------------------------
  unzip OpenFederatedLearning-master.zip
  ----------------------------------------

2.  Change into the OpenFederatedLearning-master subdirectory.

  ---------------------------------
  cd OpenFederatedLearning-master
  ---------------------------------

3.  Use a text editor to open the YAML file for the federation plan.

  ----------------------------------------------------
  vi bin/federations/plans/keras\_cnn\_mnist\_2.yaml
  ----------------------------------------------------

This YAML file defines the IP addresses for the aggregator and
collaborators. It is the main file that controls all of the execution of
the federation. By default, the YAML file is defining a federation where
the aggregator runs on the localhost at port 5050 and there are two
collaborators(it us up to the developer to make sure that the port
chosen is open and accessible to all participants). For this demo, we'll
just focus on running everything on the same server. You'll need to edit
the YAML file and replace localhost with the aggregator address. Please
make sure you specify the fully-qualified domain name
([FQDN](https://en.wikipedia.org/wiki/Fully_qualified_domain_name))
address (required for security). For example:

![A screenshot of a cell phone Description automatically
generated](media/image10.png){width="6.017299868766404in"
height="2.152985564304462in"}

You can discover the FQDN by running the Linux command: hostname --fqdn

4.  If pyyaml is not installed, then use pip to install it:

  ---------------------
  pip3 install pyyaml
  ---------------------

5.  Make sure that you followed the steps in **Configure the
    Federation** and have copied the keys and certificates onto the
    federation nodes.

6.  Build the Docker containers using the command:

  -----------------------------------------------
  make build\_containers model\_name=keras\_cnn
  -----------------------------------------------

This should create the Docker containers that are used by the aggregator
and the collaborators.

+-----------------------------------------------------------+
| Successfully tagged tfl\_agg\_keras\_cnn\_bduser:0.1      |
|                                                           |
| Successfully tagged tfl\_col\_cpu\_keras\_cnn\_bduser:0.1 |
+-----------------------------------------------------------+

### Running the Docker Demo -- "Hello Federation"

This is a quick tutorial on how to run a default federation using the
Intel® Federated Learning solution. The default federation will simply
train a TF/Keras CNN model on the MNIST dataset. We'll define one
aggregator and two collaborator nodes. All 3 nodes will use Docker to
run their code. We've tested this tutorial on Ubuntu 18.04 and CentOS
7.6 Linux machines with Docker 18.06.1 and Python 3.6. For demonstration
purposes, we'll run the aggregator and the 2 collaborators on the same
server; however, we point out in this tutorial how to easily change the
code to run the aggregator and collaborators on separate nodes.

### Steps to Run the Docker Federation

#### On the Aggregator

1.  Follow the Docker Installation steps as described previously.

2.  Run the Docker container for the aggregator:

  -------------------------------------------------
  make run\_agg\_container model\_name=keras\_cnn
  -------------------------------------------------

![](media/image13.png){width="4.643805774278215in"
height="0.41176509186351706in"}

When the Docker container for the aggregator begins you'll see the
prompt above. This means you are within the running Docker container.
You can always exit back to the original Linux shell by typing \`exit\`.

3.  It is assumed that the federation may be fine-tuning a previously
    trained model. For this reason, the pre-trained weights for the
    model will be stored within protobuf files on the aggregator and
    passed to the collaborators during initialization. As seen in the
    YAML file, the protobuf file with the initial weights is expected to
    be found in the file keras\_cnn\_mnist\_init.pbuf. For this example,
    however, we'll just create an initial set of random model weights
    and putting it into that file by running the command:

  --------------------------------------------------------------------------------------------------------------
  ./create\_initial\_weights\_file\_from\_flplan.py -p keras\_cnn\_mnist\_2.yaml -dc docker\_data\_config.yaml
  --------------------------------------------------------------------------------------------------------------

4.  Now we're ready to start the aggregator by running the Python
    script:

  -------------------------------------------------------------------------------------------------------------------
  python3 run\_aggregator\_from\_flplan.py -p keras\_cnn\_mnist\_2.yaml -ccn AGGREGATOR.FULLY.QUALIFIED.DOMAIN.NAME
  -------------------------------------------------------------------------------------------------------------------

Notice we have to pass the fully qualified domain name (FQDN) so that
the correct certificate can be presented. At this point, the aggregator
is running and waiting for the collaborators to connect. When all of the
collaborators connect, the aggregator starts training. When the last
round of training is complete, the aggregator stores the final weights
in the protobuf file that was specified in the YAML file (in this case
keras\_cnn\_mnist\_latest.pbuf).

#### On the Collaborators

**NOTE: If the demo is performed using the same node for the aggregator
and the collaborators, then steps 1-5 are already completed. You may
skip directly to step 6.**

1.  Unzip the source code

  ----------------------------------------
  unzip OpenFederatedLearning-master.zip
  ----------------------------------------

2.  Change into the OpenFederatedLearning-master subdirectory.

  ---------------------------------
  cd OpenFederatedLearning-master
  ---------------------------------

3.  Make sure that you followed the steps in **Configure the
    Federation** and have copied the keys and certificates onto the
    federation nodes.

4.  Copy the plan file (keras\_cnn\_mnist\_2.yaml) from the aggregator
    over to the collaborator to the plan subdirectory
    (bin/federations/plans)

5.  Either transfer the Docker image over from the aggregator or build
    the Docker container using the command:

  -----------------------------------------------
  make build\_containers model\_name=keras\_cnn
  -----------------------------------------------

6.  Now run the Docker on the collaborator. For collaborator \#1, run
    this command:

  ------------------------------------------------------------------
  make run\_col\_container model\_name=keras\_cnn col\_name=col\_1
  ------------------------------------------------------------------

![](media/image14.png){width="6.2590201224846895in"
height="0.41217957130358707in"}

7.  Now run the collaborator Python script to start the collaborator.
    Notice that you'll need to specify the fully qualified domain name
    (FQDN) for the collaborator node to present the correct certificate.

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------
  python3 run\_collaborator\_from\_flplan.py -p keras\_cnn\_mnist\_2.yaml -col col\_1 -dc docker\_data\_config.yaml -ccn COLLABORATOR.FULLY.QUALIFIED.DOMAIN.NAME
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------

8.  Repeat this for each collaborator in the federation. Once all
    collaborators have joined, the aggregator will start and you will
    see log messages describing the progress of the federated training.

Federated Training of the 2D U-Net (Brain Tumor Segmentation)
-------------------------------------------------------------

This tutorial assumes that you\'ve run the previous MNIST demos. We'll
provide fewer details about the installation and highlight the specific
differences needed for training a 2D U-Net. We also assume that you've
downloaded the Brain Tumor Segmentation
([BraTS](http://braintumorsegmentation.org/)) dataset and have the data
accessible to the training nodes.

1.  In the "Hello Federation" demo the MNIST data was downloaded during
    runtime from an online repository. However, in this example we are
    allocating data directly. To make this work, we create a \<Brats
    Symlinks Dir\>, which is has directories of symlinks to the data for
    each institution number. Setting this up is out-of-scope for this
    code at the moment, so we leave this to the reader. In the end, our
    directory looks like below. Note that \"0-9\" allows us to do
    data-sharing training (i.e. as if the data were centrally-pooled
    rather than federated at different collaborator sites).

+-------------------------------------------------------+
| \$ ll \<Brats Symlinks Dir\>                          |
|                                                       |
| \...                                                  |
|                                                       |
| drwxr-xr-x 90 \<user\> \<group\> 4.0K Nov 25 22:14 0  |
|                                                       |
| drwxr-xr-x 212 \<user\> \<group\> 12K Nov 2 16:38 0-9 |
|                                                       |
| drwxr-xr-x 24 \<user\> \<group\> 4.0K Nov 25 22:14 1  |
|                                                       |
| drwxr-xr-x 36 \<user\> \<group\> 4.0K Nov 25 22:14 2  |
|                                                       |
| drwxr-xr-x 14 \<user\> \<group\> 4.0K Nov 25 22:14 3  |
|                                                       |
| drwxr-xr-x 10 \<user\> \<group\> 4.0K Nov 25 22:14 4  |
|                                                       |
| drwxr-xr-x 6 \<user\> \<group\> 4.0K Nov 25 22:14 5   |
|                                                       |
| drwxr-xr-x 10 \<user\> \<group\> 4.0K Nov 25 22:14 6  |
|                                                       |
| drwxr-xr-x 16 \<user\> \<group\> 4.0K Nov 25 22:14 7  |
|                                                       |
| drwxr-xr-x 17 \<user\> \<group\> 4.0K Nov 25 22:14 8  |
|                                                       |
| drwxr-xr-x 7 \<user\> \<group\> 4.0K Nov 25 22:14 9   |
|                                                       |
| \...                                                  |
+-------------------------------------------------------+

2.  For this demo, we'll again consider only 2 collaborators. Recall
    that the design philosophy states that everything gets specified in
    the **federation plan**. For the BraTS demo, the FL plan is called
    bin/federations/plans/brats17\_insts2\_3.yaml. Edit that to file
    specify the correct addresses for your machines.

  ---------------------------------------------------------------
  \$ vi bin/federations/plans/tf\_2dunet\_brats\_insts2\_3.yaml
  ---------------------------------------------------------------

Find the keys for the address (\"agg\_addr\") and port (\"agg\_port\")
and replace them with your values.

+--------------------------------------------------------------------+
| federation:                                                        |
|                                                                    |
| fed\_id: &fed\_id \'fed\_0\'                                       |
|                                                                    |
| opt\_treatment: &opt\_treatment \'AGG\'                            |
|                                                                    |
| polling\_interval: &polling\_interval 4                            |
|                                                                    |
| rounds\_to\_train: &rounds\_to\_train 50                           |
|                                                                    |
| agg\_id: &agg\_id \'agg\_0\'                                       |
|                                                                    |
| agg\_addr: &agg\_addr \"agg.domain.com\" **\# CHANGE THIS STRING** |
|                                                                    |
| agg\_port: &agg\_port \<some\_port\> **\# CHANGE THIS INT**        |
+--------------------------------------------------------------------+

3.  Edit the docker data config file to refer to the correct username
    (the name of the account you are using. Open
    bin/federations/docker\_data\_config.yaml and replace the username
    with your username.

+-----------------------------------------------------------------------+
| \$ vi bin/federations/docker\_data\_config.yaml                       |
|                                                                       |
| collaborators:                                                        |
|                                                                       |
| > col\_one\_big:                                                      |
| >                                                                     |
| > brats: &brats\_data\_path \'/home/\<USERNAME\>/tfl/datasets/brats\' |
| > \# replace with your username                                       |
| >                                                                     |
| > col\_0:                                                             |
| >                                                                     |
| > brats: [\*](#id5)brats\_data\_path mnist\_shard: 0                  |
| >                                                                     |
| > col\_1:                                                             |
| >                                                                     |
| > brats: [\*](#id7)brats\_data\_path mnist\_shard: 1                  |
|                                                                       |
| \...                                                                  |
+-----------------------------------------------------------------------+

### Start an Aggregator

1.  Build the docker images
    \"tfl\_agg\_\<model\_name\>\_\<username\>:0.1\" and
    \"tfl\_col\_\<model\_name\>\_\<username\>:0.1\" using project folder
    Makefile targets. This uses the project folder \"Dockerfile\". We
    only build them once, unless we change Dockerfile. We pass along the
    proxy configuration from the host machine to the docker container,
    so that your container would be able to access the Internet from
    typical corporate networks. We also create a container user with the
    same UID so that it is easier to access the mapped local volume from
    the docker container. Note that we include the username to avoid
    development-time collisions on shared development servers. We build
    the collaborator Docker image upon the aggregator image, adding
    necessary dependencies such as the mainstream deep learning
    frameworks. You may modify ./models/\<model\_name\>/Dockerfile to
    install the needed packages.

  --------------------------------------------------
  \$ make build\_containers model\_name=tf\_2dunet
  --------------------------------------------------

2.  Run the aggregator container (entering a bash shell inside the
    container), again using the Makefile. Note that we map the local
    volumes ./bin/federations to the container.

  ------------------------------------------------------------------
  \$ make run\_agg\_container model\_name=tf\_2dunet dataset=brats
  ------------------------------------------------------------------

3.  In the aggregator container shell, build the initial weights files
    providing the global model initialization that will be sent from the
    aggregator out to all collaborators.

  -------------------------------------------------------------------------------------------------------------------------
  \$ ./create\_initial\_weights\_file\_from\_flplan.py -p tf\_2dunet\_brats\_insts2\_3.yaml -dc docker\_data\_config.yaml
  -------------------------------------------------------------------------------------------------------------------------

4.  In the aggregator container shell, run the aggregator, using a shell
    script provided in the project.

+-----------------------------------------------------------------------+
| \$ ./run\_brats\_aggregator.sh                                        |
|                                                                       |
| Loaded logging configuration: logging.yaml                            |
|                                                                       |
| 2020-01-15 23:17:18,143 - tfedlrn.aggregator.aggregatorgrpcserver -   |
| DEBUG - Starting aggregator.                                          |
+-----------------------------------------------------------------------+

### Start Collaborators

Note: the collaborator machines can be the same as the aggregator
machine.

1.  (**On each collaborator machine**) Enter the project folder and
    build the containers as above.

  --------------------------------------------------
  \$ make build\_containers model\_name=tf\_2dunet
  --------------------------------------------------

2.  (**On the first collaborator machine**) Run the first collaborator
    container. Note we are using collaborators 2 and 3.

  ------------------------------------------------------------------------------
  \$ make run\_col\_container model\_name=tf\_2dunet dataset=brats col\_name=2
  ------------------------------------------------------------------------------

3.  In this first collaborator shell, run the collaborator using the
    provided shell script.

  ------------------------------------
  \$ ./run\_brats\_collaborator.sh 2
  ------------------------------------

4.  (**On the second collaborator machine, which could be a second
    terminal on the first machine**) Run the second collaborator
    container (entering a bash shell inside the container).

+-----------------------------------------------------------------------+
| \$ make run\_col\_container model\_name=tf\_2dunet dataset=brats      |
| col\_name=col\_3                                                      |
|                                                                       |
| docker run \\                                                         |
|                                                                       |
| \...                                                                  |
|                                                                       |
| bash                                                                  |
+-----------------------------------------------------------------------+

5.  In the second collaborator container shell, run the second
    collaborator.

+------------------------------------+
| \$ ./run\_brats\_collaborator.sh 3 |
|                                    |
| \...                               |
|                                    |
| \...                               |
|                                    |
| \...                               |
+------------------------------------+

Porting your Experiments to Intel® Federated Learning
=====================================================

Design Philosophy
-----------------

Intel® Federated Learning is completely agnostic to deep learning
frameworks. The code works equally-well with TensorFlow, PyTorch, or
other frameworks because the code only actually passes the model and
optimizer weight values around the network. The graph and framework code
is never passed.

![A close up of a device Description automatically
generated](media/image15.png){width="2.9960159667541557in"
height="1.6852591863517061in"}

Figure The code remains framework agnostic because it only passes
protobufs across the federation.

To accomplish this, we'll define methods for each model's Python class,
to define how to take the weights from the computational graph and
convert the set of weights into a numpy array valued dictionary (and
vise-versa). As we demonstrate in our example model code, these methods
can be inherited from framework specific model classes to avoid
duplication for each model utilizing the same ML framework. The
collaborator object classes in our code (who have the model class as an
attribute) take care of further converting these arrays to a generic,
protobuf object (and vice versa, Figure 5). Once these methods of the
model classes are defined, the collaborators know how to take their
framework-specific model, convert the weights to the generic protobuf,
transfer the protobuf to the aggregator, receive the consensus protobuf
results back from the aggregator, and finally load new weights into the
local model. Figure 6 shows how the existing Keras model class example
code takes a dictionary of numpy weights (tensor\_dict) and writes the
weights into the model compute graph . The tensor\_dict contains the
names and values of the model and optimizer weights. The methods
set\_tensor\_dict and get\_tensor\_dict are used to pass tensors to and
from the model's compute graph. These methods have been pre-defined for
TensorFlow, Keras, and PyTorch . Support for new frameworks can be
created similarly, and the Intel® Federated Learning code enables
conversion of these dictionaries to and from protobuf independent of
framework. The Federation Plan's YAML file defines the protobuf
filenames for the initial, latest, and best models (e.g.
keras\_cnn\_mnist\_best.pbuf).

![A screenshot of a social media post Description automatically
generated](media/image16.png){width="5.354580052493438in"
height="2.4313003062117233in"}

Figure 6 Method defined to write the values of a dictionary of model and
optimizer weights into the compute graph of a Keras model.. Similar
methodsexist for TensorFlow and PyTorch. An analogous method exists to
read the weights off of the graph and create the tensor\_dict of weight
names to numpy array values.

Repository Structure
--------------------

Figure 7 shows that the Intel® Federated Learning code repository is
divided into 4 major sections: bin, data, models, and tfedlrn.

The tfedlrn subdirectory contains the code that orchestrates the
aggregator and collaborator nodes. It is not anticipated that the
developer will need to modify this code unless the aggregation algorithm
or communication protocols need to be modified.

In the previous "Hello Federation" demos, we showed how the plans and
weights files contained within the bin/federations subdirectory were
used to describe the federation architecture. Specifically, the
federation plan is a YAML file located in the bin/federations/plans
subdirectory.

The Federation Plan
-------------------

![](media/image17.png){width="3.0194444444444444in"
height="4.010416666666667in"}

Figure 8 shows the Federation Plan from the "Hello Federation" demo. The
first section of the YAML file (federation, line 37) defines the
federation label (line 38), number of training rounds (line 41), the
aggregator's fully defined domain name
([FQDN](https://en.wikipedia.org/wiki/Fully_qualified_domain_name))
address and port (lines 43-44). As noted in the documentation for these
demos, this FQDN should exactly match the security certificate (cf.
section **Configuring the Federation**). Beginning on line 45, the
collaborator names should be specified. Note that like the federation
and aggregator names (lines 38 and 42), these may be any label chosen by
the developer. Lines 48-50 define the filenames for the initial, latest,
and best model protobuf files. Finally, beginning at line 57, the
developer can describe the network addresses of the approved
collaborators for testing on this specific federation (aka the
whitelist). Note that these addresses must exactly match the common name
provided in the security certificates that were generated for these
collaborators (cf. section **Configuring the Federation**). By
specifically whitelisting collaborators, the Federation can include
collaborators within some federations, but easily exclude them from
other federations. This granularity of access is important for securing
access to the federation. Lines 64-65 provide switches to disable secure
connections and client authentication. These should only be changed when
debugging network connections (and only then with extreme caution as
they eliminate any security in the networking).

![](media/image18.png){width="2.513959973753281in"
height="1.4294116360454943in"}The remaining two subdirectories---models
and data---are the most important parts of the repository to the
individual developer. These two subdirectories contain the Python
scripts that define the model and data pre-processing/loading (in
whatever deep learning framework is desired). The Federation Plan
(Figure 9) has sections to define which specific Python Model and Data
class to use for the federation. Note that a subset of user-defined
class parameters (e.g. nb\_collaborators, batch\_size) may be defined
here and will be passed to the class constructors at runtime (in
addition to other parameters - for example the data\_path that is
brought in from the local data config) via \*\*kwargs parameters in the
class initializer.

Models
------

Because the code is agnostic to the framework, the developer needs to
define what to do when the federation starts training. At the
collaborator nodes, the training is expected to be using a deep learning
framework to train locally. The FLModel class defines the interface
between the model object and the rest of the code via the collaborator
object (Figure 10). This parent class is ultimately inherited by all
models.

![](media/image20.png){width="5.0in" height="2.5in"}

Figure 10 The pre-defined class FLModel.

Figure 11 shows an example DummyModel class that inherits the FLModel
parent. This class simply waits for a random number of seconds when the
train\_batches method is called and then returns a random value for the
loss metric. Similar code is used for the validate method. This Dummy
class is useful for sanity-checking the FL pipeline without having to
define a model or dataset.

![](media/image22.png){width="5.0in" height="1.5833333333333333in"}

Figure 11 A Dummy Model class. This inherits the FLModel class, but the
train\_batches method simply waits a random number of seconds and then
returns a random loss value.

TensorFlow / Keras / PyTorch
----------------------------

There are pre-defined model classes for TensorFlow, Keras, and PyTorch
in the repository under the models subdirectory. Note that
commonly-used, framework-specific methods, such as, train\_batches,
validate, get\_tensor\_dict, set\_tensor\_dict, reset\_opt\_vars, and
initialize\_globals methods are already defined in framework dependent
base classes (or examples provided) for all of these frameworks. The
only attributes that remain to be developedwhen inheriting from these
classes are those providing specific model architecture (self.model) and
the data object for serving up batches for training and validation.

For example, Figure 12 shows the class that should be inherited by all
models using the Keras DL framework. The developer only needs to inherit
from KerasFLModel, and override the self.model attribute, replacing with
a specific keras.Model object that defines their specific model topology
(line 21).

![](media/image23.png){width="5.0in" height="2.3020833333333335in"}

Figure 12 The base class for all Keras Models. New Keras models can be
created by inheriting the KerasFLModel class and overriding the
self.model instantiation with the custom Keras model.

In most cases, all other framework-specific methods should be correctly
defined. For example, Figure 13 demonstrates that the train\_batches
method for Keras simply calls the Keras fit method for a single epoch
and returns the loss value as a float.

![](media/image24.png){width="5.0in" height="1.875in"}

Figure 13 The pre-defined train\_epoch method for Keras models.

The FL Plan defines where to find the model-specific code for the
federation. In Figure 9, the FL plan specifies that the model definition
exists in the subdirectory models/tensorflow/keras\_cnn and the class
name is KerasCNN. Additional parameters to the Model class are passed at
runtime via the \*\*kwargs parameter. Figure 14 shows the code for the
custom KerasCNN Model class. This class inherits the parent class
KerasFLModel (line 13) and defines the self.model object using the
custom code in the method build\_model (line 20). Other than the initial
model definition, all other methods are pre-defined by the parent class.
Similar Keras models can be created by inheriting the KerasFLModel class
and defining the model topology using the Keras Model API.

### 

![A screenshot of a cell phone Description automatically
generated](media/image25.png){width="6.5in" height="5.64375in"}

Figure 14 A user-defined Keras model. Note that the only change the
developer needs to make is to define the Keras model. All other methods
are pre-defined and inherited from the KerasFLModel parent class.

### 

Similar templates and demos exist for PyTorch and generic (i.e.
non-Keras API) TensorFlow code. Figure 15 shows the pre-defined code for
the PyTorch train\_batches method. For these frameworks, some
customization of the train\_epoch method may be needed because PyTorch
(and generic TensorFlow) are not as rigidly templated as the Keras API.
Nevertheless, all FLModel methods assume that train\_epoch will perform
one epochs of training and return the value of the loss for the model.

![](media/image26.png){width="5.0in" height="2.2395833333333335in"}

Figure 15 A PyTorch method for train\_batches from the pt\_cnn.py custom
model class.

Data
----

![](media/image27.png){width="3.417361111111111in" height="1.3in"}The
Data subdirectory defines the Python classes for loading the data and
serving it to the model (Figure 16). The primary methods for this class
are get\_train\_loader and get\_val\_loader. These Python methods should
be modified by the developer to define data loader objects that can be
used as iterators to serve up batches of data to the model for training
and validation.

Figure 9 shows the Data definitions within the FL Plan for the "Hello
Federation" demo. The **Data** object within the YAML file defines the
path to the Python class code (*code.path*), the class name to use for
the data loader (*class\_name*), and some user-defined parameters such
as the number of collaborators (*nb\_collaborators*) and batch size
(*batch\_size*) . Number of collaborators is used in this case since the
MNIST dataset is downloaded and sharded (partitioned) according to how
many collaborators there are. The specific shard number to be used for
each collaborator is looked up (using the collaborator name) in the
local data config and passed to the data object constructor at runtime.

![A screenshot of a cell phone Description automatically
generated](media/image28.png){width="6.5in"
height="2.7354166666666666in"}

Figure 17 The custom Data class used for the \"Hello Federation\" demo.

Figure 17 shows the custom data loader defined by the Data class for the
"Hello Federation" demo. It simply loads the particular partition of
MNIST, and splits it into a training and validation set to populate the
in-memory variables self.X\_train, self.y\_train, self.X\_val, and
self.y\_val. Out-of-memory data loaders (e.g.
[tf.data](https://www.tensorflow.org/guide/data),
[torch.utils.data](https://pytorch.org/docs/stable/data.html)) may be
used but are not currently implemented in the code and are out of the
scope for this manual. Nevertheless, any Python method that returns
NumPy array producing data loaders is allowed.

### Local Data Config

Like the model code, the code for the data object may be shared among
many collaborators in a federation. We therefore allow the data file
locations to be independent of the data object code, instead passing
this information to the data object constructor at runtime. In order to
do this, we utilize a local data configuration file. This YAML file (see
Figure 18 for an example) is located in the bin/federations directory,
and is used by executables to look up the data path (or other relevant
information) for a particular collaborator and data type. At present
there is only one top level key, "collaborators", for this file. In
deployment we expect that possibly each collaborator will only have
their own collaborator name populated in their file, but for
experimentation we have many (simulated) collaborators listed. For each
collaborator name listed under this key, there are multiple dataset
names as keys, the value for which is a datapath (or other information)
to be used for that dataset. For large unpartitioned public datasets
used for experimentation, it may make sense to hard code the location of
the data in the data object, in which case the value under a specific
data name can be used to communicate the shard number being used for
that collaborator. This is what we do for MNIST and CIFAR10 in our
example code.

![](media/image29.png){width="5.0in" height="2.71875in"}

Figure 1 The local data configuration YAML file containing information
needed when looking up data for a particular collaborator.

Running the simulator 
======================

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

![A picture containing screenshot Description automatically
generated](media/image30.png){width="5.864704724409449in"
height="3.068319116360455in"}

Simulated Federated Training of an MNIST Classifier across 10 Collaborators
---------------------------------------------------------------------------

The plan we will use for this tutorial is keras\_cnn\_mnist\_10.yaml.

### Create the project virtual environment

**NOTE: Steps 1-2 may have already been performed.**

1.  To prepare, make sure you have python 3.6 (or higher) with
    virtualenv installed.

2.  Unzip the source code

  ----------------------------------------
  unzip OpenFederatedLearning-master.zip
  ----------------------------------------

3.  Change into the OpenFederatedLearning-master subdirectory.

  ---------------------------------
  cd OpenFederatedLearning-master
  ---------------------------------

4.  Create the virtual environment, and change to the bin directory.

+-----------------+
| \$ make clean   |
|                 |
| \$ make install |
|                 |
| \$ cd bin       |
+-----------------+

5.  Create the initial weights file for the model to be trained.

  -----------------------------------------------------------------------------------------------------
  \$ ../venv/bin/python create\_initial\_weights\_file\_from\_flplan.py -p keras\_cnn\_mnist\_10.yaml
  -----------------------------------------------------------------------------------------------------

4.  Start the simulation.

  --------------------------------------------------------------------------------------
  \$ ../venv/bin/python run\_simulation\_from\_flplan.py -p keras\_cnn\_mnist\_10.yaml
  --------------------------------------------------------------------------------------

5.  You\'ll find the output from the aggregator in
    bin/logs/aggregator.log. Grep this file to see results (one example
    below). You can check the progress as the simulation runs, if
    desired.

+-----------------------------------------------------------------------+
| \$ grep -A 2 \"round results\" logs/aggregator.log                    |
|                                                                       |
| 2020-03-30 13:45:33,404 - tfedlrn.aggregator.aggregator - INFO -      |
| round results for model id/version KerasCNN/1                         |
|                                                                       |
| 2020-03-30 13:45:33,404 - tfedlrn.aggregator.aggregator - INFO -      |
| validation: 0.4465000107884407                                        |
|                                                                       |
| 2020-03-30 13:45:33,404 - tfedlrn.aggregator.aggregator - INFO -      |
| loss: 1.0632034242153168                                              |
|                                                                       |
| \--                                                                   |
|                                                                       |
| 2020-03-30 13:45:35,127 - tfedlrn.aggregator.aggregator - INFO -      |
| round results for model id/version KerasCNN/2                         |
|                                                                       |
| 2020-03-30 13:45:35,127 - tfedlrn.aggregator.aggregator - INFO -      |
| validation: 0.8630000054836273                                        |
|                                                                       |
| 2020-03-30 13:45:35,127 - tfedlrn.aggregator.aggregator - INFO -      |
| loss: 0.41314733028411865                                             |
|                                                                       |
| \--                                                                   |
+-----------------------------------------------------------------------+

Note that aggregator.log is always appended to, so will include results
from previous runs.

6.  Edit the plan to train for more rounds, etc.

Bibliography
============

Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., & Shmatikov, V. (2018).
How To Backdoor Federated Learning. *arXiv,
https://arxiv.org/abs/1807.00459*.Bahmani, R., Barbosa, M., Brasser, F.,
Portela, B., Sadeghi, A.-R., Scerri, G., & Warinschi, B. (2017). Secure
multiparty computation from SGX.
*https://hal.archives-ouvertes.fr/hal-01898742/file/2016-1057.pdf*.Bhagoji,
A. N., Chakraborty, Supriyo, M. P., & Calo, S. (2018). Analyzing
Federated Learning through an Adversarial Lens. *arXiv,
https://arxiv.org/abs/1811.12470*.Bonawitz, K., Eichner, H., Grieskamp,
W., Huba, D., Ingerman, A., Ivanov, V., . . . Van Overveldt, T. (2019).
Towards federated learning at scale: System design.McMahan, H. B.
(2016). Communication-efficient learning of deep networks from
decentralized data.Sheller, M., Reina, G. A., Edwards, B., Martin, J., &
Bakas, S. (2019). Multi-institutional Deep Learning Modeling Without
Sharing Patient Data: A Feasibility Study on Brain Tumor Segmentation.
*Lecture Notes in Computer Science*.Yang, Q., Liu, Y., Chen, T., & Tong,
Y. (2019). Federated Machine Learning: Concept and Applications. *ACM
Transactions on Intelligent Systems and Technology*.
