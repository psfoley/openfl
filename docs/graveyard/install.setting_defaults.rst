.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

**************************
Setting Defaults for FLPlans (Testing)
**************************

This section walks you through configuring one or more machines to run test federations for testing/development. For production deployments, see forthcoming documentation (TODO: link to "production" deployments documentation when we have it).

Prerequisites:

1. In root directory of source code (TODO: link section on getting the code)
2. You know the fully-qualified domain name or IP address of the machine you want to use as your aggregator. For example, you can use the following linux/cygwin command:


.. code-block:: console

    $ hostname –-fqdn

.. _network_defaults:

Setting Defaults for Network settings
#########

FL Plans support default configurations to make it easier to share settings between FL Plans. This also makes it easier to customize another's plan without changing the main plan file, so you can share plans with other developers.

You can list the various files with:
   
.. code-block:: console

    $ ls -l bin/federations/plans/defaults
    total 36
    -rw-r--r-- 1 msheller intelall 285 Jul  6 15:01 collaborator.yaml
    -rw-r--r-- 1 msheller intelall 274 Jul  6 15:01 data_empty.yaml
    -rw-r--r-- 1 msheller intelall 311 Jul  6 15:01 data_pt_brats.yaml
    -rw-r--r-- 1 msheller intelall 333 Jul  6 15:01 data_pt_cifar10.yaml
    -rw-r--r-- 1 msheller intelall 327 Jul  6 15:01 data_pt_mnist.yaml
    -rw-r--r-- 1 msheller intelall 323 Jul  6 15:01 data_tf_brats.yaml
    -rw-r--r-- 1 msheller intelall 315 Jul  6 15:01 data_tf_cifar10.yaml
    -rw-r--r-- 1 msheller intelall 274 Jul  6 15:01 data_tf_mnist.yaml
    -rw-r--r-- 1 msheller intelall 400 Jul  6 15:01 network.yaml.example

We're going to create a network.yaml file from the example and set it to be specific to you.

1. First, we need to copy the example file that we're going to change. Copy it and take a look at it:

.. code-block:: console

    $ cp bin/federations/plans/defaults/network.yaml.example bin/federations/plans/defaults/network.yaml
    $ cat bin/federations/plans/defaults/network.yaml
    # Copyright (C) 2020 Intel Corporation
    # Licensed subject to Collaboration Agreement dated February 28th, 2020 between Intel Corporation and Trustees of the University of Pennsylvania.
    
    agg_addr            : FQDN of aggregator machine
    agg_port            : auto
    hash_salt           : Anything you want. Make it unique
    disable_tls         : False
    disable_client_auth : False
    cert_folder         : pki

2. Open your new file in an editor:

.. code-block:: console

    $ vi bin/federations/plans/defaults/network.yaml

3. First, we need to set the aggregator address to the FQDN or IP of the aggregator machine, such as:

.. code-block:: console

    agg_addr            : msheller-aggregator.intel.com

3. Next, you can choose a specific port, or if you intend to run multiple aggregator processes for testing, leave it as 'auto'. 'Auto' simply uses federation UUID (which is a hash of the FL Plan files, including the defaults files) to pick a random port. This way the collaborators and aggregator will compute the same "random" port. (TODO: link autoport doc).

.. code-block:: console

    agg_port            : auto # I am keeping it auto because I run lots of federations at the same time on the same machines...

4. Finally, in development teams with shared machines, it is possible for FL Plans to be exactly identical. This leads to idential FL Plan UUIDs (hashes). For this reason, we give our plans a silly salt. It can be anything, so long as it is unique among your team:

.. code-block:: console

    hash_salt            : micah.j.sheller@intel.com # your email isn't a bad choice

Now your FL Plans will use your aggregator machine, and if it is shared, you shouldn't likely run into port choice conflicts.

Note that here is where you can do things like disable tls or change which directory you use for your certs. **We don't condone disabling TLS**.


Creating Collaborator Lists
#########

When an aggregator executes an FL Plan, it also requires a list of collaborator names that are allowed to participate. In a production setting, these names are meaningful and are tightly coupled with each client's digital certificate used in the TLS connection. However, for test environments, you can name them whatever you wish (you will be passing these on the collaborator commandlines). You can find existing test lists under:

.. code-block:: console

    $ ls -l bin/federations/collaborator_lists                                                                                                                                                                                         
    total 24
    -rw-r--r-- 1 msheller intelall  46 Jul  6 15:01 col_one_big.yaml
    -rw-r--r-- 1 msheller intelall 147 Jul  6 15:01 cols_10.yaml
    -rw-r--r-- 1 msheller intelall  52 Jul  6 15:01 cols_2.yaml
    -rw-r--r-- 1 msheller intelall 432 Jul  6 15:01 cols_32.yaml
    -rw-r--r-- 1 msheller intelall  40 Jul  6 15:01 only_col_2.yaml
    -rw-r--r-- 1 msheller intelall  52 Jul  6 15:01 only_cols_2_and_3.yaml

And you'll see that they have very exciting contents, such as:

.. code-block:: console

    $ cat bin/federations/collaborator_lists/cols_10.yaml
    collaborator_common_names :
      - 'col_0'
      - 'col_1'
      - 'col_2'
      - 'col_3'
      - 'col_4'
      - 'col_5'
      - 'col_6'
      - 'col_7'
      - 'col_8'
      - 'col_9'

In a real setting, these lists would hold the common names in the certificates the collaborators (one per cert). In a development/test environment, feel free to use any naming-convention. You will need these names later, so we recommend keeping them simple. Note that you may want to run multiple collaborators on a single machine, so you may not want to use machine names here. (TODO: Add reference to auto-lists when we implement that convenience feature).

Configuring Collaborator Local Data Directories
#########

When a collaborator executes and FL Plan, the FL Plan will contain a data_name entry such as "brats" or "mnist_shard" or similar. This name serves as a key in a dictionary of paths or shards on the collaborator (we use "shards" to refer to tests where a single data is split among collaborators at runtime, i.e. "sharded"). We store these mappings in .yaml files of a structure:

.. code-block:: console
    collaborator_common_name:
        data_name: <path or shard>

This way, we can configure the data-paths for multiple collaborators in a single file. In production, such a file would only have the information for a single collaborator.

You'll find one such file in the repository that looks like this:

.. code-block:: console

    $ cat bin/federations/local_data_config.yaml
    collaborators:
      col_one_big:
        brats: '/raid/datasets/BraTS17/by_institution_NIfTY/0-9'
      col_0:
        brats: '/raid/datasets/BraTS17/by_institution_NIfTY/0'
        mnist_shard: 0
        cifar10_shard: 0
      col_1:
        brats: '/raid/datasets/BraTS17/by_institution_NIfTY/1'
        mnist_shard: 1
        cifar10_shard: 1
    ...

For the shards, you'll usually just need an index. For datasets that are already seperated, you need to set the paths for each collaborator/dataset pair here. Note that in our case, we have a shared /raid volume that each of our development nodes can access. This makes life easy, and also ensures we can run any collaborator on any machine. Highly recommended for testing and development! We even go so far as using softlinks to allow various collaborator assignments (e.g. moving data around to increase collaborator-specific biases).


Copy the Files to the other machines
#########

Currently, we don't yet support configuration through the governor. All configuration is done manually. This means we need to copy these files around to our other systems. Hopefully, you can do this with a few calls to 'scp' :) For the aggregator node, you'll need the copy the following files:

1. Copy files to your aggregator machine. You'll need to copy over the following:

+-----------------------------------+--------------------------------------------------------------+
| File Type                         | Filename                                                     |
+===================================+==============================================================+
| Plans files defaults              | bin/federations/plans/defaults/\*.yaml                       |
+-----------------------------------+--------------------------------------------------------------+
| Plan files (if changed)           | bin/federations/plans/\*.yaml                                |
+-----------------------------------+--------------------------------------------------------------+
| Collaborator lists (if changed)   | bin/federations/collaborator_lists/\*.yaml                   |
+-----------------------------------+--------------------------------------------------------------+

2. Copy files to **each** of your collaborator machines. You'll need to copy over the following:

+-----------------------------------+--------------------------------------------------------------+
| File Type                         | Filename                                                     |
+===================================+==============================================================+
| Plans files defaults              | bin/federations/plans/defaults/\*.yaml                       |
+-----------------------------------+--------------------------------------------------------------+
| Plan files (if changed)           | bin/federations/plans/\*.yaml                                |
+-----------------------------------+--------------------------------------------------------------+
| Data config file                  | bin/federations/local_data_config.yaml                       |
+-----------------------------------+--------------------------------------------------------------+
