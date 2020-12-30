.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _plan_settings:

***************
Plan Settings
***************

Plan is described by ``plan.yaml`` file located in ``plan`` folder of the workspace.
Each YAML top-level section contains 3 main subsections:
* ``template``: name of the class including top-level packages names.
  Instance of this class is created when plan gets initialized.
* ``settings``: arguments that are passed to the class constructor
* ``defaults``: file that contains default settings for this subsection.
  Any setting from defaults file can be overriden in ``plan.yaml`` file.

Example of ``plan.yaml``:

.. code-block:: yaml

  aggregator :
    defaults : plan/defaults/aggregator.yaml # where to get the default settings
    template : fledge.component.Aggregator # what class to use for the instance creation
    settings : # arguments to pass to the class constructor
      init_state_path : save/keras_lenet_init.pbuf
      best_state_path : save/keras_lenet_best.pbuf
      last_state_path : save/keras_lenet_last.pbuf
      rounds_to_train : 10

  collaborator :
    defaults : plan/defaults/collaborator.yaml
    template : fledge.component.Collaborator
    settings :
      epochs_per_round : 1.0
      polling_interval : 4
      delta_updates    : false
      opt_treatment    : RESET

  data_loader :
    defaults : plan/defaults/data_loader.yaml
    template : code.fecifar_inmemory.FastEstimatorCifarInMemory
    settings :
      collaborator_count : 2
      data_group_name    : cifar
     batch_size         : 32

  task_runner :
    defaults : plan/defaults/task_runner.yaml
    template : code.fe_fgsm.FastEstimatorFGSM

  assigner :
    defaults : plan/defaults/assigner.yaml

  tasks :
    defaults : plan/defaults/tasks_fast_estimator.yaml

======================
Configurable settings
======================

- :class:`Aggregator <fledge.component.Aggregator>`
- :class:`Collaborator <fledge.component.Collaborator>`
- :class:`Data Loader <fledge.federated.DataLoader>`
- :class:`Task Runner <fledge.federated.TaskRunner>`
- :class:`Assigner <fledge.component.Assigner>`

++++++++++++++
Tasks
++++++++++++++

Each task subsection should contain:

- ``function``: function name to call.
  The function must be the one defined in :class:`TaskRunner <fledge.federated.TaskRunner>` class.
- ``kwargs``: kwargs passed to the ``function``.