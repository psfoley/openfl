.. # Copyright (C) 2020 Intel Corporation
.. # Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

.. _multiple_plans:

Managing Multiple Plans
#######################

|productName| allows developers to use multiple Federation Plans for the same workspace. All federation plans are contained within the :code:`WORKSPACE.FOLDER/plan/plans` directory. The following :code:`fx` commands allow developers to manage these plans.

.. _creating_new_plans:

All workspaces begin with a :code:`default` Federation Plan. If you are working on a plan, you can save it for future use by running the following command:

    .. code-block:: console
    
       $ fx plan save -n NEW.PLAN.NAME
       
    where **NEW.PLAN.NAME** is the new plan for your workspace. 
    
.. _switching_plans:

    .. code-block:: console
    
       $ fx plan switch -n NEW.PLAN.NAME

       where **NEW.PLAN.NAME** is the new plan for your workspace. 

    .. note::

       If you have changed the :code:`plan` file, you should first :ref:`save the plan <creating_new_plans>` before switching. Otherwise, any changes will be lost.
       
.. _removing_plans:

    .. code-block:: console
    
        $ fx plan remove -n NEW.PLAN.NAME

    where **NEW.PLAN.NAME** is the new plan for your workspace. 
