# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

class Assigner(object):
    """
    The task assigner maintains a list of tasks, and decides the policy for which collaborator should run those tasks
    There may be many types of policies implemented, but a natural place to start is with a:

    RandomGroupedTaskAssigner  - Given a set of task groups, and a percentage, assign that task group to that percentage 
                                 of collaborators in the federation. After assigning the tasks to collaborator, those tasks
                                 should be carried out each round (no reassignment between rounds)
    GroupedTaskAssigner - Given task groups and a list of collaborators that belong to that task group, carry out tasks for each                                    round of experiment 
    """

    def __init__(self, task_groups, authorized_cols, rounds_to_train):

        self.task_groups     = task_groups
        self.authorized_cols = authorized_cols
        self.rounds          = rounds_to_train

    def get_tasks_for_collaborator(self, collaborator_name):
        raise NotImplementedError

    def get_collaborators_for_task(task_name, round_number):
        raise NotImplementedError

    def get_all_tasks_for_round(round_number):
        raise NotImplementedError
