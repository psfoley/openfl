# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class Assigner:
    """
    The task assigner maintains a list of tasks, and decides the policy for
    which collaborator should run those tasks
    There may be many types of policies implemented, but a natural place
    to start is with a:

    RandomGroupedTaskAssigner  - Given a set of task groups, and a percentage,
                                 assign that task group to that
                                 percentage of collaborators in the federation.
                                 After assigning the tasks to
                                 collaborator, those tasks should be carried
                                 out each round (no reassignment
                                 between rounds)
    GroupedTaskAssigner - Given task groups and a list of collaborators that
                          belong to that task group,
                          carry out tasks for each round of experiment
    """

    def __init__(self, task_groups, tasks, authorized_cols,
                 rounds_to_train, **kwargs):
        self.task_groups = task_groups
        self.tasks = tasks
        self.authorized_cols = authorized_cols
        self.rounds = rounds_to_train
        self.all_tasks_in_groups = []

        self.task_group_collaborators = {}
        self.collaborators_for_task = {}
        self.collaborator_tasks = {}

        self.define_task_assignments()

    def define_task_assignments(self):
        raise NotImplementedError

    def get_tasks_for_collaborator(self, collaborator_name, round_number):
        raise NotImplementedError

    def get_collaborators_for_task(self, task_name, round_number):
        raise NotImplementedError

    def get_all_tasks_for_round(self, round_number):
        """
        Currently all tasks are performed on each round, but there may be a
        reason to change this
        """
        return self.all_tasks_in_groups

    def get_aggregation_type_for_task(self, task_name):
        if 'aggregation_type' not in self.tasks[task_name]:
            return None
        return self.tasks[task_name]['aggregation_type']
