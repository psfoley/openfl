class TaskAssigner(object):
    """
    The task assigner maintains a list of tasks, and decides the policy for which collaborator should run those tasks
    There may be many types of policies implemented, but a natural place to start is with a:

    RandomGroupedTaskAssigner  - Given a set of task groups, and a percentage, assign that task group to that percentage 
                                 of collaborators in the federation. After assigning the tasks to collaborator, those tasks
                                 should be carried out each round (no reassignment between rounds)
    GroupedTaskAssigner - Given task groups and a list of collaborators that belong to that task group, carry out tasks for each                                    round of experiment 
    """
    def __init__(self,task_groups,tasks,collaborator_list,rounds):
        self.task_groups = task_groups
        self.tasks = tasks
        self.collaborator_list = collaborator_list
        self.rounds = rounds

    def get_tasks_for_collaborator(self, collaborator_name):
        raise NotImplementedError

    def get_collaborators_for_task(task_name, round_number):
        raise NotImplementedError

    def get_all_tasks_for_round(round_number):
        raise NotImplementedError


