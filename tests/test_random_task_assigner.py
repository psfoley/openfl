from tfedlrn import load_yaml
from tfedlrn import get_object
import numpy as np

def test_yaml_parsing_and_collaborator_task_assignment():
    np.random.seed(0)
    plan = load_yaml('tests/keras_cnn_mnist_2_tasks_test.yaml')
    task_assigner_config = plan['task_assigner']
    tasks = plan['tasks']
    task_assigner = get_object(**task_assigner_config,tasks=tasks,collaborator_list=['col_1','col_2'],rounds=10)
    assert(task_assigner.get_tasks_for_collaborator('col_2',0) == ['aggregated_model_validation'])
    assert(set(task_assigner.get_tasks_for_collaborator('col_1',0)) == set(['locally_tuned_model_validation','train','aggregated_model_validation']))
    assert(set(task_assigner.get_collaborators_for_task('aggregated_model_validation',0)) == set(['col_1','col_2']))
    assert(set(task_assigner.get_collaborators_for_task('train',0)) == set(['col_1']))

