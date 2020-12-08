import pytest

from fledge.component.assigner import RandomGroupedAssigner


ROUNDS_TO_TRAIN = 10


@pytest.fixture
def task_groups():
    task_groups = [
        {
            'name': 'train_and_validate',
            'percentage': 1.0,
            'tasks': [
                'aggregated_model_validation',
                'train',
                'locally_tuned_model_validation'
            ]
        }
    ]
    return task_groups


@pytest.fixture
def authorized_cols():
    return ['one', 'two']


@pytest.fixture
def assigner(task_groups, authorized_cols):
    assigner = RandomGroupedAssigner

    assigner = assigner(task_groups, None, authorized_cols, ROUNDS_TO_TRAIN)
    return assigner


def test_define_task_assignments(assigner):
    assigner.define_task_assignments()


@pytest.mark.parametrize('round_number', range(ROUNDS_TO_TRAIN))
def test_get_tasks_for_collaborator(assigner, task_groups,
                                    authorized_cols, round_number):
    tasks = assigner.get_tasks_for_collaborator(
        authorized_cols[0], round_number)
    assert tasks == task_groups[0]['tasks']


@pytest.mark.parametrize('round_number', range(ROUNDS_TO_TRAIN))
def test_get_collaborators_for_task(
        assigner, task_groups, round_number, authorized_cols):
    for task_name in task_groups[0]['tasks']:
        cols = assigner.get_collaborators_for_task(task_name, round_number)
        assert set(cols) == set(authorized_cols)
