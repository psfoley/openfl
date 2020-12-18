# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest import mock
import pytest

from fledge.component.assigner import Assigner


@pytest.fixture()
def assigner():
    assigner = Assigner
    assigner.define_task_assignments = mock.Mock()
    return assigner


def test_get_aggregation_type_for_task_none(assigner):
    task_name = 'test_name'
    tasks = {task_name: {}}

    assigner = assigner(None, tasks, None, None)

    aggregation_type = assigner.get_aggregation_type_for_task(task_name)

    assert aggregation_type is None


def test_get_aggregation_type_for_task(assigner):
    task_name = 'test_name'
    test_aggregation_type = 'test_aggregation_type'
    tasks = {task_name: {
        'aggregation_type': test_aggregation_type
    }}
    assigner = assigner(None, tasks, None, None)

    aggregation_type = assigner.get_aggregation_type_for_task(task_name)

    assert aggregation_type == test_aggregation_type


def test_get_all_tasks_for_round(assigner):
    assigner = Assigner(None, None, None, None)
    tasks = assigner.get_all_tasks_for_round('test')

    assert isinstance(tasks, list)
