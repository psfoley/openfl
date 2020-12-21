# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This package includes dependencies of the FLedge project."""

from setuptools import setup

setup(
    name='fledge',
    version='0.30',
    author='Intel Corporation',
    description='Federated Learning on the Edge',
    packages=[
        'fledge',
        'fledge.interface',
        'fledge.component',
        'fledge.native',
        'fledge.component.assigner',
        'fledge.component.aggregator',
        'fledge.component.collaborator',
        'fledge.utilities',
        'fledge.protocols',
        'fledge.pipelines',
        'fledge.databases',
        'fledge.transport',
        'fledge.transport.grpc',
        'fledge.federated',
        'fledge.federated.plan',
        'fledge.federated.task',
        'fledge.federated.data',
        'fledge-workspace',
        'fledge-docker',
        'fledge-tutorials',
    ],
    include_package_data=True,
    install_requires=[
        'Click>=7.0',
        'PyYAML>=5.1',
        'numpy',
        'pandas',
        'protobuf',
        'grpcio==1.30.0',
        'grpcio-tools==1.30.0',
        'rich',
        'tqdm',
        'scikit-learn',
        'docker',
        'jupyter',
        'ipykernel',
        'flatten_json',
    ],
    entry_points={
        'console_scripts': ['fx=fledge.interface.cli:entry']
    }
)
