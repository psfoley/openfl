# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from setuptools import setup, find_packages

setup(
    name = 'fledge',
    version = '0.1',
    author = 'Intel Corporation',
    description = 'Federated Learning on the Edge',
    packages = [
        'fledge',
        'fledge.interface',
        'fledge.component',
        'fledge.component.assigner',
        'fledge.component.aggregator',
        'fledge.component.collaborator',
        'fledge.component.envoy',
        'fledge.component.governor',
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
        'fledge-workspace'
    ],
    include_package_data = True,
    install_requires = [
        'Click>=7.0',
        'PyYAML',
        'numpy',
        'pandas',
        'protobuf',
        'grpcio',
        'rich',
        'tqdm',
        'nibabel',
        'scikit-learn',
    ],
    entry_points = {
        'console_scripts' : ['fx=fledge.interface.cli:entry']
    }
)
