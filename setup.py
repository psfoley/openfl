# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

from setuptools import setup, find_packages

setup(
    name = 'fledge',
    version = '0.20',
    author = 'Intel Corporation',
    description = 'Federated Learning on the Edge',
    packages = [
        'fledge',
        'fledge.interface',
        'fledge.component',
        'fledge.native',
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
        'fledge-workspace',
        'fledge-tutorials',
    ],
    include_package_data = True,
    install_requires = [
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
        'jupyter',
        'ipykernel',
        'flatten_json',
    ],
    entry_points = {
        'console_scripts' : ['fx=fledge.interface.cli:entry']
    }
)
