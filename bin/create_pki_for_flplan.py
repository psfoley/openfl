#!/usr/bin/env python3

# Copyright (C) 2020 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between Intel Corporation and you.

# This file takes in a plan and builds the ca, agg_cert and all col_certs for the plan (if needed)
import argparse
import os
from subprocess import call
import yaml

def load_yaml(path):
    plan = None
    with open(path, 'r') as f:
        plan = yaml.safe_load(f.read())
    return plan


def create_if_needed(path, args):
    if not os.path.exists(path):
        call(args)
        print('created', path)
    else:
        print('no need to create', path)


def create_certs_for(cert_dir, name, full_hostname, ca_key_path, ca_crt_path):
    # compute cert file paths
    key_path = os.path.join(cert_dir, '{}.key'.format(name))
    csr_path = os.path.join(cert_dir, '{}.csr'.format(name))
    crt_path = os.path.join(cert_dir, '{}.crt'.format(name))

    # create key
    create_if_needed(key_path, ['openssl', 'genrsa', '-out', key_path, '3072'])

    # create csr
    create_if_needed(csr_path, ['openssl', 'req', '-new', '-key', key_path, '-out',
                                csr_path, '-subj', '/CN={}'.format(full_hostname)])
    
    # create crt by signing csr
    create_if_needed(crt_path, ['openssl', 'x509', '-req', '-in', csr_path, '-CA',
                                ca_crt_path, '-CAkey', ca_key_path, '-CAcreateserial', '-out', crt_path])

def main(plan):    
    # FIXME: consistent filesystem (#15)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.join(script_dir, 'federations')
    plan_dir = os.path.join(base_dir, 'plans')

    flplan = load_yaml(os.path.join(plan_dir, plan))
    hostnames = flplan['hostnames']

    # get the cert path
    cert_dir = os.path.join(base_dir, 'certs', flplan['grpc']['cert_folder'])

    # FIXME: are these permissions okay?
    # ensure the cert dir exists
    os.makedirs(cert_dir, exist_ok=True)

    # compute the paths for the ca files
    ca_key_path = os.path.join(cert_dir, 'ca.key')
    ca_crt_path = os.path.join(cert_dir, 'ca.crt')

    # create ca.key
    create_if_needed(ca_key_path, ['openssl', 'genrsa', '-out', ca_key_path, '3072'])

    # create ca.crt
    create_if_needed(ca_crt_path, ['openssl', 'req', '-new', '-x509', '-key', ca_key_path, '-out', ca_crt_path, 
                                   '-subj', "/CN=Trusted Federated Learning Test Cert Authority"])

    # create the agg cert
    name = flplan['federation']['agg_id']
    full_hostname = flplan['federation']['agg_addr']
    create_certs_for(cert_dir, name, full_hostname, ca_key_path, ca_crt_path)

    # for each collaborator, create a key, csr and crt
    for name in flplan['federation']['col_ids']:
        if name not in hostnames.keys():
            full_hostname = hostnames['__DEFAULT_HOSTNAME__']
        else:
            full_hostname = hostnames[name]
        create_certs_for(cert_dir, name, full_hostname, ca_key_path, ca_crt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', '-p', type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
