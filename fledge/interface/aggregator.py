# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from socket import getfqdn
from logging import getLogger
from pathlib import Path
from click import Path as ClickPath
from click import group, option, pass_context
from click import echo, style

from fledge.federated import Plan
from fledge.interface.cli_helper import vex
from fledge.interface.cli_helper import PKI_DIR


logger = getLogger(__name__)


@group()
@pass_context
def aggregator(context):
    """Manage Federated Learning Aggregator."""
    context.obj['group'] = 'aggregator'


@aggregator.command(name='start')
@pass_context
@option('-p', '--plan', required=False,
        help='Federated learning plan [plan/plan.yaml]',
        default='plan/plan.yaml',
        type=ClickPath(exists=True))
@option('-c', '--authorized_cols', required=False,
        help='Authorized collaborator list [plan/cols.yaml]',
        default='plan/cols.yaml', type=ClickPath(exists=True))
@option('-s', '--secure', required=False,
        help='Enable Intel SGX Enclave', is_flag=True, default=False)
def start_(context, plan, authorized_cols, secure):
    """Start the aggregator service."""
    plan = Plan.Parse(plan_config_path=Path(plan),
                      cols_config_path=Path(authorized_cols))

    logger.info('🧿 Starting the Aggregator Service.')

    plan.get_server().serve()


@aggregator.command(name='generate-cert-request')
@option('--fqdn', required=False,
        help=f'The fully qualified domain name of'
             f' aggregator node [{getfqdn()}]',
        default=getfqdn())
def _generate_cert_request(fqdn):
    generate_cert_request(fqdn)


def generate_cert_request(fqdn):
    """Create aggregator certificate key pair."""
    common_name = f'{fqdn}'.lower()
    subject_alternative_name = f'DNS:{common_name}'
    file_name = f'agg_{common_name}'

    echo(f'Creating AGGREGATOR certificate key pair with following settings: '
         f'CN={style(common_name, fg="red")},'
         f' SAN={style(subject_alternative_name, fg="red")}')

    server_conf = 'config/server.conf'
    vex('openssl req -new '
        f'-config {server_conf} '
        f'-subj "/CN={common_name}" '
        f'-out {file_name}.csr -keyout {file_name}.key',
        workdir=PKI_DIR, env={'SAN': subject_alternative_name})

    echo('  Moving AGGREGATOR certificate key pair to: ' + style(
        f'{PKI_DIR}/server', fg='green'))

    (PKI_DIR / 'server').mkdir(parents=True, exist_ok=True)
    (PKI_DIR / f'{file_name}.csr').rename(
        PKI_DIR / 'server' / f'{file_name}.csr')
    (PKI_DIR / f'{file_name}.key').rename(
        PKI_DIR / 'server' / f'{file_name}.key')


def findCertificateName(file_name):
    """Search the CRT for the actual aggregator name."""
    # This loop looks for the collaborator name in the key
    with open(file_name, 'r') as f:
        for line in f:
            if 'Subject: CN=' in line:
                col_name = line.split('=')[-1].strip()
                break
    return col_name


def sign_certificate(file_name):
    """Sign the certificate."""
    echo(' Signing AGGREGATOR certificate key pair')

    signing_conf = 'config/signing-ca.conf'
    vex('openssl ca -batch '
        f'-config {signing_conf} '
        f'-extensions server_ext '
        f'-in {file_name}.csr -out {file_name}.crt', workdir=PKI_DIR)

    echo('  Moving AGGREGATOR certificate key pair'
         ' to: ' + style(f'{PKI_DIR}/server', fg='green'))

    (PKI_DIR / 'server').mkdir(parents=True, exist_ok=True)
    (PKI_DIR / f'{file_name}.crt').rename(
        PKI_DIR / 'server' / f'{file_name}.crt')
    (PKI_DIR / f'{file_name}.csr').unlink()


@aggregator.command(name='certify')
@option('-n', '--fqdn',
        help='The fully qualified domain name of aggregator node [{getfqdn()}]',
        default=getfqdn())
@option('-s', '--silent', help='Do not prompt', is_flag=True)
def _certify(fqdn, silent):
    certify(fqdn, silent)


def certify(fqdn, silent):
    """Sign/certify the aggregator certificate key pair."""
    from shutil import copyfile

    from click import confirm

    common_name = f'{fqdn}'.lower()
    file_name = f'agg_{common_name}'
    cert_name = f'server/{file_name}'

    # Copy PKI to cert directory
    # TODO:  Circle back to this. Not sure if we need to copy the file or
    #  if we can call it directly from openssl
    # Was getting a file not found error otherwise.
    copyfile(PKI_DIR / f'{cert_name}.csr', PKI_DIR / f'{file_name}.csr')

    output = vex(f'openssl sha256  '
                 f'{file_name}.csr', workdir=PKI_DIR)

    csr_hash = output.stdout.split('=')[1]

    echo('The CSR Hash for file '
         + style(f'{file_name}.csr', fg='green')
         + ' = '
         + style(f'{csr_hash}', fg='red'))

    if silent:

        sign_certificate(file_name)

    else:

        if confirm("Do you want to sign this certificate?"):

            sign_certificate(file_name)

        else:
            echo(style('Not signing certificate.', fg='red')
                 + ' Please check with this AGGREGATOR to get the correct'
                   ' certificate for this federation.')
