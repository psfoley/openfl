from cli_helper import *

from fledge.component import Aggregator
from fledge.transport import AggregatorGRPCServer
from fledge.component import Collaborator
from fledge.transport import CollaboratorGRPCClient
from fledge.federated import Plan

logger = getLogger(__name__)

@group()
@pass_context
def aggregator(context):
    '''Manage Federated Learning Aggregator'''
    context.obj['group'] = 'aggregator'

@aggregator.command()
@pass_context
@option('-p', '--plan',            required = False, help = 'Federated learning plan [plan/plan.yaml]',      default = 'plan/plan.yaml', type = ClickPath(exists = True))
@option('-c', '--authorized_cols', required = False, help = 'Authorized collaborator list [plan/cols.yaml]', default = 'plan/cols.yaml', type = ClickPath(exists = True))
@option('-s', '--secure',          required = False, help = 'Enable Intel SGX Enclave', is_flag = True, default = False)
def start(context, plan, authorized_cols, secure):
    '''Start the aggregator service'''

    plan = Plan.Parse(plan_config_path = Path(plan),
                      cols_config_path = Path(authorized_cols))

    logger.info('🧿 Starting the Aggregator Service.')

    plan.get_server().serve()

@aggregator.command()
@pass_context
@option('--fqdn', required = False, help = f'The fully qualified domain name of aggregator node [{getfqdn()}]', default = getfqdn())
def certify(context, fqdn):
    '''Create aggregator certificate key pair'''

    common_name              = f'{fqdn}'.lower()
    subject_alternative_name = f'DNS:{common_name}'
    file_name                = f'agg_{common_name}'

    echo(f'Creating AGGREGATOR certificate key pair with following settings: '
         f'CN={style(common_name, fg = "red")}, SAN={style(subject_alternative_name, fg = "red")}')

    server_conf = 'config/server.conf'
    vex(f'openssl req -new '
        f'-config {server_conf} '
        f'-subj "/CN={common_name}" '
        f'-out {file_name}.csr -keyout {file_name}.key', workdir = PKI_DIR, env = {'SAN': subject_alternative_name})

    echo(f' Signing AGGREGATOR certificate key pair')

    signing_conf = 'config/signing-ca.conf'
    vex(f'openssl ca -batch '
        f'-config {signing_conf} '
        f'-extensions server_ext '
        f'-in {file_name}.csr -out {file_name}.crt', workdir = PKI_DIR)

    echo(f'  Moving AGGREGATOR certificate key pair to: ' + style(f'{PKI_DIR}/{file_name}', fg = 'green'))
    
    (PKI_DIR / f'{file_name}').mkdir(parents = True, exist_ok = True)
    (PKI_DIR / f'{file_name}.crt').rename(PKI_DIR / f'{file_name}' / f'{file_name}.crt')
    (PKI_DIR / f'{file_name}.key').rename(PKI_DIR / f'{file_name}' / f'{file_name}.key')
    (PKI_DIR / f'{file_name}.csr').unlink()
