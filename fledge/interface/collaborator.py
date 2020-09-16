from cli_helper import *

from fledge.component import Aggregator
from fledge.transport import AggregatorGRPCServer
from fledge.component import Collaborator
from fledge.transport import CollaboratorGRPCClient
from fledge.federated import Plan

logger = getLogger(__name__)

@group()
@pass_context
def collaborator(context):
    '''Manage Federated Learning Collaborators'''
    context.obj['group'] = 'service'

@collaborator.command()
@pass_context
@option('-p', '--plan',              required = False, help = 'Federated learning plan [plan/plan.yaml]',               default = 'plan/plan.yaml', type = ClickPath(exists = True))
@option('-d', '--data_config',       required = False, help = 'The data set/shard configuration file [plan/data.yaml]', default = 'plan/data.yaml', type = ClickPath(exists = True))
@option('-n', '--collaborator_name', required = True,  help = 'The certified common name of the collaborator')
@option('-s', '--secure',            required = False, help = 'Enable Intel SGX Enclave', is_flag = True, default = False)
def start(context, plan, collaborator_name, data_config, secure):
    '''Start a collaborator service'''

    plan = Plan.Parse(plan_config_path = Path(plan),
                      data_config_path = Path(data_config))

    if  collaborator_name not in plan.cols_data_paths:
        logger.error(f'Collaborator [red]{collaborator_name}[/] not found in Data Configuration file [red]{data_config}[/].', extra = {'markup' : True})
        exit()

    if  plan.data_group_name not in plan.cols_data_paths[collaborator_name]:
        logger.error(f'Group [red]{plan.data_group_name}[/] for '
                     f'Collaborator [red]{collaborator_name}[/] not found in Data Configuration file [red]{data_config}[/].', extra = {'markup' : True})
        exit()

    logger.info('🧿 Starting a Collaborator Service.')

    plan.get_collaborator(collaborator_name).run()

@collaborator.command()
@pass_context
@option('-n', '--collaborator_name', required = True,  help = 'The certified common name of the collaborator')
def certify(context, collaborator_name):
    '''Create collaborator certificate key pair'''

    common_name              = f'{collaborator_name}'.lower()
    subject_alternative_name = f'DNS:{common_name}'
    file_name                = f'col_{common_name}'

    echo(f'Creating COLLABORATOR certificate key pair with following settings: '
         f'CN={style(common_name, fg = "red")}, SAN={style(subject_alternative_name, fg = "red")}')

    if  True:
        extensions = 'client_reqext_san'
    else:
        echo(f'Note: Collaborator CN is not a valid FQDN and will not be added to the DNS entry of the subject alternative names')
        extensions = 'client_reqext'

    client_conf = 'config/client.conf'
    vex(f'openssl req -new '
        f'-config {client_conf} '
        f'-subj "/CN={common_name}" '
        f'-out {file_name}.csr -keyout {file_name}.key '
        f'-reqexts {extensions}', workdir = PKI_DIR, env = {'SAN': subject_alternative_name})

    echo(f' Signing COLLABORATOR certificate key pair')

    signing_conf = 'config/signing-ca.conf'
    vex(f'openssl ca -batch '
        f'-config {signing_conf} '
        f'-extensions server_ext '
        f'-in {file_name}.csr -out {file_name}.crt', workdir = PKI_DIR)

    echo(f'  Moving COLLABORATOR certificate key pair to: ' + style(f'{PKI_DIR}/{file_name}', fg = 'green'))

    (PKI_DIR / f'{file_name}').mkdir(parents = True, exist_ok = True)
    (PKI_DIR / f'{file_name}.crt').rename(PKI_DIR / f'{file_name}' / f'{file_name}.crt')
    (PKI_DIR / f'{file_name}.key').rename(PKI_DIR / f'{file_name}' / f'{file_name}.key')
    (PKI_DIR / f'{file_name}.csr').unlink()

@collaborator.command()
def test():

    with progressbar(range(10)) as bar:
        for item in bar:
            sleep(1)
