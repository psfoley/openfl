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

@collaborator.command(name='start')
@pass_context
@option('-p', '--plan',              required = False, help = 'Federated learning plan [plan/plan.yaml]',               default = 'plan/plan.yaml', type = ClickPath(exists = True))
@option('-d', '--data_config',       required = False, help = 'The data set/shard configuration file [plan/data.yaml]', default = 'plan/data.yaml', type = ClickPath(exists = True))
@option('-n', '--collaborator_name', required = True,  help = 'The certified common name of the collaborator')
@option('-s', '--secure',            required = False, help = 'Enable Intel SGX Enclave', is_flag = True, default = False)
def start_(context, plan, collaborator_name, data_config, secure):
    '''Start a collaborator service'''

    plan = Plan.Parse(plan_config_path = Path(plan),
                      data_config_path = Path(data_config))

    # TODO: Need to restructure data loader config file loader

    echo(f'Data = {plan.cols_data_paths}')
    logger.info('🧿 Starting a Collaborator Service.')

    plan.get_collaborator(collaborator_name).run()

def RegisterDataPath(plan_name, silent=False):
    '''Register dataset path in the plan/data.yaml file

    Args:
        plan_name (str): Name of the plan file
        silent (bool)  : Silent operation (don't prompt)
         
    '''

    from click import prompt

    # Ask for the data directory
    default_data_path = f'data/{plan_name}'
    if not silent:
        dirPath = prompt(f'\nWhere is the data directory for this collaborator in plan ' +
                        style(f'{plan_name}', fg='green') +
                        ' ? ', default=default_data_path)
    else:
        dirPath = default_data_path  # TODO: Need to figure out the default for this.

    # Read the data.yaml file
    d = {}
    data_yaml = 'plan/data.yaml'
    separator = ','
    with open(data_yaml, 'r') as f:
        for line in f:
            if separator in line:
                key, val = line.split(separator, maxsplit=1)
                d[key] = val.strip()

    d[plan_name] = dirPath

    # Write the data.yaml
    with open(data_yaml, 'w') as f:
        for key, val in d.items():
            f.write(f'{key}{separator}{val}\n')

@collaborator.command(name='create')
@pass_context
@option('-n', '--collaborator_name', required = True,  help = 'The certified common name of the collaborator')
@option('-s', '--silent', help = 'Do not prompt', is_flag=True)
def create_(context, collaborator_name, silent):
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

    echo(f'  Moving COLLABORATOR certificate to: ' + style(f'{PKI_DIR}/{file_name}', fg = 'green'))

    (PKI_DIR / f'{file_name}').mkdir(parents = True, exist_ok = True)
    (PKI_DIR / f'{file_name}.csr').rename(PKI_DIR / f'{file_name}' / f'{file_name}.csr')
    (PKI_DIR / f'{file_name}.key').rename(PKI_DIR / f'{file_name}' / f'{file_name}.key')

    RegisterDataPath(f'default', silent=silent)  # TODO: Is there a way to figure out the plan name automatically? Or do we have a new function for adding new paths for different plans?

def findCertificateName(file_name):
    '''Searches the CRT for the actual collaborator name
    '''

    # This loop looks for the collaborator name in the key
    with open(file_name, 'r') as f: 
        for line in f: 
            if 'Subject: CN=' in line: 
                col_name = line.split('=')[-1].strip()
                break 
    return col_name


def RegisterCollaborator(file_name):
    '''Register the collaborator name in the cols.yaml list

    Args:
        file_name (str): The name of the collaborator in this federation

    '''
    from yaml    import load, dump, FullLoader

    col_name = findCertificateName(file_name)

    cols_file = 'plan/cols.yaml'

    with open(cols_file, 'r') as f:
        doc = load(f, Loader=FullLoader)

    if not doc:   # YAML is not correctly formatted
        doc = {}  # Create empty dictionary

    if 'collaborators' not in doc.keys() or not doc['collaborators']:  # List doesn't exist
        doc['collaborators'] = [] # Create empty list

    if  col_name in doc['collaborators']:

        echo(f'\nCollaborator ' + 
             style(f'{col_name}', fg='green') +
             f' is already in the ' +
             style(f'{cols_file}', fg='green'))

    else:

        doc['collaborators'].append(col_name)
        with open(cols_file, 'w') as f:
            dump(doc, f)

        echo(f'\nRegistering ' +
             style(f'{col_name}', fg='green') +
             f' in ' +
             style(f'{cols_file}', fg='green'))

def sign_certificate(file_name):
    '''Sign the certificate
    '''

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

    RegisterCollaborator(PKI_DIR / f'{file_name}' / f'{file_name}.crt')

@collaborator.command(name='certify')
@pass_context
@option('-n', '--certificate_name', required = True,  help = 'The certificate signing request filename (*.csr) for the collaborator')
@option('-s', '--silent', help = 'Do not prompt', is_flag=True)
def certify_(context, certificate_name, silent):
    '''Sign/certify collaborator certificate key pair'''

    from os.path import splitext, basename
    from shutil  import copyfile

    from click   import confirm

    cert_name = splitext(certificate_name)[0]
    file_name = basename(cert_name)

    # Copy PKI to cert directory
    # TODO:  Circle back to this. Not sure if we need to copy the file or if we can call it directly from openssl
    # Was getting a file not found error otherwise.
    copyfile(f'{cert_name}.csr', PKI_DIR / f'{file_name}.csr')
    copyfile(f'{cert_name}.key', PKI_DIR / f'{file_name}.key')

    output = vex(f'openssl sha256  '
                  f'{file_name}.csr', workdir=PKI_DIR)

    csr_hash = output.stdout.split('=')[1]

    echo(f'The CSR Hash for file ' + 
         style(f'{file_name}.csr', fg='green') +
         ' = ' +
         style(f'{csr_hash}', fg='red'))

    if silent:

        sign_certificate(file_name)

    else:

        if confirm("Do you want to sign this certificate?"):

            sign_certificate(file_name)

        else:
            echo(style('Not signing certificate.', fg='red') +
                ' Please check with this collaborator to get the correct certificate for this federation.')
