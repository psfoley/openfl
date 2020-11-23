from fledge.interface.cli_helper import *

from subprocess import check_call
from sys        import executable
from warnings import warn

@group()
@pass_context
def workspace(context):
    '''Manage Federated Learning Workspaces'''
    context.obj['group'] = 'workspace'

def create_dirs(prefix):

    echo(f'Creating Workspace Directories')

    (prefix / 'cert').mkdir(parents = True, exist_ok = True) # certifications
    (prefix / 'data').mkdir(parents = True, exist_ok = True) # training data
    (prefix / 'logs').mkdir(parents = True, exist_ok = True) # training logs
    (prefix / 'plan').mkdir(parents = True, exist_ok = True) # federated learning plans
    (prefix / 'save').mkdir(parents = True, exist_ok = True) # model weight saves / initialization
    (prefix / 'code').mkdir(parents = True, exist_ok = True) # model code

    src = WORKSPACE / 'workspace/plan/defaults' # from default workspace
    dst = prefix    /           'plan/defaults' #   to created workspace

    copytree(src = src, dst = dst, dirs_exist_ok = True)

def create_cert(prefix):

    echo(f'Creating Workspace Certifications')

    src = WORKSPACE  / 'workspace/cert/config' # from default workspace
    dst = prefix     /           'cert/config' #   to created workspace

    copytree(src = src, dst = dst, dirs_exist_ok = True)

def create_temp(prefix, template):

    echo(f'Creating Workspace Templates')

    copytree(src = WORKSPACE / template , dst = prefix , dirs_exist_ok = True, ignore = ignore_patterns('__pycache__')) # from template workspace


def get_templates():
    """
    Grab the default templates from the distribution
    """

    return [d.name for d in WORKSPACE.glob('*') if d.is_dir() and d.name not in ['__pycache__', 'workspace']]

@workspace.command(name='create')
@option('--prefix',   required = True, help = 'Workspace name or path', type = ClickPath())
@option('--template', required = True, type = Choice(get_templates()))
def create_(prefix, template):
    create(prefix, template)


def create(prefix, template):
    """Create federated learning workspace"""
    from os.path  import isfile
    if not FLEDGE_HOME.exists():
        FLEDGE_HOME.mkdir()

    prefix   = Path(prefix)
    template = Path(template)

    create_dirs(prefix)
    create_cert(prefix)
    create_temp(prefix, template)

    requirements_filename = "requirements.txt"

    if isfile(f'{str(prefix)}/{requirements_filename}'):
        check_call([executable, "-m", "pip", "install", "-r", f"{prefix}/requirements.txt"])
        echo(f"Successfully installed packages from {prefix}/requirements.txt.")
    else:
        echo("No additional requirements for workspace defined. Skipping...")
    prefix_hash = _get_dir_hash(str(prefix.absolute()))
    with open(FLEDGE_HOME / f'requirements.{prefix_hash}.txt', 'w') as f:
        check_call([executable, '-m', 'pip', 'freeze'], stdout=f)

    print_tree(prefix, level = 3)

@workspace.command(name = 'export')
@pass_context
def export_(context):
    """Export federated learning workspace"""

    from shutil   import make_archive, copytree, copy2, ignore_patterns, rmtree
    from tempfile import mkdtemp
    from os       import getcwd, makedirs
    from os.path  import basename, join
    from plan     import FreezePlan

    # TODO: Does this need to freeze all plans?
    planFile = f'plan/plan.yaml'
    try:
        FreezePlan(planFile) 
    except:
        echo(f'Plan file "{planFile}" not found. No freeze performed.')

    requirements_filename = f'requirements.txt'

    prefix = getcwd()
    with open(requirements_filename, "w") as f:
        check_call([executable, "-m", "pip", "freeze"], stdout=f)
    workspace_hash = _get_dir_hash(prefix)
    origin_dict = _get_requirements_dict(FLEDGE_HOME / f'requirements.{workspace_hash}.txt')
    current_dict = _get_requirements_dict(requirements_filename)
    export_requirements_filename = 'requirements.export.txt'
    with open(export_requirements_filename, "w") as f:
        for package, version in current_dict.items():
            if package not in origin_dict or version != origin_dict[package]:
                # we save only the difference between original workspace after 'fx create workspace' 
                # and current one.
                echo(f'Writing {package}=={version} to {requirements_filename}...')
                f.write(f'{package}=={version}\n')
            elif version is None: # local dependency
                warn(f'Could not generate requirements for {package}. Consider installing it manually after workspace import.')
    echo(f'{export_requirements_filename} written.')

    archiveType = 'zip'
    archiveName = basename(getcwd())
    archiveFileName = archiveName + '.' + archiveType

    # Aggregator workspace
    tmpDir = join(mkdtemp(), 'fledge', archiveName)

    ignore = ignore_patterns('__pycache__', '*.crt', '*.key', '*.csr', '*.srl', '*.pem', '*.pbuf')


    # We only export the minimum required files to set up a collaborator 
    makedirs(f'{tmpDir}/save', exist_ok=True)
    makedirs(f'{tmpDir}/logs', exist_ok=True)
    makedirs(f'{tmpDir}/data', exist_ok=True)
    copytree('./code', f'{tmpDir}/code', ignore=ignore) # code
    copytree('./cert/config', f'{tmpDir}/cert/config', ignore=ignore) # cert
    copytree('./plan', f'{tmpDir}/plan', ignore=ignore) # plan
    copy2(export_requirements_filename, f'{tmpDir}/requirements.txt') # requirements
    copy2('.workspace', tmpDir) # .workspace
   
    make_archive(archiveName, archiveType, tmpDir)      # Create Zip archive of directory

    echo(f'Workspace exported to archive: {archiveFileName}')

@workspace.command(name = 'import')
@option('--archive', required = True, help = 'Zip file containing workspace to import', type = ClickPath(exists=True))
def import_(archive):
    """Import federated learning workspace"""

    from shutil   import unpack_archive
    from os.path  import isfile, basename
    from os       import chdir
    
    dirPath = basename(archive).split('.')[0]
    unpack_archive(archive, extract_dir=dirPath)
    chdir(dirPath)

    requirements_filename = "requirements.txt"

    if isfile(requirements_filename):
        check_call([executable, "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        echo("No " + requirements_filename + " file found.")

    echo(f'Workspace {archive} has been imported.')
    echo(f'You may need to copy your PKI certificates to join the federation.')


@workspace.command(name='certify')
def certify_():
    certify()

def certify():
    '''Create certificate authority for federation'''

    echo('Setting Up Certificate Authority...\n')

    echo('1.  Create Root CA')
    echo('1.1 Create Directories')

    (PKI_DIR / 'ca/root-ca/private').mkdir(parents = True, exist_ok = True, mode = 0o700)
    (PKI_DIR / 'ca/root-ca/db'     ).mkdir(parents = True, exist_ok = True)

    echo('1.2 Create Database')

    with open(PKI_DIR / 'ca/root-ca/db/root-ca.db',      'w') as f: pass # write empty file
    with open(PKI_DIR / 'ca/root-ca/db/root-ca.db.attr', 'w') as f: pass # write empty file

    with open(PKI_DIR / 'ca/root-ca/db/root-ca.crt.srl', 'w') as f: f.write('01') # write file with '01'
    with open(PKI_DIR / 'ca/root-ca/db/root-ca.crl.srl', 'w') as f: f.write('01') # write file with '01'

    echo('1.3 Create CA Request')

    root_conf = 'config/root-ca.conf'
    root_csr  = 'ca/root-ca.csr'
    root_crt  = 'ca/root-ca.crt'
    root_key  = 'ca/root-ca/private/root-ca.key'

    vex(f'openssl req -new '
        f'-config {root_conf} '
        f'-out {root_csr} '
        f'-keyout {root_key}', workdir = PKI_DIR)

    echo('1.4 Create CA Certificate')

    vex(f'openssl ca -batch -selfsign '
        f'-config {root_conf} '
        f'-in {root_csr} '
        f'-out {root_crt} '
        f'-extensions root_ca_ext', workdir = PKI_DIR)

    echo('2.  Create Signing Certificate')
    echo('2.1 Create Directories')

    (PKI_DIR / 'ca/signing-ca/private').mkdir(parents = True, exist_ok = True, mode = 0o700)
    (PKI_DIR / 'ca/signing-ca/db'     ).mkdir(parents = True, exist_ok = True)

    echo('2.2 Create Database')

    with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.db',      'w') as f: pass # write empty file
    with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.db.attr', 'w') as f: pass # write empty file

    with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.crt.srl', 'w') as f: f.write('01') # write file with '01'
    with open(PKI_DIR / 'ca/signing-ca/db/signing-ca.crl.srl', 'w') as f: f.write('01') # write file with '01'

    echo('2.3 Create Signing Certificate CSR')

    signing_conf = 'config/signing-ca.conf'
    root_conf    = 'config/root-ca.conf'
    signing_csr  = 'ca/signing-ca.csr'
    signing_crt  = 'ca/signing-ca.crt'
    signing_key  = 'ca/signing-ca/private/signing-ca.key'

    vex(f'openssl req -new '
        f'-config {signing_conf} '
        f'-out {signing_csr} '
        f'-keyout {signing_key}', workdir = PKI_DIR)

    echo('2.4 Sign Signing Certificate CSR')

    vex(f'openssl ca -batch '
        f'-config {root_conf} '
        f'-in {signing_csr} '
        f'-out {signing_crt} '
        f'-extensions signing_ca_ext', workdir = PKI_DIR)

    echo('3   Create Certificate Chain')

  # create certificate chain file by combining root-ca and signing-ca
    with open(PKI_DIR / 'cert_chain.crt', 'w') as d:
        with open(PKI_DIR / 'ca/root-ca.crt'   ) as s: d.write(s.read())
        with open(PKI_DIR / 'ca/signing-ca.crt') as s: d.write(s.read())

 
    echo('\nDone.')

def _get_requirements_dict(txtfile):
    with open(txtfile, 'r') as snapshot:
        snapshot_dict = {}
        for line in snapshot:
            try:
                k,v = line.split('==') # 'pip freeze' generates requirements with exact versions
                snapshot_dict[k] = v
            except ValueError:
                snapshot_dict[line] = None
        return snapshot_dict

def _get_dir_hash(path):
    from hashlib import md5
    hash_ = md5()
    hash_.update(path.encode('utf-8'))
    hash_ = hash_.hexdigest()
    return hash_
