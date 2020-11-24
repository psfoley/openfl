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
    copyfile(WORKSPACE / 'workspace' / '.workspace', prefix / '.workspace')

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
    if not FLEDGE_USERDIR.exists():
        FLEDGE_USERDIR.mkdir()

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
    with open(FLEDGE_USERDIR / f'requirements.{prefix_hash}.txt', 'w') as f:
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
    origin_dict = _get_requirements_dict(FLEDGE_USERDIR / f'requirements.{workspace_hash}.txt')
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

    try:
        copy2('.workspace', tmpDir) # .workspace
    except FileNotFoundError:
        echo('\'.workspace\' file not found.')
        if confirm('Create a default \'.workspace\' file?'):
            copy2(WORKSPACE / 'workspace' / '.workspace', tmpDir)   
        else:
            echo('To proceed, you must have a \'.workspace\' file in the current directory.')
            raise

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

@workspace.command(name='dockerize')
@option('--save',required = False, help = 'Save the Docker image into the workspace', is_flag=True)
def dockerize_(save):
    '''Package FL.Edge and the workspace as a Docker image'''


    import os
    import subprocess
    import docker
    from shutil import copy
    from shutil import move
    from shutil import rmtree
    from os.path import basename

    WORKSPACE_PATH   = os.getcwd()
    WORKSPACE_PARENT = Path(WORKSPACE_PATH).parent

             
    ## ~TMP dir
    dirname = ".docker_tmp"
    (SITEPACKS / dirname     ).mkdir(parents = True, exist_ok = True)

    DOCKER_TMP = os.path.join(SITEPACKS, dirname)
    
    ## Move relevant files into the tmp dir
    # FL.edge BIN files
    # paths definition
    fledge_libs = str(SITEPACKS)
    fledge_bin  = get_fx_path(fledge_libs)
    copy(fledge_bin, DOCKER_TMP)
    
    fx_file = os.path.join(DOCKER_TMP,'fx')
    replace_line_in_file('#!/usr/bin/python3\n',0,fx_file)

    # Create fl_docker_requirements.txt file
    docker_requirements = "requirements-docker.txt"
    docker_dir = 'fledge-docker'
    copy(os.path.join(SITEPACKS, docker_dir, docker_requirements), DOCKER_TMP)

    # Workspace content
    #copytree(WORKSPACE_PATH,os.path.join(DOCKER_TMP,"workspace"), dirs_exist_ok = True)

    ### Docker BUILD COMMAND
    # Define "build_args". These are the equivalent of the "--build-arg" passed to "docker build"
    build_args = {'DOCKER_TMP': dirname}
    # Add here custom build_args for the build command
    # i.e: build_args["CUSTOM_BUILD_ARG"] = custom_build_arg_var


    # Retrieve args from env vars
    check_varenv('http_proxy', build_args)
    check_varenv('HTTP_PROXY', build_args)
    check_varenv('HTTPS_PROXY',build_args)
    check_varenv('https_proxy',build_args)
    check_varenv('socks_proxy',build_args)
    check_varenv('ftp_proxy',  build_args)
    check_varenv('no_proxy',   build_args)


    ## Compose "build cmd"
    workspace_name = basename(WORKSPACE_PATH)
    fledge_img_base='fledge'
    fledge_img_name=f'{fledge_img_base}/docker_{workspace_name}'

    # Clone Dockerfile within SITEPACKS
    docker_dir   = "fledge-docker"
    base_df      = "Dockerfile.base"
    workspace_df = "Dockerfile.workspace"

    #Copy base dockerfile
    src = os.path.join(SITEPACKS, docker_dir, base_df )
    dst = os.path.join(SITEPACKS, base_df)
    copy(src,dst)
    
    #Copy workspace dockerfile to directory above workspace
    src = os.path.join(SITEPACKS, docker_dir, workspace_df )
    dst = os.path.join(WORKSPACE_PATH, workspace_df)
    copy(src,dst)

    client = docker.from_env(timeout=3600)
    
    ## Build the image
    try:
        
        os.chdir(SITEPACKS)
        if f'{fledge_img_base}:latest' in str(client.images.list()):
            if confirm(f'Base image {fledge_img_base} found. Would you like to rebuild?'):
                echo(f'Building docker image {fledge_img_base}. This may take 5-10 minutes')   
                client.images.build(path=str(SITEPACKS),tag=fledge_img_base,buildargs=build_args,dockerfile=base_df)
        else:
            echo(f'Building docker image {fledge_img_base}. This may take 5-10 minutes')   
            client.images.build(path=str(SITEPACKS),tag=fledge_img_base,buildargs=build_args,dockerfile=base_df)

        echo(f'Building docker image {fledge_img_name}. This will likely take 5-10 minutes')   
        os.chdir(WORKSPACE_PARENT)
        client.images.build(path=str(WORKSPACE_PATH),tag=fledge_img_name,buildargs=build_args,dockerfile=workspace_df)

    except:
        rmtree(DOCKER_TMP)
        os.remove(os.path.join(SITEPACKS,base_df))
        os.remove(os.path.join(WORKSPACE_PATH,workspace_df))
        raise Exception("Error found while building the image. Aborting!")
        exit()
    
    echo(f'Docker image {fledge_img_name} successfully built')


    # Clean environment
    rmtree(DOCKER_TMP)
    os.remove(os.path.join(SITEPACKS,base_df))
    os.remove(os.path.join(WORKSPACE_PATH,workspace_df))

    ## Produce .tar file containing the freshly built image
    if save:
        archive_fn = f'docker_{workspace_name}.tar'

        os.chdir(WORKSPACE_PATH)
        echo('Saving Docker image...')   
        image = client.images.get(f'{fledge_img_name}')
        resp = image.save()
        f = open(archive_fn,'wb')
        for chunk in resp:
            f.write(chunk)
        f.close()

        echo(f'{fledge_img_name} saved to {WORKSPACE_PATH}/{archive_fn}')   
