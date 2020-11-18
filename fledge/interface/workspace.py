from fledge.interface.cli_helper import *

from subprocess import check_call
from sys        import executable

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

    with open(requirements_filename, "w") as f:
        check_call([executable, "-m", "pip", "freeze"], stdout=f)

    echo(f'{requirements_filename} written.')

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
    copy2('requirements.txt', tmpDir) # requirements
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


@workspace.command(name='dockerize')
@option('--compress',required = False, help = 'Compress the docker img into a .zip file saved in workspace/workspace.zip', is_flag=True)
def dockerize_(compress):

    import os
    import subprocess
    from shutil import copy

    ## Retrieve SITEPACKS
    SITEPACKS = Path(__file__).parent.parent.parent
    WORKSPACE_PATH = os.getcwd()

    def get_info(cmd=[]):

        ''' Returns the output of the cmd executed at shell level
            cmd is a list of strings that contains the actual instruction
            followed by the parameters.
            for example, the command "ls -a", would be ['ls','a'] '''

        import subprocess
        result = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8').strip('\n')

        return result


    def check_varenv(env="", args={}):
        ''' Updates "args" (dictionary) with <env: env_value> if env has a defined value in the host'''

        env_val = os.environ.get(env)
        if env and (env_val is not None):
            args[env] = env_val

        return args

    def get_wspaceName(path=""):
        ''' Returns the name of the workspace extracted from path'''

        if (path[-1] == "/"):
            path = path[0:-1]

        return (path.split("/"))[-1]

    def get_binPath(curr_path=""):
       ''' Returns the path to go from current dir to bin/fx'''
       import re
       import os

       match=re.search("lib", curr_path)
       idx = match.end()
       path_prefix = curr_path[0:idx]
       bin_path = os.path.relpath(path_prefix,curr_path) + "/../bin"



       return bin_path
              
    def get_fledgeRequirements(save_path=""):
       ''' Freezes the current virtualenv requirements in the file called:
               fl_docker_requirements.txt'''

       import os
       filename = "fl_docker_requirements.txt"
       cmd = "pip freeze > "+ os.path.join(save_path, filename)
       os.system(cmd)
        


    # Clone Dockerfile
    docker_dir="fledge-docker"
    dockerfile_template="Dockerfile_wspace_template"
    tmp_dockerfile="Dockerfile_tmp"

    src = os.path.join(SITEPACKS,docker_dir,dockerfile_template)
    dst = os.path.join(SITEPACKS,docker_dir,tmp_dockerfile)
    copy(src,dst)


    ### Dockerfile CONFIGURATION
    ## Fledge files
    # paths definition
    fledge_libs = str(SITEPACKS)
    fledge_bin  = get_binPath(fledge_libs)

    # Create fl_docker_requirements.txt file
    filename = "fl_docker_requirements.txt"
    filepath = os.path.join(SITEPACKS, filename)
    cmd = "pip freeze > "+ filepath
    os.system(cmd)

    os.system('sed -i "/fledge @ file:/d" ' + filepath)

    ## Workspace files 
    # Get relative WORKSPACE_PATH for the Dockerfile
    dockerfile_to_workspace = os.path.relpath(WORKSPACE_PATH,SITEPACKS)

    # Uncomment WORKSPACE_PATH details in Dockerfile
    os.system('sed -i "s/#__RUN mkdir -p /RUN mkdir -p /g" ' + dst)
    os.system('sed -i "s/#__COPY $WORKSPACE_PATH/COPY $WORKSPACE_PATH/g" ' + dst)
    os.system('sed -i "s/#__RUN pip3 install/RUN pip3 install/g" ' + dst)


    ### Docker BUILD COMMAND
    # Define "build_args". These are the equivalent of the "--build-arg" passed to "docker build"
    username = get_info(['whoami'])
    build_args = {'USERNAME':   username,
                  'USER_ID':    get_info(['id','-u',username]),
                  'GROUP_ID':   get_info(['id','-g',username])
                  }


    # Retrieve args from env vars
    check_varenv('http_proxy', build_args)
    check_varenv('HTTP_PROXY', build_args)
    check_varenv('HTTPS_PROXY',build_args)
    check_varenv('https_proxy',build_args)
    check_varenv('socks_proxy',build_args)
    check_varenv('ftp_proxy',  build_args)
    check_varenv('no_proxy',   build_args)

    # Update list of build args for "docker build" 
    build_args['FLEDGE_BIN_PATH'] = fledge_bin
    build_args['FLEDGE_LIB_PATH'] = fledge_libs
    build_args['WORKSPACE_PATH'] = dockerfile_to_workspace


    ## Compose "build cmd"
    FLEDGE_IMG_NAME="fledge/docker_test"

    args = ['--build-arg '+ var +'='+val for var,val in build_args.items()]
    build_cmd_args = ['docker build'] + args + ['-t', FLEDGE_IMG_NAME,'-f Dockerfile','.']
    build_command = ' '.join(build_cmd_args)


    ## Move the Dockerfile outside fledge dirs
    dockerfile="Dockerfile"
    src = os.path.join(SITEPACKS,docker_dir,tmp_dockerfile)
    #dst = os.path.abspath(os.path.join(SITEPACKS, os.pardir))
    dst=SITEPACKS
    copy(src, dst)
    os.system("mv "+ os.path.join(dst,tmp_dockerfile) + " " + os.path.join(dst,dockerfile))

    ## Build the image
    print("BUILD COMMAND:\n\t",build_command)
    try:
        
        os.chdir(dst)
        if os.system(build_command) != 0:
            raise Exception("Error found while building the image. Aborting!")

    except:
        raise Exception("Error found while building the image. Aborting!")
        exit()
    
    echo('\nDone: Dockerfile successfully built')


    # Clean environment
    cmd='rm '+ os.path.join(SITEPACKS,docker_dir,tmp_dockerfile)
    os.system(cmd)


    ## Produce .tar file containing the freshly built image
    if compress:
        workspace_name = get_wspaceName(WORKSPACE_PATH)
        archiveType = "tar"
        archiveFileName = "docker_"+ workspace_name + "." + archiveType

        os.chdir(WORKSPACE_PATH)
        compress_cmd = 'docker save -o '+ archiveFileName +' '+ FLEDGE_IMG_NAME
        os.system(compress_cmd)

        echo('\nDone: Docker image compressed!')


   
