#!/usr/bin/env python

from click       import Group, CommandCollection, command, argument, group, clear, echo, option, pass_context, style
from sys         import path
from os.path     import join, exists, dirname, realpath, basename, splitext
from logging     import getLogger, basicConfig

from rich.traceback import install as colorTraces
from rich.console   import Console
from rich.logging   import RichHandler

#colorTraces(width = 160)

console = Console(width = 160)

def setup_logging(log_config = 'logging.yaml', level = 'debug'):

    import logging
    import logging.config
    import yaml

    levels = \
    {
        'notset'   : logging.NOTSET,
        'debug'    : logging.DEBUG,
        'info'     : logging.INFO,
        'warning'  : logging.WARNING,
        'error'    : logging.ERROR,
        'critical' : logging.CRITICAL
    }

    level = levels.get(level.lower(), levels['notset'])

    basicConfig(level = level, format = '%(message)s', datefmt = '[%X]', handlers = [RichHandler(console = console)])

class CLI(Group):
    def __init__(self, name = None, commands = None, **kwargs):
        super(CLI, self).__init__(name, commands, **kwargs)
        self.commands = commands or {}

    def list_commands(self, ctx):
        return self.commands

    def format_help(self, ctx, formatter):

        uses = [f'{ctx.command_path}', '[options]', style('[command]', fg = 'blue'), style('[subcommand]', fg = 'cyan'),'[args]']

        formatter.write(style('CORRECT USAGE\n\n', bold = True, fg = 'bright_black'))
        formatter.write(' '.join(uses) + '\n')

        opts = []
        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                opts.append(rv)

        formatter.write(style('\nGLOBAL OPTIONS\n\n', bold = True, fg = 'bright_black'))
        formatter.write_dl(opts)

        cmds = []
        for cmd in self.list_commands(ctx):
            cmd  = self.get_command(ctx, cmd)
            cmds.append((cmd.name, cmd, 0))

            for sub in cmd.list_commands(ctx):
                sub  = cmd.get_command(ctx, sub)
                cmds.append((sub.name, sub, 1))

        formatter.write(style('\nAVAILABLE COMMANDS\n', bold = True, fg = 'bright_black'))

        for name, cmd, level in cmds:
            help = cmd.get_short_help_str()
            if  level == 0:
                formatter.write(f'\n{style(name, fg = "blue", bold = True):<30} {style(help, bold = True)}' + '\n')
                formatter.write(f'─' * 80 + '\n')
            if  level == 1:
                formatter.write(f'  {style("⮞", fg = "green")} {style(name, fg = "cyan"):<21} {help}' + '\n')

@group(cls = CLI)
@option('--log-config', default = 'logging.yaml', help = 'Logging configuration file.')
@option('--log-level',  default = 'info', help = 'Logging verbosity level.')
@pass_context
def cli(context, log_config, log_level):
    '''Command-line Interface'''

    context.ensure_object(dict)

    context.obj['log_config'] = log_config
    context.obj['log_level' ] = log_level
    context.obj['fail'      ] = False

    setup_logging(log_config, log_level)

@cli.resultcallback()
@pass_context
def end(context, result, **kwargs):
    if context.obj['fail']:
        echo('\n ❌ :(')
    else:
        echo('\n ✔️ OK')

@command()
@pass_context
@argument('subcommand', required = False)
def help(context, subcommand):
    pass

def entry():

    clear()
    banner = f'Intel FLedge - Secure Federated Learning at the Edge™'
    echo(style(f'{banner:<80}', bold = True, bg = 'bright_blue'))
    echo()

    from glob      import glob
    from os.path   import dirname, realpath, basename, splitext
    from importlib import import_module
    from sys       import path
    from os        import getcwd

    root = dirname(realpath(__file__)) # interface root, containing command modules
    base = dirname(root)
    work = getcwd()

    path.append(root)
    path.insert(0, work)

    for module in glob(f'{root}/*.py'): # load command modules

        package = dirname(module)
        module  = splitext(basename(module))[0]

        if  module in ['__init__', 'cli', 'cli_helper']:
            continue

        group   = import_module(module, package)
        
        cli.add_command(group.__getattribute__(module))

    try:
        cli()
    except Exception as e:
        echo(style(f'EXCEPTION', fg = 'red', bold = True) + ' : ' + style(f'{e}', fg = 'red'))
        raise e

if  __name__ == '__main__':
    
    entry()
