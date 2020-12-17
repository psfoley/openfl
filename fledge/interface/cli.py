#!/usr/bin/env python
# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from click import Group, command, argument, group, clear
from click import echo, option, pass_context, style
from sys import argv
from pathlib import Path


def setup_logging(log_config='logging.yaml', level='debug'):
    from logging import basicConfig, NOTSET, DEBUG, INFO
    from logging import WARNING, ERROR, CRITICAL
    # from rich.traceback import install as colorTraces
    from rich.console import Console
    from rich.logging import RichHandler

    # traces = colorTraces(width=160)
    console = Console(width=160)

    levels = \
        {
            'notset': NOTSET,
            'debug': DEBUG,
            'info': INFO,
            'warning': WARNING,
            'error': ERROR,
            'critical': CRITICAL
        }

    level = levels.get(level.lower(), levels['notset'])

    basicConfig(level=level, format='%(message)s',
                datefmt='[%X]', handlers=[RichHandler(console=console)])


class CLI(Group):
    def __init__(self, name=None, commands=None, **kwargs):
        super(CLI, self).__init__(name, commands, **kwargs)
        self.commands = commands or {}

    def list_commands(self, ctx):
        return self.commands

    def format_help(self, ctx, formatter):

        uses = [
            f'{ctx.command_path}',
            '[options]',
            style('[command]', fg='blue'),
            style('[subcommand]', fg='cyan'),
            '[args]'
        ]

        formatter.write(style(
            'CORRECT USAGE\n\n', bold=True, fg='bright_black'))
        formatter.write(' '.join(uses) + '\n')

        opts = []
        for param in self.get_params(ctx):
            rv = param.get_help_record(ctx)
            if rv is not None:
                opts.append(rv)

        formatter.write(style(
            '\nGLOBAL OPTIONS\n\n', bold=True, fg='bright_black'))
        formatter.write_dl(opts)

        cmds = []
        for cmd in self.list_commands(ctx):
            cmd = self.get_command(ctx, cmd)
            cmds.append((cmd.name, cmd, 0))

            for sub in cmd.list_commands(ctx):
                sub = cmd.get_command(ctx, sub)
                cmds.append((sub.name, sub, 1))

        formatter.write(style(
            '\nAVAILABLE COMMANDS\n', bold=True, fg='bright_black'))

        for name, cmd, level in cmds:
            help = cmd.get_short_help_str()
            if level == 0:
                formatter.write(
                    f'\n{style(name, fg="blue", bold=True):<30}'
                    f' {style(help, bold=True)}' + '\n')
                formatter.write('─' * 80 + '\n')
            if level == 1:
                formatter.write(
                    f'  {style("⮞", fg="green")}'
                    f' {style(name, fg="cyan"):<21} {help}' + '\n')


@group(cls=CLI)
@option('--log-config', default='logging.yaml', help='Logging configuration file.')
@option('--log-level', default='info', help='Logging verbosity level.')
@pass_context
def cli(context, log_config, log_level):
    """Command-line Interface."""
    context.ensure_object(dict)

    context.obj['log_config'] = log_config
    context.obj['log_level'] = log_level
    context.obj['fail'] = False
    context.obj['script'] = argv[0]
    context.obj['arguments'] = argv[1:]

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
@argument('subcommand', required=False)
def help(context, subcommand):
    pass


def error_handler(error):
    if 'cannot import' in str(error):
        if 'TensorFlow' in str(error):
            echo(style('EXCEPTION', fg='red', bold=True) + ' : ' + style(
                'Tensorflow must be installed prior to running this command',
                fg='red'))
        if 'PyTorch' in str(error):
            echo(style('EXCEPTION', fg='red', bold=True) + ' : ' + style(
                'Torch must be installed prior to running this command',
                fg='red'))
    echo(style('EXCEPTION', fg='red', bold=True)
         + ' : ' + style(f'{error}', fg='red'))
    raise error


def entry():
    from importlib import import_module
    from sys import path

    file = Path(__file__).resolve()
    root = file.parent.resolve()  # interface root, containing command modules
    work = Path.cwd().resolve()

    path.append(str(root))
    path.insert(0, str(work))

    clear()
    banner = 'Intel FLedge - Secure Federated Learning at the Edge™'
    echo(style(f'{banner:<80}', bold=True, bg='bright_blue'))
    echo()

    for module in root.glob('*.py'):  # load command modules

        package = module.parent
        module = module.name.split('.')[0]

        if module.count('__init__') or module.count('cli'):
            continue

        command_group = import_module(module, package)

        cli.add_command(command_group.__getattribute__(module))

    try:
        cli()
    except Exception as e:
        error_handler(e)
        # echo(style(f'EXCEPTION', fg = 'red', bold = True) + ' : '
        # + style(f'{e}', fg = 'red'))
        # raise e


if __name__ == '__main__':
    entry()
