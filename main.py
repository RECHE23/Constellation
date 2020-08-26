# -*- coding: utf-8 -*-

import os
import re
import click
from natsort import natsorted
from poutyne import set_seeds
from simple_term_menu import TerminalMenu
from animate import *
from animate.layouts import layout_menu
from record import *


@click.group()
def cli():
    pass


@cli.command()
@click.option('-s', '--seed', default=None, type=int,
              help='Seed for reproducibility.')
@click.option('--use_seed/--no_seed', default=False,
              help='Use a seed for reproducibility?')
@click.option('-d', '--device', default=None, type=str,
              help='Torch device to use.')
@click.option('-e', '--epochs', default=EPOCHS, type=int,
              help='Number of epochs for training.')
@click.option('-b', '--batch_size', default=BATCH_SIZE, type=int,
              help='Size of batches for training.')
@click.option('-r', '--representations', default='pca,lda,original', type=str,
              help='Representations to use: Original, PCA, LDA, Fast_ICA, '
                   'Kernel_PCA, and/or UMAP')
@click.option('-m', '--metrics', default='accuracy', type=str,
              help='Metrics to use: accuracy, gma, bce_with_logits, l1, mse, '
                   'smooth_l1, and/or soft_margin')
def record(seed, use_seed, device, epochs, batch_size,
           representations, metrics):
    """
    Start and record the learning phase of the neural network.
    """
    from user_data import datasets, models
    if not os.path.exists(RECORDS_PATH):
        os.mkdir(RECORDS_PATH)

    if use_seed:
        if seed is None:
            seed = DEFAULT_SEED
        else:
            click.echo(f'Seed set to {seed} for Python, Numpy and PyTorch.\n')
        set_seeds(seed)
    else:
        click.echo('Not using a seed. Results will not be reproducible...\n')

    device = set_device(device)

    click.echo('Loaded models:')
    for model in models.keys():
        click.echo(f'- {model}')
    click.echo('')

    click.echo('Loaded datasets:')
    for ds in datasets:
        click.echo(f'- {ds.name}')
    click.echo('')

    if metrics == 'all':
        metrics = metrics_list.keys()
    else:
        metrics = re.sub(r"[^\w]", ' ', metrics).split()
    click.echo('Metrics:')
    for metric in metrics:
        click.echo(f'- {metric}')
    click.echo('')

    if representations == 'all':
        representations = list(projections_list.keys())
    else:
        representations = re.sub(r"[^\w]", ' ', representations).split()
    click.echo('Representations:')
    for representation in representations:
        click.echo(f'- {representation}')
    click.echo('')

    start_record(models, datasets, device, batch_size, representations, epochs,
                 metrics)


@cli.command()
@click.argument('what', default='specified', type=str)
@click.option('-f', '--framerate', default=FRAMERATE, type=int,
              help='Number of frames per second.')
@click.option('-r', '--representations', default='all', type=str,
              help='Representations to use: Original, PCA, LDA, Fast_ICA, '
                   'Kernel_PCA, and/or UMAP')
@click.option('-s', '--specified', default='', type=click.Path(),
              help='Specified record file.')
@click.option('-l', '--animations_location', default=ANIMATIONS_PATH,
              type=str, help='Animations\' location')
def animate(what, framerate, representations, specified, animations_location):
    if not os.path.exists(RECORDS_PATH):
        click.echo(f'The following directory doesn\'t exist:\n{RECORDS_PATH}\n')
        return

    if not os.path.exists(animations_location):
        os.mkdir(animations_location)

    representations = re.sub(r"[^\w]", ' ', representations).split()

    files = os.listdir(RECORDS_PATH)
    files = list(natsorted(files))
    saves = list(filter(
        lambda x: x[-len(RECORD_EXTENSION):] == RECORD_EXTENSION, files))
    if what == 'all':
        for i, name in enumerate(saves):
            animate_single_depiction(filename=name,
                                     framerate=framerate,
                                     representations=representations,
                                     animations_location=animations_location)
    elif what == 'specified':
        if not len(specified):
            files = list(natsorted(os.listdir(RECORDS_PATH)))
            selected = TerminalMenu(menu_entries=files,
                                    title="Which record should be animated?",
                                    menu_cursor='\u2022 ',
                                    menu_cursor_style=(
                                        'fg_gray', 'bold')).show()
            specified = files[selected]
        specified = re.sub(r"[^\w.]", ' ', specified).split()[-1]
        assert specified in saves
        animate_single_depiction(filename=specified,
                                 framerate=framerate,
                                 representations=representations,
                                 animations_location=animations_location)
    elif what == 'layout':
        layout_menu()
    else:
        click.echo(f'Invalid option: {what}')


if __name__ == '__main__':
    cli()
