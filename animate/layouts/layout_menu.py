# -*- coding: utf-8 -*-

import re
import click
from matplotlib.animation import FuncAnimation
from simple_term_menu import TerminalMenu
from record import *
from ..utils import *
from ..plotting import InputData, DecisionBoundaries, GridLayout


def layout_menu():
    title = click.prompt('Please enter the title of the figure', type=str,
                         default=None)
    filename = re.sub(r"[^\w.]", '_', title)
    filename = click.prompt('Please enter the name of the animation file',
                            type=str, default=filename)
    filename = re.sub(r"[^\w.]", '_', filename)
    eps = click.prompt('Please enter the number of epochs per second',
                       type=int, default=3)

    print("Loading the experiments... It may take a while.")

    horizontal_entries = ['Models', 'Datasets', 'Representations']
    horizontal_axis = TerminalMenu(menu_entries=horizontal_entries,
                                   title="Which should be on the "
                                         "horizontal axis?",
                                   menu_cursor='\u2022 ',
                                   menu_cursor_style=(
                                       'fg_gray', 'bold')).show()
    horizontal_axis = horizontal_entries[horizontal_axis]

    vertical_entries = ['Models', 'Datasets', 'Representations']
    vertical_entries.remove(horizontal_axis)
    vertical_axis = TerminalMenu(menu_entries=vertical_entries,
                                 title="Which should be on the "
                                       "vertical axis:",
                                 menu_cursor='\u2022 ',
                                 menu_cursor_style=(
                                     'fg_gray', 'bold')).show()
    vertical_axis = vertical_entries[vertical_axis]

    constraint = ['Models', 'Datasets', 'Representations']
    constraint.remove(horizontal_axis)
    constraint.remove(vertical_axis)
    constraint = constraint[0]

    experiment_loader = ExperimentLoader()

    horizontals = experiment_loader.list(horizontal_axis)
    click.echo(f'The following {horizontal_axis.lower()} are available:')
    for i, horizontal in enumerate(horizontals):
        click.echo(f'({i}) {horizontal}')
    default = ', '.join([str(i) for i, _ in enumerate(horizontals)][:7])
    selection = None
    while not selection:
        temp = click.prompt(f'Select the {horizontal_axis.lower()} to use '
                            f'(in the order to be displayed on the '
                            f'horizontal axis)',
                            type=str,
                            default=default)
        temp = list(map(int, re.sub(r"[^\d+]", ' ', temp).split()))
        if any(s not in range(len(horizontals)) for s in temp):
            click.echo('Invalid selection!')
        else:
            selection = temp
    selected_horizontals = [horizontals[i] for i in selection]
    ret = experiment_loader.filter(**{horizontal_axis.lower(): selected_horizontals})

    verticals = experiment_loader.list_from(ret, vertical_axis)
    click.echo(f'The following {vertical_axis.lower()} are available:')
    for i, vertical in enumerate(verticals):
        click.echo(f'({i}) {vertical}')
    default = ', '.join([str(i) for i, _ in enumerate(verticals)][:3])
    selection = None
    while not selection:
        temp = click.prompt(f'Select the {vertical_axis.lower()} to use '
                            f'(in the order to be displayed on the '
                            f'vertical axis)',
                            type=str,
                            default=default)
        temp = list(map(int, re.sub(r"[^\d+]", ' ', temp).split()))
        if any(s not in range(len(verticals)) for s in temp):
            click.echo('Invalid selection!')
        else:
            selection = temp
    selected_verticals = [verticals[i] for i in selection]
    ret = experiment_loader.filter(**{vertical_axis.lower(): selected_verticals,
                                      horizontal_axis.lower(): selected_horizontals})

    available_constraints = experiment_loader.list_from(ret, constraint)
    if len(available_constraints) == 0:
        selected_constraint = None
        print(f'No {constraint.lower()[-1]} matching these constraints '
              f'available to build the layout...')
        exit()
    elif len(available_constraints) == 1:
        selected_constraint = available_constraints[0]
    else:
        selected_constraint = TerminalMenu(menu_entries=available_constraints,
                                           title=f"Which {constraint} should "
                                                 f"be used?",
                                           menu_cursor='\u2022 ',
                                           menu_cursor_style=(
                                               'fg_gray', 'bold')).show()
        selected_constraint = available_constraints[selected_constraint]

    cols = selected_horizontals
    rows = selected_verticals

    matrix = []
    epochs = []
    for i, vertical in enumerate(selected_verticals):
        row = []
        for j, horizontal in enumerate(selected_horizontals):
            display_progress_bar(i * len(cols) + j,
                                 total=len(cols) * len(rows),
                                 prefix=f'Building layout: {filename}',
                                 suffix='',
                                 length=60)
            experiment = experiment_loader.get(
                **{horizontal_axis.lower()[:-1]: horizontal,
                   vertical_axis.lower()[:-1]: vertical,
                   constraint.lower()[:-1]: selected_constraint})
            epochs.append(experiment['record'][-1]['epoch'])
            if constraint == 'Representations':
                reduction = selected_constraint
                show_badge = True
            elif horizontal_axis == 'Representations':
                reduction = horizontal
                show_badge = False
                cols = list(map(lambda x: re.sub(r"_", ' ', x.upper()), cols))
            elif vertical_axis == 'Representations':
                reduction = vertical
                show_badge = False
                rows = list(map(lambda x: re.sub(r"_", ' ', x.upper()), rows))
            else:
                reduction = None
                show_badge = False
                print('Something went wrong!')
                exit()
            i_d = InputData(experiment, 0,
                            reduction=reduction,
                            show_badge=show_badge)
            d_b = DecisionBoundaries(i_d)
            row.append(d_b)
        matrix.append(row)

    epochs = min(epochs)
    frames = round(epochs * FRAMERATE / eps)

    layout = GridLayout(matrix, frames, title=title, cols=cols, rows=rows)

    def update(k):
        epoch = round(k / FRAMERATE * eps)

        display_progress_bar(k,
                             total=frames,
                             prefix=f'Animating layout: {filename}',
                             suffix='',
                             length=60)
        for row in layout.matrix:
            for db in row:
                db.update(round(k / frames * db.input_data.experiment['record'][-1]['batch'] * epochs))

        layout.update(epoch=epoch,
                      epochs=epochs)

    fig = layout.fig
    anim = FuncAnimation(fig, update, frames=frames, interval=1)
    anim_path = f'{ANIMATIONS_PATH}{filename}{ANIMATION_EXTENSION}'
    anim.save(anim_path, dpi=DPI, fps=22,
              writer=ANIMATION_WRITER)
    display_progress_bar(10, total=10, prefix='Done!', suffix='', length=100)
