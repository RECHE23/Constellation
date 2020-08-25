# -*- coding: utf-8 -*-

import joblib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from record import *
from ..plotting import *


def animate_single_depiction(filename, framerate, representations,
                             animations_location):
    full_path = f'{RECORDS_PATH}{filename}'
    filename = filename[:-len(RECORD_EXTENSION)]
    experiment = joblib.load(full_path)
    model_name = experiment['model_name']
    dataset_name = experiment['dataset_name']
    rec = experiment['record']
    if representations == ['all']:
        reprs = experiment['representations']
    else:
        reprs = representations
    for j, representation in enumerate(reprs):
        plt.close(plt.gcf())
        fig, ax = plt.subplots(figsize=(14, 14), dpi=DPI)
        fig.suptitle(t=f'{model_name} - {dataset_name}', x=0.5, y=0.95,
                     ha='center', va='top',
                     fontsize=34,
                     fontweight='normal')
        epochs = rec[-1]['epoch']
        epoch_txt = fig.text(x=0.896, y=0.06, s=' ',
                             fontsize=24, fontweight='bold',
                             ha='right', va='bottom',
                             transform=fig.transFigure)

        i_d = InputData(experiment, 0,
                        reduction=representation,
                        show_badge=True)
        d_b = DecisionBoundaries(i_d)
        d_b.plot(ax)

        def update(k):
            display_progress_bar(j + k / len(rec),
                                 total=len(reprs),
                                 prefix=f'Working: {filename}',
                                 suffix='',
                                 length=60)
            d_b.update(k)
            d_b.plot(ax)
            epoch = rec[k]['epoch']
            epoch_label = f'Epoch {epoch} / {epochs}'
            epoch_txt.set_text(epoch_label)

        anim = FuncAnimation(fig, update, frames=len(rec), interval=1)
        anim_path = f'{animations_location}{filename}.' \
                    f'{representation}{ANIMATION_EXTENSION}'
        anim.save(anim_path, dpi=DPI, fps=framerate,
                  writer=ANIMATION_WRITER)
    display_progress_bar(10, total=10, prefix='Done!', suffix='', length=100)
