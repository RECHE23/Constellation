# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from record import *

FIG = (32, 18)      # Main canvas size
DPI = 80            # Pixel density for canvas
DDS = (28, 14)      # Dataset canvas size
TXT = 22            # Annotation font size
TTL = 35            # Title font size
EPC = 25            # Epoch font size


class GridLayout(object):
    """
    A class used to contain and display a layout.
    """
    def __init__(self, matrix: list, frames: int, title: str = None,
                 cols: list = None, rows: list = None,
                 figsize: (int, int) = FIG):
        """
        Constructor of the class.

        :param matrix: A matrix of depictions built using lists.
        :param title: The title displayed at the top of the layout.
        :param cols: The title of the columns.
        :param rows: The title of the rows.
        :param figsize: The size of the canvas.
        """
        self.type = 'Layout'
        self.matrix = matrix
        self.title = title
        self.cols = [] if cols is None else cols
        self.rows = [] if rows is None else rows
        self.figsize = figsize
        self.fig = None
        self.axes = None
        self.epoch_txt = None
        self.title_txt = None
        self.frames = frames
        try:
            plt.close()
            if isinstance(self.matrix[0], list):
                self.fig, self.axes = plt.subplots(nrows=len(self.matrix),
                                                   ncols=len(self.matrix[0]),
                                                   figsize=self.figsize,
                                                   subplot_kw={'xticks': [],
                                                               'yticks': []},
                                                   dpi=DPI)

                self.plot_depictions()
                self.plot_cols_title()
                self.plot_rows_title()

            else:
                self.fig, self.axes = plt.subplots(nrows=1,
                                                   ncols=len(self.matrix),
                                                   figsize=self.figsize,
                                                   subplot_kw={'xticks': [],
                                                               'yticks': []},
                                                   dpi=DPI)
                if len(self.matrix) == 1:
                    self.matrix[0].plot(self.axes)
                else:
                    for i, depiction in enumerate(self.matrix):
                        depiction.plot(self.axes[i])

                if len(self.matrix) == 1:
                    self.axes.set_title(cols[0], fontsize=TXT)
                else:
                    for ax, col in zip(self.axes, cols):
                        ax.set_title(col, fontsize=TXT)

            if self.title is not None:
                self.plot_title()

            self.adjust_layout()

        except KeyboardInterrupt:
            pass

    def update(self, epoch: int = 0, epochs: int = None) -> None:
        """
        Updates the layout for when the models or the depictions are updated.

        :param epoch: The current epoch.
        :param epochs: The total number of epochs.
        """
        try:
            self.plot_depictions()
            self.plot_cols_title()
            self.plot_rows_title()

            self.plot_title(title=self.title)
            self.plot_epoch(epoch=epoch, epochs=epochs)
            self.adjust_layout(mode='video')

        except KeyboardInterrupt:
            pass

    def plot_depictions(self) -> None:
        """
        Plot the depictions contained in the matrix.
        """
        for i, _ in enumerate(self.matrix):
            for j, depiction in enumerate(self.matrix[i]):
                if len(self.axes.shape) > 1:
                    ax = self.axes[i][j]
                else:
                    ax = self.axes[j]
                depiction.plot(ax)

    def plot_cols_title(self) -> None:
        """
        Draws the title for each column.
        """
        if len(self.axes.shape) > 1:
            for ax, col in zip(self.axes[0], self.cols):
                ax.set_title(col, fontsize=TXT - 0.2 * len(col))
        else:
            for ax, col in zip(self.axes, self.cols):
                ax.set_title(col, fontsize=TXT - 0.2 * len(col))

    def plot_rows_title(self) -> None:
        """
        Draws the title for each row.
        """
        if len(self.axes.shape) > 1:
            for ax, row in zip(self.axes[:, 0], self.rows):
                ax.set_ylabel(row, rotation=90,
                              size='large', fontsize=TXT - 0.2 * len(row))
        else:
            for ax, row in zip(self.axes, self.rows):
                ax.set_ylabel(row, rotation=90,
                              size='large', fontsize=TXT - 0.2 * len(row))

    def plot_title(self, title: str = None) -> None:
        """
        Adds or updates the title of the layout.

        :param title: The content of the title. (If None, uses the one defined
                      during initialisation.)
        """
        if self.title_txt is None:
            if title is None:
                title = self.title
            self.title_txt = self.fig.suptitle(t=title, x=0.5, y=0.88,
                                               ha='center', va='top',
                                               fontsize=TTL,
                                               fontweight='normal')
        elif title is not None and self.title != title:
            self.title = title
            self.title_txt.set_text(title)

    def plot_epoch(self, epoch: int = 0, epochs: int = None) -> None:
        """
        Displays the current epoch.

        :param epoch: Current epoch.
        :param epochs: Total number of epochs.
        """
        if epochs is not None:
            epoch_label = f'Epoch {epoch} / {epochs}'
        elif epoch != 0:
            epoch_label = f'Epoch {epoch}'
        else:
            epoch_label = ' '
        if self.epoch_txt is None:
            self.epoch_txt = self.fig.text(x=0.9, y=0.16, s=epoch_label,
                                           fontsize=EPC, fontweight='bold',
                                           ha='right', va='bottom',
                                           transform=self.fig.transFigure)
        else:
            self.epoch_txt.set_text(epoch_label)

    def adjust_layout(self, mode: str = 'preview') -> None:
        """
        Adjusts the sizes and distances in order to improve the look.

        :param mode: The mode that dictates the rules used.
        """
        if isinstance(self.matrix[0], list):
            m = len(self.matrix)
            n = len(self.matrix[0])
        else:
            m = 1
            n = len(self.matrix)
        if mode == 'preview':
            x_epoch = 0.494 + 0.0583 * n
            y_epoch = 0.16
            x_title = 0.5
            y_title = 0.88
            left = 0.485 - 0.055 * n
            right = 0.515 + 0.055 * n
            bottom = 0.5 - 0.1 * m
            top = 0.5 + 0.1 * m
            wspace = -0.025 + 0.015 * n
            hspace = 0.22 - 0.02 * n
        else:
            x_epoch = 0.50 + 0.075 * n if n < 7 else 0.95
            y_epoch = 0.05 if n < 7 else 0.11
            x_title = 0.5
            y_title = 0.99
            left = 0.50 - 0.075 * n if n < 7 else 0.05
            right = 0.50 + 0.075 * n if n < 7 else 0.95
            bottom = 0.08 if n < 7 else 0.14
            top = 0.88 if n < 7 else 0.82
            wspace = -0.025 + 0.015 * n
            hspace = 0.22 - 0.02 * n
        if self.title_txt is not None:
            self.title_txt.set_x(x_title)
            self.title_txt.set_y(y_title)
        if self.epoch_txt is not None:
            self.epoch_txt.set_x(x_epoch)
            self.epoch_txt.set_y(y_epoch)
        self.fig.tight_layout(pad=0)
        self.fig.subplots_adjust(left=left, right=right,
                                 bottom=bottom, top=top,
                                 wspace=wspace, hspace=hspace)
