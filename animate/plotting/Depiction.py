# -*- coding: utf-8 -*-
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib import patheffects
from sys import float_info
from record import *

eps = float_info.epsilon


class Depiction(object):
    """
    An abstract class for displaying content in layouts.
    """

    def __init__(self, name: str = 'Depiction',
                 *args, **kwargs) -> None:
        """
        Constructor of the class.

        :param name: The name of what's being shown.
        :param args: Positional arguments.
        :param kwargs: Named arguments.
        """
        self.type = 'Depiction'
        self.name = name

    def plot(self, subplot: Axes) -> None:
        """
        Draws the content of the depiction in the subplot.

        :param subplot: The subplot where the content should be drawn.
        """
        pass

    @staticmethod
    def draw_text(subplot: Axes,
                  text: str, position: (float, float),
                  va: str = 'bottom',
                  ha: str = 'left') -> None:
        """
        Draws text on the subplot.

        :param subplot: The subplot where the content should be drawn.
        :param text: The text to be drawn.
        :param position: The position along the xy axis.
        :param va: Vertical alignment.
        :param ha: Horizontal alignment.
        """
        txt = subplot.text(*position, text,
                           horizontalalignment=ha,
                           verticalalignment=va,
                           fontsize=20,
                           color='w', weight='bold',
                           transform=subplot.transAxes)
        txt.set_path_effects(
            [patheffects.withStroke(linewidth=2, foreground='k')])


class InputData(Depiction):
    """
    A class for depicting the content of a dataset as a two dimensional scatter
    plot.
    """

    def __init__(self, experiment, frame, name: str = 'Input data',
                 reduction: str = 'none', show_badge: bool = True) -> None:
        """
        Constructor of the class.

        :param experiment:
        :param frame:
        :param name: The name of what's being shown.
        :param reduction:
        :param show_badge:
        """
        super(InputData, self).__init__(name=name)
        self.type = 'Input data'
        self.reduction = reduction.lower()
        self.reduction_txt = re.sub(r"_", ' ', self.reduction).upper()
        self.experiment = experiment
        self.frame = frame
        self.representation = experiment['representations'][reduction]
        self.rec = experiment['record']
        self.X_plot = self.representation['X_plot']
        self.y_plot = experiment['targets']
        self.grid_x, self.grid_y = self.representation['grid_xy']
        if reduction.lower() == 'none' or reduction.lower() == 'original':
            self.show_badge = False
        else:
            self.show_badge = show_badge

    def plot(self, subplot: Axes) -> None:
        """
        Draws the content of the depiction in the subplot.

        :param subplot: The subplot where the content should be drawn.
        """
        subplot.clear()
        self.scatter_plot(subplot)
        if self.show_badge:
            self.plot_badge(subplot)

    def scatter_plot(self, subplot: Axes) -> None:
        """
        Draws the elements of the dataset as a scatter plot.

        :param subplot: The subplot where the content should be drawn.
        """
        fill_colors = np.zeros((self.X_plot.shape[0], 4))
        edge_colors = np.zeros((self.X_plot.shape[0], 4))
        distance = np.zeros((self.X_plot.shape[0], 1))
        for i, x in enumerate(self.X_plot):
            distance[i] = np.linalg.norm(x[2:])
        distance *= 12 / (max(distance) + eps)
        for i, x in enumerate(self.X_plot):
            alpha = min(1, 1 / (distance[i] + eps))
            fill_colors[i, 0] = 0.267004 if self.y_plot[i] == 1 else 0.993248
            fill_colors[i, 1] = 0.004874 if self.y_plot[i] == 1 else 0.906157
            fill_colors[i, 2] = 0.329415 if self.y_plot[i] == 1 else 0.143936
            fill_colors[i, 3] = alpha
            edge_colors[i, 3] = alpha * 0.6
        scale = 60 * fill_colors[:, 3]
        subplot.scatter(self.X_plot[:, 0], self.X_plot[:, 1], s=scale,
                        c=self.y_plot.ravel(),
                        edgecolors=edge_colors)
        a = self.grid_x.min()
        b = self.grid_x.max()
        c = self.grid_y.min()
        d = self.grid_y.max()
        subplot.set_xlim(a, b)
        subplot.set_ylim(c, d)
        aspect = (b - a) / (d - c)
        subplot.set_aspect(aspect)
        subplot.set_xticks(())
        subplot.set_yticks(())

    def plot_badge(self, subplot: Axes) -> None:
        """
        Draws a badge indicating which dimensionality reduction is applied.

        :param subplot: The subplot where the content should be drawn.
        """
        txt = subplot.text(0.04, 0.96,
                           self.reduction_txt,
                           horizontalalignment='left',
                           verticalalignment='top', fontsize=12,
                           color='w', weight='bold',
                           transform=subplot.transAxes)
        bbox = dict(boxstyle="round4,pad=0.5", ec="w", fc="r", alpha=1)
        plt.setp(txt, bbox=bbox)


class DecisionBoundaries(Depiction):
    """
    A class for depicting the decision boundaries of a neural network model.
    """

    def __init__(self, input_data: InputData, name: str = None) -> None:
        """
        Constructor of the class.

        :param input_data: An instance of InputData containing the dataset.
        :param name: The name of what's being shown.
        """
        super(DecisionBoundaries, self).__init__(name=name)
        self.type = 'Model outcome'
        self.input_data = input_data
        self.X_plot = self.input_data.X_plot
        self.grid_x = self.input_data.grid_x
        self.grid_y = self.input_data.grid_y
        self.y_plot = self.input_data.y_plot
        self.z_plot = self.input_data.rec[0]['z'][self.input_data.reduction]
        self.score = self.input_data.rec[0]['accuracy']
        self.loss = self.input_data.rec[0]['loss']
        self.lr = self.input_data.rec[0]['lr']

    def plot(self, subplot: Axes) -> None:
        """
        Draws the content of the depiction in the subplot.

        :param subplot: The subplot where the content should be drawn.
        """
        subplot.clear()
        self.input_data.scatter_plot(subplot)
        self.contour_plot(subplot)
        if self.score is not None:
            score_txt = ('Acc: %.2f' % self.score).lstrip('0')
            self.draw_text(subplot, score_txt,
                           position=(0.96, 0.03), ha='right', va='bottom')
        if self.loss is not None:
            loss_txt = ('Loss: %.2f' % self.loss).lstrip('0')
            self.draw_text(subplot, loss_txt,
                           position=(0.04, 0.03), ha='left', va='bottom')
        if self.lr is not None:
            lr_txt = ('LR: %.4f' % self.lr).lstrip('0')
            self.draw_text(subplot, lr_txt,
                           position=(0.96, 0.96), ha='right', va='top')
        if self.input_data.show_badge:
            self.input_data.plot_badge(subplot)

    def contour_plot(self, subplot: Axes) -> None:
        """
        Draws the decision boundaries as a contour plot. The boundaries are
        drawn as red lines and the color gradient represents the value
        outputted by the network.

        :param subplot: The subplot where the content should be drawn.
        """

        norm = cm.colors.Normalize(vmin=-1.0, vmax=1.0)
        subplot.contourf(self.grid_x, self.grid_y, self.z_plot,
                         zorder=0, vmin=-1.0, vmax=1.0,
                         norm=norm, alpha=.95)
        warnings.filterwarnings("ignore")
        subplot.contour(self.grid_x, self.grid_y, self.z_plot,
                        levels=[-.0001, 0.0001], colors='r')
        warnings.filterwarnings("default")

    def update(self, frame):
        self.z_plot = self.input_data.rec[frame]['z'][self.input_data.reduction]
        self.score = self.input_data.rec[frame]['accuracy']
        self.loss = self.input_data.rec[frame]['loss']
        self.lr = self.input_data.rec[frame]['lr']
