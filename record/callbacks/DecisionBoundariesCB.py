# -*- coding: utf-8 -*-

import numpy as np
from poutyne.framework import Callback
from poutyne.utils import torch_to_numpy, numpy_to_torch
from ..projections import projections_list
from .. import *


class DecisionBoundariesCB(Callback):
    def __init__(self, base_callback, dataset, representations):
        representations = [representation.lower()
                           for representation in representations]
        assert all([representation in projections_list
                    for representation in representations])

        super().__init__()
        self.base_callback = base_callback
        self.dataset = dataset
        self.data = dataset.data
        self.targets = torch_to_numpy(dataset.targets)
        self.record = self.base_callback.record
        self.experiment = self.base_callback.experiment

        self.projections = {representation: projections_list[representation](n_components=self.data.shape[1])
                            for representation in representations}
        self.representations = {representation: dict()
                                for representation in representations}

        for representation, content in self.representations.items():
            X_plot = self._get_x_plot(torch_to_numpy(self.data), representation)
            grid_xy = self._get_mesh_grid(X_plot)
            content['X_plot'] = X_plot
            content['grid_xy'] = grid_xy
            content['grid'] = self._get_grid(grid_xy, representation)

        self.experiment['data'] = torch_to_numpy(dataset.data)
        self.experiment['targets'] = torch_to_numpy(dataset.targets)
        self.experiment['representations'] = self.representations

    def on_train_begin(self, logs):
        self.record[-1]['z'] = {representation: self._get_z(representation)
                                for representation in self.representations}

    def on_train_batch_end(self, batch, logs):
        self.record[-1]['z'] = {representation: self._get_z(representation)
                                for representation in self.representations}

    def _get_z(self, representation):
        grid = self.representations[representation]['grid']
        grid = numpy_to_torch(grid).float()
        shape = self.representations[representation]['grid_xy'][0].shape
        z = self.model.predict_on_batch(grid)

        return z.reshape(shape)

    def _get_x_plot(self, data, representation):
        projection = self.projections[representation]

        if representation == 'lda':
            return projection.fit_transform(X=data, y=self.targets)

        return projection.fit_transform(X=data)

    def _get_grid(self, grid_xy, representation):
        projection = self.projections[representation]
        grid_x, grid_y = grid_xy
        grid = np.c_[grid_x.ravel(), grid_y.ravel()]

        if representation != 'umap':
            grid_full = np.zeros((grid.shape[0], self.data.shape[1]))
            grid_full[:, 0:grid.shape[1]] = grid[:, 0:grid.shape[1]]
            grid = grid_full

        return projection.inverse_transform(grid)

    @staticmethod
    def _get_mesh_grid(x_plot):
        x = set(x_plot[:, 0])
        y = set(x_plot[:, 1])
        x_span = np.linspace(min(x), max(x), num=RESOLUTION)
        y_span = np.linspace(min(y), max(y), num=RESOLUTION)

        return np.meshgrid(x_span, y_span)
