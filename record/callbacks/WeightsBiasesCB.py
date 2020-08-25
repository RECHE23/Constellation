# -*- coding: utf-8 -*-

from poutyne.framework import Callback


class WeightsBiasesCB(Callback):
    def __init__(self, base_callback):
        super().__init__()
        self.base_callback = base_callback
        self.record = self.base_callback.record

    def on_train_begin(self, logs):
        self.record[-1]['wb'] = self.model.network.state_dict()

    def on_train_batch_end(self, batch, logs):
        self.record[-1]['wb'] = self.model.network.state_dict()
