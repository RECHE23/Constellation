# -*- coding: utf-8 -*-

from poutyne.framework import Callback


class LearningRateCB(Callback):
    def __init__(self, base_callback):
        super().__init__()
        self.base_callback = base_callback
        self.record = self.base_callback.record

    def on_train_begin(self, logs):
        self.record[-1]['lr'] = self.model.optimizer.param_groups[0]['lr']

    def on_train_batch_end(self, batch, logs):
        self.record[-1]['lr'] = self.model.optimizer.param_groups[0]['lr']
