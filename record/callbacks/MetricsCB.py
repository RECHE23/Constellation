# -*- coding: utf-8 -*-

from poutyne.framework import Callback


class MetricsCB(Callback):
    def __init__(self, base_callback, batch_metrics):
        super().__init__()
        self.base_callback = base_callback
        self.batch_metrics = batch_metrics
        self.record = self.base_callback.record

    def on_train_begin(self, logs):
        self.record[-1]['loss'] = None
        for metric in self.batch_metrics:
            self.record[-1][metric] = None

    def on_train_batch_end(self, batch, logs):
        self.record[-1]['loss'] = logs['loss']
        for metric in self.batch_metrics:
            self.record[-1][metric] = logs[metric]
