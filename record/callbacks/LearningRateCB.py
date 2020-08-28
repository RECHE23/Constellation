# -*- coding: utf-8 -*-

from poutyne.framework import Callback


class LearningRateCB(Callback):
    """
    The callback that saves the current learning rate.
    """
    def __init__(self, base_callback):
        """
        Constructor of the class.

        :param base_callback: The base callback.
        """
        super().__init__()
        self.base_callback = base_callback
        self.record = self.base_callback.record

    def on_train_begin(self, logs):
        """
        The method called when the training begins.
        """
        self.record[-1]['lr'] = self.model.optimizer.param_groups[0]['lr']

    def on_train_batch_end(self, batch, logs):
        """
        The method called when the training on a batch is completed.
        """
        self.record[-1]['lr'] = self.model.optimizer.param_groups[0]['lr']
