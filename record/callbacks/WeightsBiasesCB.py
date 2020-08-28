# -*- coding: utf-8 -*-

from poutyne.framework import Callback


class WeightsBiasesCB(Callback):
    """
    The callback that saves the weights and biases of the neural network.
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
        self.record[-1]['wb'] = self.model.network.state_dict()

    def on_train_batch_end(self, batch, logs):
        """
        The method called when the training on a batch is completed.
        """
        self.record[-1]['wb'] = self.model.network.state_dict()
