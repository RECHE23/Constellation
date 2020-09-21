# -*- coding: utf-8 -*-

import os
import re
import warnings
import joblib
from poutyne.framework import Callback
from .. import *

warnings.simplefilter(action='ignore', category=FutureWarning)


class BaseCB(Callback):
    """
    The base callback. It is the callback that establishes the basis for the
    recordings. It saves the recording in the file at the end of every epoch.
    """
    def __init__(self, model_name, dataset_name,
                 records_path=RECORDS_PATH):
        """
        Constructor of the class.

        :param model_name: The name of the neural network.
        :param dataset_name: The name of the dataset being recorded.
        :param records_path: The location where the file is being saved.
        """
        super().__init__()

        filename = f'{model_name}_{dataset_name}'
        filename = re.sub(r"[^\w.]", '_', filename)

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.records_path = records_path
        self.epoch_number = 0
        self.record = []
        self.experiment = {'model_name': model_name,
                           'dataset_name': dataset_name,
                           'epochs_index': [[0]]}
        self.filename = filename

    def on_train_begin(self, logs):
        """
        The method called when the training begins.
        """
        item = {'epoch': self.epoch_number,
                'batch': 0}

        self.record.append(item)

    def on_train_batch_end(self, batch, logs):
        """
        The method called when the training on a batch is completed.
        """
        item = {'epoch': self.epoch_number,
                'batch': batch}

        self.experiment['epochs_index'][-1].append(len(self.record))
        self.record.append(item)

    def on_epoch_begin(self, epoch_number, logs):
        """
        The method called when a new epoch begins.
        """
        self.epoch_number = epoch_number
        self.experiment['epochs_index'].append([])

    def on_epoch_end(self, epoch_number, logs):
        """
        The method called when an epoch ends.
        """
        if not os.path.exists(self.records_path):
            os.mkdir(self.records_path)

        full_path = f'{self.records_path}{self.filename}{RECORD_EXTENSION}'

        self.experiment['records_path'] = self.records_path
        self.experiment['filename'] = self.filename
        self.experiment['record'] = self.record
        self.experiment['length'] = len(self.record)
        self.experiment['epochs'] = epoch_number

        joblib.dump(self.experiment, full_path)

    #def on_train_end(self, logs):
    #    print(self.experiment['epochs_index'])