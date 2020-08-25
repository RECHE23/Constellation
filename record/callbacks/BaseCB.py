# -*- coding: utf-8 -*-

import os
import re
import warnings
import joblib
from poutyne.framework import Callback
from .. import *

warnings.simplefilter(action='ignore', category=FutureWarning)


class BaseCB(Callback):
    def __init__(self, model_name, dataset_name,
                 records_path=RECORDS_PATH):
        super().__init__()

        filename = f'{model_name}_{dataset_name}'
        filename = re.sub(r"[^\w.]", '_', filename)

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.records_path = records_path
        self.epoch_number = 0
        self.record = []
        self.experiment = {'model_name': model_name,
                           'dataset_name': dataset_name}
        self.filename = filename

    def on_train_begin(self, logs):
        item = {'epoch': self.epoch_number,
                'batch': 0}

        self.record.append(item)

    def on_train_batch_end(self, batch, logs):
        item = {'epoch': self.epoch_number,
                'batch': batch}

        self.record.append(item)

    def on_epoch_begin(self, epoch_number, logs):
        self.epoch_number = epoch_number

    def on_epoch_end(self, epoch_number, logs):
        if not os.path.exists(self.records_path):
            os.mkdir(self.records_path)

        full_path = f'{self.records_path}{self.filename}{RECORD_EXTENSION}'

        self.experiment['records_path'] = self.records_path
        self.experiment['filename'] = self.filename
        self.experiment['record'] = self.record
        self.experiment['length'] = len(self.record)
        self.experiment['epochs'] = epoch_number

        joblib.dump(self.experiment, full_path)