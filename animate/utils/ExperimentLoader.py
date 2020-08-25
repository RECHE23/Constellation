# -*- coding: utf-8 -*-

import joblib
from natsort import natsorted
from .get_records import get_records
from record import *


class ExperimentLoader(object):
    def __init__(self, records_path: str = RECORDS_PATH):
        records = get_records(records_path)
        self.experiments_list = [joblib.load(f'{RECORDS_PATH}{record}')
                                 for record in records]

    def get(self, model: str = None, dataset: str = None,
            representation: str = None):
        experiments = []
        for experiment in self.experiments_list:
            model_name = experiment['model_name']
            dataset_name = experiment['dataset_name']
            representations = experiment['representations'].keys()

            add_to_list = True
            if model is not None and model_name != model:
                add_to_list &= False
            if dataset is not None and dataset_name != dataset:
                add_to_list &= False
            if representation is not None and representation not in representations:
                add_to_list &= False
            if add_to_list and experiment not in experiments:
                experiments.append(experiment)
        if model is not None and dataset is not None \
                and representation is not None and len(experiments) == 1:
            return experiments[0]
        return experiments

    def filter(self, models: list = None, datasets: list = None,
               representations: list = None):
        experiments = []
        for experiment in self.experiments_list:
            model_name = experiment['model_name']
            dataset_name = experiment['dataset_name']
            representations_list = experiment['representations'].keys()

            add_to_list = True
            if models is not None and model_name not in models:
                add_to_list &= False
            if datasets is not None and dataset_name not in datasets:
                add_to_list &= False
            if representations is not None and \
                    set(representations).difference(set(representations_list)):
                add_to_list &= False
            if add_to_list and experiment not in experiments:
                experiments.append(experiment)
        return experiments

    def list(self, what: str):
        assert what in ['Models', 'Datasets', 'Representations']
        ret = set()
        if what == 'Models':
            for experiment in self.experiments_list:
                ret.add(experiment['model_name'])
        elif what == 'Datasets':
            for experiment in self.experiments_list:
                ret.add(experiment['dataset_name'])
        elif what == 'Representations':
            for experiment in self.experiments_list:
                for representation in experiment['representations'].keys():
                    ret.add(representation)
        return list(natsorted(ret))

    @staticmethod
    def list_from(source, what: str):
        assert what in ['Models', 'Datasets', 'Representations']
        ret = set()
        if what == 'Models':
            for experiment in source:
                ret.add(experiment['model_name'])
        elif what == 'Datasets':
            for experiment in source:
                ret.add(experiment['dataset_name'])
        elif what == 'Representations':
            for experiment in source:
                for representation in experiment['representations'].keys():
                    ret.add(representation)
        return list(natsorted(ret))
