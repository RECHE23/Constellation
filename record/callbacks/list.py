# -*- coding: utf-8 -*-

from .BaseCB import BaseCB
from .DatasetCB import DatasetCB
from .DecisionBoundariesCB import DecisionBoundariesCB
from .LearningRateCB import LearningRateCB
from .MetricsCB import MetricsCB
from .WeightsBiasesCB import WeightsBiasesCB

callbacks_list = {'BaseCB': BaseCB,
                  'DatasetCB': DatasetCB,
                  'DecisionBoundariesCB': DecisionBoundariesCB,
                  'LearningRateCB': LearningRateCB,
                  'MetricsCB': MetricsCB,
                  'WeightsBiasesCB': WeightsBiasesCB}
