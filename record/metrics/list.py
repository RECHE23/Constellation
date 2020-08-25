# -*- coding: utf-8 -*-

from .accuracy import accuracy
from .gma import gma
from poutyne.framework.metrics import bin_acc, bce_with_logits, l1, mse, \
    smooth_l1, soft_margin

metrics_list = {'accuracy': accuracy,
                'bin_acc': bin_acc,
                'gma': gma,
                'bce_with_logits': bce_with_logits,
                'l1': l1,
                'mse': mse,
                'smooth_l1': smooth_l1,
                'soft_margin': soft_margin}
