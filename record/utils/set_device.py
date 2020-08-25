# -*- coding: utf-8 -*-

import re
import torch


def set_device(device_name):
    if device_name is None:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Running on {device_name}.\n')
    elif device_name[0:4] == 'cuda' and not torch.cuda.is_available():
        print(f'Device \'{device_name}\' isn\'t available. '
              f'Running on CPU instead...\n')
        device_name = 'cpu'
    elif not re.match(r"^(?:cpu|cuda|cuda:\d+)$", device_name):
        print(f'Device \'{device_name}\' doesn\'t seem to be a valid device. '
              f'Running on CPU instead...\n')
        device_name = 'cpu'
    else:
        print(f'Running on {device_name}.\n')

    return torch.device(device_name)
