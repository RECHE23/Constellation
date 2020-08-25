# -*- coding: utf-8 -*-

import os
from natsort import natsorted
from record import *


def get_layouts(records_path: str = RECORDS_PATH) -> list:
    files = list(natsorted(os.listdir(records_path)))
    records = list(filter(
        lambda x: x[-len(LAYOUT_EXTENSION):] == LAYOUT_EXTENSION, files))
    return records
