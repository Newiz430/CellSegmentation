import re
import numpy as np


def sort_files(file_list, filename_pattern=None):
    if filename_pattern is not None:
        index = [int(re.findall(filename_pattern, name)[0]) for name in file_list]
        file_list = np.asarray(list(file_list))[np.argsort(index)]
    else:
        file_list = sorted(list(file_list))
    return file_list

