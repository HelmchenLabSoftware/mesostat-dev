import os, sys, psutil
import numpy as np
from time import gmtime, strftime
from datetime import datetime
from pathlib import Path


# Convert a list of string integers to a date. The integers correspond to ["YYYY", "MM", "DD"] - Others have not been tested
def strlst2date(strlst):
    return datetime(*np.array(strlst, dtype=int))


# Calculate difference in days between two dates in a pandas column
# def date_diff(l):
#     return np.array([(v - l.iloc[0]).days for v in l])
def date_diff(lst, v0):
    return np.array([(v - v0).days for v in lst])


# Get current time as string
def time_now_as_str():
    return strftime("[%Y.%m.%d %H:%M:%S]", gmtime())


# Get current memory use as string
def mem_now_as_str():
    return bytes2str(psutil.virtual_memory().used)


def bytes2str(bytes):
    if bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(np.log(bytes) / np.log(1024)))
    s = round(bytes / (1024 ** i), 2)
    return "%s %s" % (s, size_name[i])


# Print progress bar with percentage
def progress_bar(i, imax, suffix=None):
    sys.stdout.write('\r')
    sys.stdout.write('[{:3d}%] '.format(i * 100 // imax))
    if suffix is not None:
        sys.stdout.write(suffix)
    if i == imax:
        sys.stdout.write("\n")
    sys.stdout.flush()




def make_path(path, parents=True, exist_ok=True):
    Path(path).mkdir(parents=parents, exist_ok=exist_ok)


# Find all folders in this folder (excluding subdirectories)
def get_subfolders(folderpath):
    if not os.path.isdir(folderpath):
        raise NotADirectoryError('Path', folderpath, 'does not exist')

    return [_dir for _dir in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath, _dir))]


# Find all finles in this folder (excluding subdirectories)
def getfiles(inputpath, keys):
    if not os.path.isdir(inputpath):
        raise NotADirectoryError('Path', inputpath, 'does not exist')

    rez = []
    for fname in  os.listdir(inputpath):
        if os.path.isfile(os.path.join(inputpath, fname)):
            if np.all([key in fname for key in keys]):
                rez += [fname]
    return np.array(rez)


# Find all files in a given directory including subdirectories
# All keys must appear in file name
def getfiles_walk(inputpath, keys):
    if not os.path.isdir(inputpath):
        raise NotADirectoryError('Path', inputpath, 'does not exist')

    rez = []
    for dirpath, dirnames, filenames in os.walk(inputpath):
        for filename in filenames:
            if np.all([key in filename for key in keys]):
                rez += [(dirpath, filename)]
    return np.array(rez)
