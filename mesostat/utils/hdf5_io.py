import os
import h5py
import numpy as np
import pandas as pd

from datetime import datetime
from shutil import copyfile
from ast import literal_eval as str2tuple

from mesostat.utils.pandas_helper import pd_query

'''
NOTE:
    * Attribute names "name", "dset", are reserved and should not be used
    * Attribute "datetime" may be overwritten by user
    * Considered also storing pandas dataframes, but it is much easier to use pandas-hdf5 interface directly

Features
[+] Store time for each dataset
[+] Allow non-unique names: generate fake dataset keys
[+] Backup when saving
[+] Delete datasets by query
[+] Select/Delete datasets by datetime interval
'''

class DataStorage:
    def __init__(self, fname):
        self.fname = fname
        self.fnamebackup = os.path.splitext(fname)[0] + '.bak'
        self.basename = 'dset'
        self.dateTimeFormat = "%d/%m/%Y %H:%M:%S"
        self.pandasTimeFormat = "%Y-%m-%d %H:%M:%S"

    def _incr_name(self, nameLst, basename):
        nBase = len(basename)
        maxIdx = np.max([int(name[nBase:]) for name in nameLst])
        return basename + str(maxIdx + 1)

    def _backup(self):
        copyfile(self.fname, self.fnamebackup)  # Backup

    def _get_mode_and_next_dsetname(self):
        if os.path.isfile(self.fname):
            fileMode = "a"
            dsetName = self._incr_name(self.list_dset_names(), self.basename)
        else:
            fileMode = "w"
            dsetName = self.basename + '0'

        return fileMode, dsetName

    def _write_attrs(self, dset, name, attrDict):
        attrDictThis = attrDict.copy()
        attrDictThis['name'] = name

        # Set Datetime as now if not provided by user
        if datetime not in attrDictThis.keys():
            attrDictThis['datetime'] = datetime.now().strftime(self.dateTimeFormat)
        else:
            attrDictThis['datetime'] = attrDictThis['datetime'].strftime(self.dateTimeFormat)

        # Write attributes to file
        for k, v in attrDictThis.items():
            dset.attrs[k] = v

    def get_data(self, dsetName):
        with h5py.File(self.fname, "r") as f5file:
            return np.array(f5file[dsetName])

    def get_data_recent_by_query(self, queryDict, listDF=None):
        if listDF is None:
            listDF = self.list_dsets_pd()

        rows = pd_query(listDF, queryDict)

        # Find index of the latest result
        maxRowIdx = rows['datetime'].idxmax()
        attrs = rows.loc[maxRowIdx]
        data = self.get_data(attrs['dset'])

        return data, attrs

    def del_data(self, dsetName):
        self._backup()
        with h5py.File(self.fname, "a") as f5file:
            del f5file[dsetName]

    def save_data(self, name, data, attrDict):
        fileMode, dsetName = self._get_mode_and_next_dsetname()

        if fileMode == 'a':
            self._backup()

        with h5py.File(self.fname, fileMode) as f5file:
            dset = f5file.create_dataset(dsetName, data=data)
            self._write_attrs(dset, name, attrDict)

    def delete_rows(self, rows):
        self._backup()
        print('Deleting', len(rows), 'rows')
        with h5py.File(self.fname, "a") as f5file:
            for idx, row in rows.iterrows():
                del f5file[row['dset']]

    def delete_by_query(self, queryDict):
        summaryDF = self.list_dsets_pd()
        rows = pd_query(summaryDF, queryDict)
        self.delete_rows(rows)

    def delete_recent(self, timestr):
        timeObj = datetime.strptime(timestr, self.pandasTimeFormat)
        summaryDF = self.list_dsets_pd()
        rows = summaryDF[summaryDF['datetime'] >= timeObj]
        self.delete_rows(rows)

    def list_dset_names(self):
        with h5py.File(self.fname, "r") as f5file:
            return list(f5file.keys())

    def list_dsets_pd(self):
        with h5py.File(self.fname, "r") as f5file:
            dfRez = pd.DataFrame()

            for dname in f5file.keys():
                dset = f5file[dname]
                attrs = dset.attrs

                dictThis = {
                        **{"dset": dname, "shape": dset.shape},
                        **{attrName: attrs[attrName] for attrName in attrs.keys()}
                }

                dfRez = dfRez.append(dictThis, ignore_index=True)

        # Convert DateTime back to object
        datetimes = [datetime.strptime(d, self.dateTimeFormat) for d in list(dfRez["datetime"])]
        dfRez = dfRez.assign(datetime=datetimes)

        # Convert shape back to tuple
        if 'target_dim' in dfRez.keys():
            shapeLabels = [str2tuple(d) for d in list(dfRez["target_dim"])]
            dfRez = dfRez.assign(target_dim=shapeLabels)

        return dfRez