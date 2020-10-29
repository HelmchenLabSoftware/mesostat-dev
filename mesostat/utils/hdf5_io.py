import os
import h5py
import numpy as np
import pandas as pd

from datetime import datetime
from shutil import copyfile

from mesostat.utils.pandas_helper import get_rows_colvals

'''
NOTE:
    * Attribute names "name", "dset", are reserved and should not be used
    * Attribute "datetime" may be overwritten by user

Features
[+] Store time for each dataset
[+] Allow non-unique names: generate fake dataset keys
[+] Backup when saving
[+] Store Pandas Dataframes
[+] Delete datasets by query
[+] Select/Delete datasets by datetime interval
'''

class DataStorage:
    def __init__(self, fname):
        self.fname = fname
        self.fnamebackup = os.path.splitext(fname)[0] + '.bak'
        self.basename = 'dset'
        self.dateTimeFormat = "%d/%m/%Y %H:%M:%S"

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

    def save_dataframe(self, name, df, attrDict):
        fileMode, dsetName = self._get_mode_and_next_dsetname()

        if fileMode == 'a':
            self._backup()

        df.to_hdf(self.basename, dsetName, mode=fileMode, format='table', data_columns=True)

        with h5py.File(self.fname, fileMode) as f5file:
            self._write_attrs(f5file[dsetName], name, attrDict)

    def delete_by_query(self, queryDict):
        self._backup()

        summaryDF = self.list_dsets_pd()
        rows = get_rows_colvals(summaryDF, queryDict)

        with h5py.File(self.fname, "a") as f5file:
            for idx, row in rows.iterrows():
                del f5file[row['dset']]

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

        return dfRez