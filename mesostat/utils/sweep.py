import numpy as np

from mesostat.utils.arrays import perm_map_str, numpy_take_all, unique_subtract, numpy_nonelist_to_array, numpy_transpose_byorder
from mesostat.utils.algorithms import non_uniform_base_arithmetic_iterator


# Sweep data with a running window along the Samples axis
class WindowSweeper:
    def __init__(self, data, dimOrder, timeWindow, nSweepMax=None):
        self.data = data
        self.window = timeWindow
        self.axisSamples = dimOrder.index("s")  # Index of Samples dimension in data
        self.nSamples = data.shape[self.axisSamples]
        nSweepPossible = self.nSamples - self.window + 1
        assert nSweepPossible >= 1, "Window can't be less than number of samples"
        self.nSweep = np.min([nSweepPossible, nSweepMax]) if nSweepMax is not None else nSweepPossible

    def iterator(self):
        for iSweep in range(self.nSweep):
            yield np.split(self.data, [iSweep, iSweep + self.window], axis=self.axisSamples)[1]

    def get_target_time_idxs(self):
        return self.window - 1 + np.arange(self.nSweep).astype(int)


# TODO: Convention for what to do with result dimensions
# TODO: Test unpack for zero dimensions (scalar)
class SweepGenerator:
    def __init__(self, data, dimOrderSrc, dimOrderTrg, timeWindow=None, settingsSweep=None):
        '''
        Multiple parameter exploration toolkit. Includes
        * Iterator: iterate over multidimensional slices of data and optional parameters
        * Unpacker: get resulting list and reshape it


        :param data:            A dataset of at most 3 dimensions
        :param dimOrderSrc:     Dimension names of data in the "psr" convention
        :param dimOrderTrg:     Dimension names to iterate over
        :param timeWindow:      Optional. Iteration over samples will be done via a sweeping window of given length.
                                  Default is single point time iteration
        :param settingsSweep:   Optional. A dictionary "name" -> list. In addition to dimension sweep, this algorithm
                                   can sweep over parameters of the method. Each parameter is given by its name and
                                   a list of its possible values.

        Unpacking Conventions:
            * Output dimensions are always in the order dimOrderTrg, followed by dimensions of single metric output if metric is non-scalar
            * If timeWindow is used, sample dimension must be provided in dimOrderTrg. The sample sweep will appear in the sample dimension
            * If settingsIterator is used, the settings dimensions will go immediately after target dimensions in alphabetic order
        '''
        self.data = data
        self.dimOrderSrc = dimOrderSrc
        self.dimOrderTrg = dimOrderTrg
        self.timeWindow = timeWindow
        self.settingsSweep = settingsSweep


        if self.timeWindow is None:
            self.dimOrderTrgPlainSweep = dimOrderTrg
        else:
            # Samples dimension will be sweeped over by window sweeper, there is no need to sweep over it a second
            # time using a regular sweeper. However, it is sitll important that samples dimension is provided in the
            # dimOrderTrg, because we need to know to which position to transpose it at the unpacking stage
            assert "s" in dimOrderTrg, "If window iteration selected, target dimension must include samples"
            self.dimOrderTrgPlainSweep = unique_subtract(dimOrderTrg, "s")

        # Find number of axis and settings to iterate over
        self.nAxisPlainIter = len(self.dimOrderTrgPlainSweep)

        # Find axis to iterate over
        # self.iterPlainAxis = tuple([i for i, e in enumerate(self.dimOrderSrc) if e in self.dimOrderTrgPlainSweep])
        self.iterPlainAxis = tuple([self.dimOrderSrc.index(e) for e in self.dimOrderTrgPlainSweep])

        # Find shapes of iterated axes
        plainIterAxisShape = tuple([self.data.shape[i] for i in self.iterPlainAxis])

        # Find shapes of iterated settings
        iterSettingsShape = tuple([len(v) for v in self.settingsSweep.values()]) if self.settingsSweep is not None else tuple()

        # Combine data shapes and settings shapes for joint iteration
        self.iterInternalShape = plainIterAxisShape + iterSettingsShape

        if self.timeWindow is None:
            self.iterTotalShape = self.iterInternalShape
        else:
            idxSampleAxis = self.dimOrderSrc.index("s")
            nSample = self.data.shape[idxSampleAxis]
            nSampleWindowed = nSample - self.timeWindow + 1
            self.iterTotalShape = (nSampleWindowed,) + self.iterInternalShape


    def _plain_iterator(self, data=None):
        '''
        Algorithm:
            1. Determine iterable axis sizes and settings sizes, list values
            2. Construct outer product of those values
            3. Iterate over values
              3.1 For each iteration, split values into data indices and settings params
              3.2 Obtain data slice for given indices
              3.3 Construct settings dict for given values
              3.4 Yield

        :return: Data slice for current iteration, as well as parameter value dictionary
        '''
        if data is None:
            data = self.data

        dimOrderSrcRemainder = unique_subtract(self.dimOrderSrc, self.dimOrderTrgPlainSweep)

        if len(self.iterInternalShape) == 0:
            yield data, {}
        else:
            outerIterator = non_uniform_base_arithmetic_iterator(self.iterInternalShape)

            for iND in outerIterator:
                iNDData = iND[:self.nAxisPlainIter]
                iNDSettings = iND[self.nAxisPlainIter:]

                # Take data along iterated axes, and augment fake axes instead, so that data is always the same shape
                dataThis = numpy_take_all(data, self.iterPlainAxis, iNDData)
                dataThis = numpy_transpose_byorder(dataThis, dimOrderSrcRemainder, self.dimOrderSrc, augment=True)

                if self.settingsSweep is None:
                    yield dataThis, {}
                else:
                    extraSettings = {k : v[iSett] for iSett, (k,v) in zip(iNDSettings, self.settingsSweep.items())}
                    yield dataThis, extraSettings


    def iterator(self):
        # If time window is not specified, iterate normally over axis
        if self.timeWindow is None:
            for it in self._plain_iterator():
                yield it
        else:
            # Otherwise, first iterate over window, then over everything else
            winIter = WindowSweeper(self.data, self.dimOrderSrc, self.timeWindow)
            for dataWindow in winIter.iterator():
                for it in self._plain_iterator(dataWindow):
                    yield it


    def unpack(self, rezLst):
        '''
        It is assumed that the shape of each result is exactly the same

        :param rezLst: A list of results of some method, when iterated over the above iterator
        :return: A numpy array, where the list is reshaped into dimensions that have been iterated over
        '''

        if len(rezLst) == 1:
            return rezLst[0]

        rezShape = rezLst[0].shape
        rezArr = np.array(rezLst).reshape(self.iterTotalShape + rezShape)

        # At this point, the resulting array should already be in the correct order, namely
        #  dimOrdTrg + settingsSweep + resultShape
        # The only exception is for the window sweep. In this case, samples go first, and we need to move them where they belong
        # Even if there is a window sweep, if it is the only parameter of the plain iterator, there is nothing to change
        if (len(self.dimOrderTrg) < 2) or (self.timeWindow is None):
            return rezArr
        else:
            dimOrderIntermediate = "s" + self.dimOrderTrgPlainSweep

            # We only want to transpose the target dimensions, leaving the remaining dimensions as they are
            transposedDims = tuple(perm_map_str(dimOrderIntermediate, self.dimOrderTrg))
            trailingDims = tuple(range(len(self.dimOrderTrg), rezArr.ndim))

            return rezArr.transpose(transposedDims + trailingDims)


    def iterator_dimension_names(self):
        dataDimOrder = list(self.dimOrderTrgPlainSweep)
        settingsDimOrder = list(self.settingsSweep) if self.settingsSweep is not None else []
        return dataDimOrder + settingsDimOrder


# TODO: enable iteration over extra settings
class SweepGeneratorNonUniform:
    def __init__(self, dataLst, dimOrderTrg, settingsSweep=None):
        self.dataLst = dataLst
        self.dimOrderTrg = dimOrderTrg
        self.settingsSweep = settingsSweep

        self.shapeDict = {
            "r" : len(dataLst),
            "p" : self.dataLst[0].shape[0]
        }

        if "s" in dimOrderTrg:
            raise ValueError("Sweeping over non-uniform samples is not allowed")

    def iterator(self):
        if "r" in self.dimOrderTrg:
            for data in self.dataLst:
                if "p" in self.dimOrderTrg:
                    for iProcess in range(self.shapeDict["p"]):
                        yield [numpy_transpose_byorder(data[iProcess], "s", "ps", augment=True)], {}
                else:
                    yield [data], {}
        else:
            if "p" in self.dimOrderTrg:
                for iProcess in range(self.shapeDict["p"]):
                    yield [numpy_transpose_byorder(data[iProcess], "s", "ps", augment=True)
                           for data in self.dataLst], {}
            else:
                yield self.dataLst, {}

    # Convert from sweep to expected output shape
    # Input has shape (sweep, result), where sweep is 1D and result may be multiple dimensions
    # Goal is to convert sweep into expected dimensions, while keeping result intact
    def unpack(self, rezLst):
        if len(rezLst) == 1:
            return np.array(rezLst)[0]

        # Check shapes of all results are the same, and fill in None's where data was illegal
        # assert np.all([rez.shape == rezLst[0].shape for rez in rezLst])
        rezArr = numpy_nonelist_to_array(rezLst)

        rezShape = rezLst[0].shape
        trgShape = tuple(self.shapeDict[dim] for dim in "rp" if dim in self.dimOrderTrg)
        rezArr = rezArr.reshape(trgShape + rezShape)

        if len(trgShape) < 2:
            return rezArr
        else:
            # The result may be non-scalar, we need to only transpose the loop dimensions, not the result dimensions
            # Thus, add fake dimensions for result dimensions
            nFakeDim = len(rezShape)
            fakeDim = "".join([str(i) for i in range(nFakeDim)])
            return rezArr.transpose(perm_map_str("rp" + fakeDim, self.dimOrderTrg + fakeDim))
