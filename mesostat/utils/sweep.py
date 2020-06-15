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
        '''
        self.data = data
        self.dimOrderSrc = dimOrderSrc
        self.dimOrderTrg = dimOrderTrg
        self.timeWindow = timeWindow
        self.settingsSweep = settingsSweep

        if self.timeWindow is None:
            self.dimOrderTrgFinal = dimOrderTrg
        else:
            assert "s" in dimOrderTrg, "If window iteration selected, target dimension must include samples"

            # Samples dimension will be sweeped over by window sweeper, there is no need to sweep over it a second
            # time using a regular sweeper. However, it is sitll important that samples dimension is provided in the
            # dimOrderTrg, because we need to know to which position to transpose it at the unpacking stage
            self.dimOrderTrg = unique_subtract(dimOrderTrg, "s")
            self.dimOrderTrgFinal = "s" + self.dimOrderTrg

        # Find number of axis and settings to iterate over
        self.nAxis = len(self.dimOrderTrg)

        # Find axis to iterate over
        self.iterDataAxis = tuple([i for i,e in enumerate(self.dimOrderSrc) if e in self.dimOrderTrg])

        # Find shapes of iterated axes
        iterDataShape = tuple([self.data.shape[i] for i in self.iterDataAxis])

        # Find shapes of iterated settings
        iterSettingsShape = tuple([len(v) for v in self.settingsSweep.values()]) if self.settingsSweep is not None else tuple()

        # Combine data shapes and settings shapes for joint iteration
        self.iterTotalShape = iterDataShape + iterSettingsShape

        if self.timeWindow is None:
            self.iterTotalShapeFinal = self.iterTotalShape
        else:
            idxSampleAxis = self.dimOrderSrc.index("s")
            nSample = self.data.shape[idxSampleAxis]
            nSampleWindowed = nSample - self.timeWindow + 1
            self.iterTotalShapeFinal = (nSampleWindowed, ) + self.iterTotalShape


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

        dimOrderTrue = unique_subtract(self.dimOrderSrc, self.dimOrderTrg)

        if len(self.iterTotalShape) == 0:
            yield data, {}
        else:
            outerIterator = non_uniform_base_arithmetic_iterator(self.iterTotalShape)

            for iND in outerIterator:
                iNDData = iND[:self.nAxis]
                iNDSettings = iND[self.nAxis:]

                # Take data along iterated axes, and augment fake axes instead, so that data is always the same shape
                dataThis = numpy_take_all(data, self.iterDataAxis, iNDData)
                dataThis = numpy_transpose_byorder(dataThis, dimOrderTrue, self.dimOrderSrc, augment=True)

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
        rezArr = np.array(rezLst).reshape(self.iterTotalShapeFinal + rezShape)

        # After sweep is done, the order of the sweep axis may not match the requested order in the dimOrderTrg
        # We may need to transpose to account for that
        if len(self.dimOrderTrgFinal) < 2:
            return rezArr  # If the iterated dimension is less than 2, there is only 1 possible outcome anyway
        else:
            postDimOrder = "".join([e for e in self.dimOrderSrc if e in self.dimOrderTrgFinal])

            # The result may be non-scalar, we need to only transpose the loop dimensions, not the result dimensions
            # Thus, add fake dimensions for result dimensions
            nFakeDim = rezArr.ndim - self.nAxis
            fakeDim = "".join([str(i) for i in range(nFakeDim)])

            return rezArr.transpose(perm_map_str(postDimOrder + fakeDim, self.dimOrderTrgFinal + fakeDim))


    def iterator_dimension_names(self):
        dataDimOrder = list(self.dimOrderTrg)
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
            return np.array(rezLst)

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
