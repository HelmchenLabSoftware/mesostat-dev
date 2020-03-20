import numpy as np

from mesostat.utils.signals import zscore
from mesostat.metric.corr import corr_3D, cross_corr_3D, avg_corr_3D
from mesostat.metric.autocorr import autocorr_3D, autocorr_d1_3D
from mesostat.metric.npeet import entropy, predictive_info
from mesostat.metric.mar import ar1_coeff, ar1_testerr, ar_testerr
from mesostat.metric.stretch import stretch_basis_projection
from mesostat.metric.sequence import cumul_ord_3D, avg_cumul_ord_3D

from mesostat.utils.sweep import SweepGenerator
from mesostat.utils.parallel import GenericMapper

'''
TODO:
    * Have some adequate solution for extra axis for methods that return non-scalars. Especially dynamics like PI, TE
    * Copy over connectomics metrics
    * H5 I/O For data results
    * Add MI, TE from IDTxl
    * Add orderability
    * Research other temporal statistics
'''

class MetricCalculator:
    def __init__(self, data, dimOrderSrc, timeWindow=None, zscoreDim=None):
        # extract params
        self.dimOrderSrc = dimOrderSrc
        self.timeWindow = timeWindow

        # zscore whole data array if requested
        if zscoreDim is not None:
            axisZScore = tuple([i for i, e in enumerate(dimOrderSrc) if e in zscoreDim])
            self.data = zscore(data, axisZScore)
        else:
            self.data = data

        # Initialize metric library
        self.metricDict = {
            "mean":                 self._nanmean,
            "std":                  self._nanstd,
            "autocorr":             autocorr_3D,
            "corr":                 corr_3D,
            "crosscorr":            cross_corr_3D,
            "autocorr_d1":          autocorr_d1_3D,
            "ar1_coeff":            ar1_coeff,
            "ar1_testerr":          ar1_testerr,
            "ar_testerr":           ar_testerr,
            "avgcorr":              avg_corr_3D,
            "entropy":              entropy,
            "PI":                   predictive_info,
            "cumul_ord":            cumul_ord_3D,
            "cumul_ord_coeff":      avg_cumul_ord_3D,
            "temporal_basis":       stretch_basis_projection
        }

    def _nanmean(self, data, settings):
        return np.nanmean(data)

    def _nanstd(self, data, settings):
        return np.nanstd(data)

    def metric3D(self, metricName, dimOrderTrg, metricSettings=None, sweepSettings=None, serial=False):
        '''
        :param metricName:      Name of the metric to be computed
        :param metricSettings:  Settings specific to this metric. These settings are fixed for all iterations
        :param sweepSettings:   Settings specific to the metric. All combinations of these settings are iterated over.
                                  Must follow the exact form "key" -> list of values
        :param serial:          Whether to use thread-based parallelization when computing the metric
        :return:
        '''

        # Pass additional fixed settings to the metric function
        if metricSettings is None:
            metricSettings = {}
        metricFunc = self.metricDict[metricName]
        wrappedFunc = lambda data, settings: metricFunc(data, {**settings, **metricSettings})

        mapper = GenericMapper(serial)
        sweepGen = SweepGenerator(self.data, self.dimOrderSrc, dimOrderTrg, timeWindow=self.timeWindow, settingsSweep=sweepSettings)

        return sweepGen.unpack(mapper.mapMultiArg(wrappedFunc, sweepGen.iterator()))
