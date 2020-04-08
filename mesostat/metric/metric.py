import numpy as np

from mesostat.utils.signals import zscore
from mesostat.metric.corr import corr_3D, cross_corr_3D, avg_corr_3D
from mesostat.metric.autocorr import autocorr_3D, autocorr_d1_3D
from mesostat.metric.npeet import average_entropy, average_predictive_info
from mesostat.metric.autoregression import ar1_coeff, ar1_testerr, ar_testerr
from mesostat.metric.stretch import stretch_basis_projection
from mesostat.metric.sequence import cumul_ord_3D, avg_cumul_ord_3D
from mesostat.metric.idtxl import idtxl_single_target, idtxl_network

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
    def __init__(self, serial=False, verbose=True):
        # Initialize parallel mapper once
        self.mapper = GenericMapper(serial, verbose=verbose)

        # Initialize metric library
        self.metricDict = {
            "mean":                 self._nanmean,
            "std":                  self._nanstd,
            "autocorr":             autocorr_3D,
            "corr":                 corr_3D,
            "crosscorr":            cross_corr_3D,
            "BivariateMI":          lambda data, settings : self._TE("BivariateMI", data, settings),
            "BivariateTE":          lambda data, settings: self._TE("BivariateTE", data, settings),
            "MultivariateMI":       lambda data, settings: self._TE("MultivariateMI", data, settings),
            "MultivariateTE":       lambda data, settings: self._TE("MultivariateTE", data, settings),
            "autocorr_d1":          autocorr_d1_3D,
            "ar1_coeff":            ar1_coeff,
            "ar1_testerr":          ar1_testerr,
            "ar_testerr":           ar_testerr,
            "avgcorr":              avg_corr_3D,
            "avg_entropy":          average_entropy,
            "avg_PI":               average_predictive_info,
            "cumul_ord":            cumul_ord_3D,
            "cumul_ord_coeff":      avg_cumul_ord_3D,
            "temporal_basis":       stretch_basis_projection,
            "generic_metric":       self._generic_metric
        }

    def set_data(self, data, dimOrderSrc, timeWindow=None, zscoreDim=None):
        # extract params
        self.dimOrderSrc = dimOrderSrc
        self.timeWindow = timeWindow

        # zscore whole data array if requested
        if zscoreDim is not None:
            axisZScore = tuple([i for i, e in enumerate(dimOrderSrc) if e in zscoreDim])
            self.data = zscore(data, axisZScore)
        else:
            self.data = data

    def _nanmean(self, data, settings):
        return np.nanmean(data)

    def _nanstd(self, data, settings):
        return np.nanstd(data)

    def _TE(self, method, data, settings):
        if settings["parallelTrg"]:
            return idtxl_single_target(settings["iTrg"], method, data, settings)
        else:
            return idtxl_network(method, data, settings)

    # Metric defined by user
    def _generic_metric(self, data, settings):
        return settings["metric"](data, settings)

    def _preprocess(self, metricName, metricSettings, sweepSettings):
        # Wrapper for IDTxl - allow to parallelize over targets
        useIDTxl = ("Bivariate" in metricName) or ("Multivatiate" in metricName)
        if useIDTxl:
            if metricSettings["parallelTrg"]:
                if sweepSettings is None:
                    sweepSettings = {}
                sweepSettings["iTrg"] = np.arange(self.data.shape[self.dimOrderSrc.index("p")]).astype(int)
        return sweepSettings

    def _postprocess(self, result, metricName, metricSettings, sweepSettings):
        # Wrapper for IDTxl - allow to parallelize over targets
        useIDTxl = ("Bivariate" in metricName) or ("Multivatiate" in metricName)
        if useIDTxl:
            if metricSettings["parallelTrg"]:
                # 1. Find the dimOrder of the target sweep
                assert "iTrg" in sweepSettings.keys()
                idxTrgDim = list(sweepSettings.keys()).index("iTrg")
                if self.timeWindow is not None:
                    idxTrgDim += 1   # Window sweep happens before target sweep

                # 2. Transpose s.t. target goes immediately after source
                # Note: Sources and targets are always last, so it suffices to put targets last
                dimSeq = list(range(result.ndim))
                dimSeq = dimSeq[:idxTrgDim] + dimSeq[idxTrgDim+1:] + [dimSeq[idxTrgDim]]
                return result.transpose(tuple(dimSeq))
        return result

    def metric3D(self, metricName, dimOrderTrg, metricSettings=None, sweepSettings=None):
        '''
        :param metricName:      Name of the metric to be computed
        :param metricSettings:  Settings specific to this metric. These settings are fixed for all iterations
        :param sweepSettings:   Settings specific to the metric. All combinations of these settings are iterated over.
                                  Must follow the exact form "key" -> list of values
        :param serial:          Whether to use thread-based parallelization when computing the metric
        :return:
        '''

        # Preprocess sweep settings, in case some metrics need internal parallelization
        sweepSettings = self._preprocess(metricName, metricSettings, sweepSettings)

        # Pass additional fixed settings to the metric function
        if metricSettings is None:
            metricSettings = {}
        metricFunc = self.metricDict[metricName]
        wrappedFunc = lambda data, settings: metricFunc(data, {**settings, **metricSettings})

        sweepGen = SweepGenerator(self.data, self.dimOrderSrc, dimOrderTrg, timeWindow=self.timeWindow, settingsSweep=sweepSettings)
        rez = sweepGen.unpack(self.mapper.mapMultiArg(wrappedFunc, sweepGen.iterator()))

        return self._postprocess(rez, metricName, metricSettings, sweepSettings)
