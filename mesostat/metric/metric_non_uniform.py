import numpy as np

from mesostat.stat.stat import mu_std
from mesostat.metric.corr import corr_3D, avg_corr_3D, cross_corr_non_uniform_3D
from mesostat.metric.autocorr import autocorr_3D, autocorr_d1_3D, autocorr_trunc_1D
from mesostat.metric.npeet import average_entropy, average_predictive_info_non_uniform
from mesostat.metric.autoregression import ar1_coeff, ar1_testerr, ar_testerr
from mesostat.metric.sequence import cumul_ord_3D_non_uniform, avg_cumul_ord_3D_non_uniform
from mesostat.metric.stretch import stretch_basis_projection_non_uniform

from mesostat.utils.arrays import test_uniform_dimension
from mesostat.utils.sweep import SweepGeneratorNonUniform
from mesostat.utils.parallel import GenericMapper

'''
TODO:
    * Add orderability
    * Research other temporal statistics
'''

class MetricCalculatorNonUniform:
    def __init__(self, serial=False, verbose=False):
        # Initialize parallel mapper once
        self.mapper = GenericMapper(serial, verbose=verbose)

        # Initialize metric library
        self.metricDict = {
            "mean":                 lambda dataLst, settings : self._nanmean(dataLst, settings),
            "std":                  lambda dataLst, settings : self._nanstd(dataLst, settings),
            "corr":                 lambda dataLst, settings : self._hstack_wrapper(corr_3D, dataLst, settings),
            "avgcorr":              lambda dataLst, settings : self._hstack_wrapper(avg_corr_3D, dataLst, settings),
            "avg_entropy":          lambda dataLst, settings : self._hstack_wrapper(average_entropy, dataLst, settings),
            "avg_PI":               average_predictive_info_non_uniform,
            "crosscorr":            cross_corr_non_uniform_3D,
            "autocorr":             self._autocorr,
            "autocorr_d1":          self._autocorr_d1,
            # "ar1_coeff":            ar1_coeff,
            # "ar1_testerr":          ar1_testerr,
            # "ar_testerr":           ar_testerr,
            "cumul_ord":            cumul_ord_3D_non_uniform,
            "cumul_ord_coeff":      avg_cumul_ord_3D_non_uniform,
            "temporal_basis":       stretch_basis_projection_non_uniform,
            "generic_metric":       self._generic_metric
        }

    def set_data(self, dataLst, zscoreChannel=False):
        assert np.all([d.ndim == 2 for d in dataLst]), "For each trial must have exactly 2D array of shape [nChannel, nSample]"
        shapeArr = np.array([d.shape for d in dataLst]).T
        # assert np.all(shapeArr > 0), "All dimensions must be non-zero"

        self.nChannel = shapeArr[0][0]

        # zscore whole data array if requested
        if zscoreChannel:
            muCh, stdCh = mu_std(np.hstack(dataLst), axis=1)
            muCh = muCh[:, None]
            stdCh = stdCh[:, None]
            self.dataLst = [(d - muCh) / stdCh for d in dataLst]
        else:
            self.dataLst = dataLst

    def metric3D(self, metricName, dimOrderTrg, metricSettings=None, sweepSettings=None):
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
            metricSettings = dict()

        metricFunc = self.metricDict[metricName]
        wrappedFunc = lambda data, settings: metricFunc(data, {**settings, **metricSettings})

        sweepGen = SweepGeneratorNonUniform(self.dataLst, dimOrderTrg, settingsSweep=sweepSettings)

        return sweepGen.unpack(self.mapper.mapMultiArg(wrappedFunc, sweepGen.iterator()))

    def _hstack_wrapper(self, metricFunc, dataLst, settings):
        test_uniform_dimension(dataLst, settings["dim_order"], "p")
        return metricFunc(np.hstack(dataLst), settings)

    # Metric defined by user
    def _generic_metric(self, data, settings):
        return settings["metric"](data, settings)

    def _nanmean(self, dataLst, settings):
        flatArr = np.hstack([data.flatten() for data in dataLst])
        return np.nanmean(flatArr)

    def _nanstd(self, dataLst, settings):
        flatArr = np.hstack([data.flatten() for data in dataLst])
        return np.nanstd(flatArr)

    def _autocorr(self, dataLst, settings):
        return autocorr_trunc_1D([autocorr_3D(data, settings) for data in dataLst])

    # TODO: If ever important, a better method may be to combine data for the estimate
    def _autocorr_d1(self, dataLst, settings):
        return np.nanmean([autocorr_d1_3D(data, settings) for data in dataLst])

