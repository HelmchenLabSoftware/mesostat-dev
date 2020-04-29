import numpy as np

from mesostat.stat.stat import mu_std
from mesostat.metric.corr import corr_3D, avg_corr_3D, cross_corr_non_uniform_3D
from mesostat.metric.autocorr import autocorr_3D, autocorr_d1_3D, autocorr_trunc_1D
from mesostat.metric.npeet import average_entropy, average_predictive_info_non_uniform
from mesostat.metric.autoregression import ar1_coeff, ar1_testerr, ar_testerr
import mesostat.metric.sequence as sequence
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
    def __init__(self, serial=False, verbose=False, nCore=None):
        # Initialize parallel mapper once
        self.mapper = GenericMapper(serial, verbose=verbose, nCore=nCore)

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
            "ord_moments":          sequence.temporal_moments_3D_non_uniform,
            "ord_binary":           sequence.bivariate_binary_orderability_3D_non_uniform,
            "ord_binary_avg":       sequence.avg_bivariate_binary_orderability_3D_non_uniform,
            "ord_student":          sequence.bivariate_student_orderability_3D_non_uniform,
            "ord_student_avg":      sequence.avg_bivariate_student_orderability_3D_non_uniform,
            "temporal_basis":       stretch_basis_projection_non_uniform,
            "generic_metric":       self._generic_metric
        }

        # Initialize composite metrics library
        self.compositeMetricDict = {
            "avg_entropy_1D" :      self._avg_entropy_1D,
            "avg_PI_1D":            self._avg_PI_1D
        }


    def set_data(self, dataLst, zscoreChannel=False):
        assert np.all([d.ndim == 2 for d in dataLst]), "For each trial must have exactly 2D array of shape [nChannel, nSample]"
        shapeArr = np.array([d.shape for d in dataLst]).T
        # assert np.all(shapeArr > 0), "All dimensions must be non-zero"
        self.nChannelMatch = np.all(shapeArr[0] == shapeArr[0][0])

        # zscore whole data array if requested
        if zscoreChannel:
            if not self.nChannelMatch:
                raise ValueError("Trying to ZScore data, where number of channels varies over repetitions")

            muCh, stdCh = mu_std(np.hstack(dataLst), axis=1)
            muCh = muCh[:, None]
            stdCh = stdCh[:, None]

            if np.any(stdCh < 1.0e-10):
                raise ValueError("Attempting to ZScore a channel with zero variance")

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
        if ("p" in dimOrderTrg) and (not self.nChannelMatch):
            raise ValueError("Trying to sweep over channels, but channel number varies across trials")

        if metricName in self.compositeMetricDict.keys():
            return self.compositeMetricDict[metricName](dimOrderTrg, metricSettings, sweepSettings)

        # Pass additional fixed settings to the metric function
        if metricSettings is None:
            metricSettings = dict()

        metricFunc = self.metricDict[metricName]

        def wrappedFunc(dataLst, settings):
            # Drop all repetitions for which the data is degenerate
            dataLstDrop = [data for data in dataLst if np.prod(data.shape) != 0]

            if len(dataLstDrop):
                return metricFunc(dataLstDrop, {**settings, **metricSettings})
            else:
                return None

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

    # Calculate entropy averaged over all channels
    def _avg_entropy_1D(self, dimOrderTrg, metricSettings=None, sweepSettings=None):
        if "p" in dimOrderTrg:
            raise ValueError("Parallelizing over processes not applicable for this metric")

        valH = self.metric3D("avg_entropy", "p" + dimOrderTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)
        return np.nanmean(valH, axis=0)

    # Calculate predictive info averaged over all channels
    def _avg_PI_1D(self, dimOrderTrg, metricSettings=None, sweepSettings=None):
        if "p" in dimOrderTrg:
            raise ValueError("Parallelizing over processes not applicable for this metric")

        valPI = self.metric3D("avg_PI", "p" + dimOrderTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)
        return np.nanmean(valPI, axis=0)

    # Calculate total correlation
    def _avg_TC(self, dimOrderTrg, metricSettings=None, sweepSettings=None):
        H1D = self._avg_entropy_1D(dimOrderTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)
        HND = self.metric3D("avg_entropy", dimOrderTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)
        return H1D - HND

    # Calculate total predictive information
    def _avg_TPI(self, dimOrderTrg, metricSettings=None, sweepSettings=None):
        PI1D = self._avg_PI_1D(dimOrderTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)
        PIND = self.metric3D("avg_PI", dimOrderTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)
        return PIND - PI1D