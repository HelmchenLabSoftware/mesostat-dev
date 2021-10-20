import numpy as np

from mesostat.stat.stat import mu_std
import mesostat.metric.dim2d.corr as corr
from mesostat.metric.dim1d.autocorr import autocorr_d1_3D_non_uniform
import mesostat.metric.infotheory.npeet as npeet_wrapper
import mesostat.metric.scalar.autoregression as autoregression
import mesostat.metric.temporal.sequence as sequence
import mesostat.metric.temporal.stretch as stretch
import mesostat.metric.scalar.pca as pca

from mesostat.utils.arrays import set_list_shapes
from mesostat.utils.iterators.sweep import SweepGeneratorNonUniform
from mesostat.utils.parallel import GenericMapper

'''
TODO:
    * Research other temporal statistics
'''

class MetricCalculatorNonUniform:
    def __init__(self, serial=False, verbose=False, nCore=None):
        # Initialize parallel mapper once
        self.mapper = GenericMapper(serial, verbose=verbose, nCore=nCore)

        # Initialize metric library
        self.metricDict = {
            # SCALAR METRICS
            "sum":                  lambda dataLst, settings : self._flat_reduce(np.nansum, dataLst),
            "mean":                 lambda dataLst, settings : self._flat_reduce(np.nanmean, dataLst),
            "std":                  lambda dataLst, settings : self._flat_reduce(np.nanstd, dataLst),
            "autocorr_d1":          autocorr_d1_3D_non_uniform,
            "ar1_coeff":            autoregression.ar1_coeff_non_uniform,
            "ar1_testerr":          autoregression.ar1_testerr_non_uniform,
            "ar_testerr":           autoregression.ar_testerr_non_uniform,
            "mar1_coeff":           autoregression.mar1_coeff_non_uniform,
            "mar1_testerr":         autoregression.mar1_testerr_non_uniform,
            "mar_testerr":          autoregression.mar_testerr_non_uniform,
            "mar_inp_testerr":      autoregression.mar_inp_testerr_non_uniform,
            "avgcorr":              corr.avg_corr_3D_non_uniform,
            "avg_entropy":          npeet_wrapper.average_entropy_3D,
            "avg_PI":               npeet_wrapper.average_predictive_info_non_uniform,
            "rank_smooth":          pca.rank_smooth3D,

            # FUNCTIONAL CONNECTIVITY
            "corr":                 corr.corr_3D_non_uniform,
            "crosscorr":            corr.cross_corr_non_uniform_3D,
            # "autocorr":             autocorr_trunc_1D,

            # TEMPORAL METRICS
            "ord_mean":             sequence.temporal_mean_3D_non_uniform,
            "ord_moments":          sequence.temporal_moments_3D_non_uniform,
            "ord_binary":           sequence.bivariate_binary_orderability_3D_non_uniform,
            "ord_binary_avg":       sequence.avg_bivariate_binary_orderability_3D_non_uniform,
            "ord_student":          sequence.bivariate_student_orderability_3D_non_uniform,
            "ord_student_avg":      sequence.avg_bivariate_student_orderability_3D_non_uniform,
            "temporal_basis":       stretch.stretch_basis_projection_non_uniform,
            "resample_fixed":       stretch.resample_non_uniform,

            "generic_metric":       self._generic_metric
        }

        # Initialize composite metrics library
        self.compositeMetricDict = {
            "avg_entropy_1D" :      self._avg_entropy_1D,
            "avg_PI_1D":            self._avg_PI_1D,
            "avg_TC_1D":            self._avg_TC,
            "avg_TPI_1D":           self._avg_TPI,
        }

    def set_data(self, dataLst, zscoreChannel=False):
        assert len(dataLst) > 0, "Attempted to set data with zero trials"
        assert np.all([d.ndim == 2 for d in dataLst]), "For each trial must have exactly 2D array of shape [nChannel, nSample]"
        self.nChannelMatch = len(set_list_shapes(dataLst, axis=0)) == 1

        # zscore whole data array if requested
        if zscoreChannel:
            if not self.nChannelMatch:
                raise ValueError("Trying to ZScore data, where number of channels varies over repetitions")

            data2D = np.hstack(dataLst)
            muCh, stdCh = mu_std(data2D, axis=1)
            muCh = muCh[:, None]
            stdCh = stdCh[:, None]

            if np.any(stdCh < 1.0e-10):
                print("nTrials", len(dataLst), "concatShape", data2D.shape, stdCh)
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
            return self.compositeMetricDict[metricName](dimOrderTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)

        # Pass additional fixed settings to the metric function
        if metricSettings is None:
            metricSettings = dict()

        metricFunc = self.metricDict[metricName]

        def wrappedFunc(dataLst, settings):
            # Drop all repetitions for which the data is degenerate
            dataLstDrop = [data for data in dataLst if np.prod(data.shape) != 0]

            if len(dataLstDrop) > 0:
                return metricFunc(dataLstDrop, {**settings, **metricSettings})
            else:
                return None

        sweepGen = SweepGeneratorNonUniform(self.dataLst, dimOrderTrg, settingsSweep=sweepSettings)

        return sweepGen.unpack(self.mapper.mapMultiArg(wrappedFunc, sweepGen.iterator()))

    # Metric defined by user
    def _generic_metric(self, data, settings):
        return settings["metric"](data, settings)

    def _flat_reduce(self, func, dataLst):
        return func(np.hstack([data.flatten() for data in dataLst]))

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