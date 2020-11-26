import numpy as np

from mesostat.utils.arrays import numpy_transpose_byorder
from mesostat.utils.signals import zscore_dim_ord
from mesostat.metric.corr import corr_3D, cross_corr_3D, avg_corr_3D
from mesostat.metric.autocorr import autocorr_3D, autocorr_d1_3D
import mesostat.metric.npeet as npeet
import mesostat.metric.autoregression as autoregression
from mesostat.metric.stretch import stretch_basis_projection
import mesostat.metric.sequence as sequence
import mesostat.metric.pca as pca
from mesostat.metric.idtxl_te import idtxl_single_target, idtxl_network
from mesostat.metric.idtxl_pid import bivariate_pid_3D

from mesostat.utils.sweep import SweepGenerator
from mesostat.utils.parallel import GenericMapper

'''
TODO:
    * Optimization: Implement fft autocorrelation, its faster and more or less same accuracy
    * Have some adequate solution for extra axis for methods that return non-scalars. Especially dynamics like PI, TE
    * Copy over connectomics metrics
    * H5 I/O For data results
    * Research other temporal statistics
    
    * Optimization: Implement optional hybrid parallelization for channels.
      In some cases, parallelizing over channels inside metric is more efficient than a parallel python for loop.
      In particular, if numpy array functions can be used to calc channels, it is faster than python looping
      
      Metrics relevant: sum, mean, std, autocorr_d1, temporal_basis
      Algorithm:
        * In metric3D, check if ("p" in dimOrdTrg) and (supportsHybrid(metricName))
          * If yes, exclude "p" from dimOrdTrgEff, and provide hybrid flag
          * If no, complain that hybrid flag requested but not available
        * In metric, if have hybrid flag, use alternative implementation
        * In sweep unpack, transpose result to bring processes back to expected position 
'''

class MetricCalculator:
    def __init__(self, serial=False, verbose=True, nCore=None):
        # Initialize parallel mapper once
        self.mapper = GenericMapper(serial, verbose=verbose, nCore=nCore)

        # Initialize metric library
        self.metricDict = {
            # SCALAR METRICS
            "sum":                  self._nansum,
            "mean":                 self._nanmean,
            "std":                  self._nanstd,
            "autocorr_d1":          autocorr_d1_3D,
            "ar1_coeff":            autoregression.ar1_coeff,
            "ar1_testerr":          autoregression.ar1_testerr,
            "ar_testerr":           autoregression.ar_testerr,
            "mar1_coeff":           autoregression.mar1_coeff,
            "mar1_testerr":         autoregression.mar1_testerr,
            "mar_testerr":          autoregression.mar_testerr,
            "mar_inp_testerr":      autoregression.mar_inp_testerr,
            "avgcorr":              avg_corr_3D,
            "avg_entropy":          npeet.average_entropy_3D,
            "avg_PI":               npeet.average_predictive_info,
            "rank_smooth":          pca.rank_smooth3D,
            "rank_effective":       pca.erank3D,

            # FUNCTIONAL CONNECTIVITY METRICS
            "corr":                 corr_3D,
            "crosscorr":            cross_corr_3D,
            "cross_MI":             npeet.cross_mi_3D,
            "BivariateMI":          lambda data, settings : self._TE("BivariateMI", data, settings),
            "BivariateTE":          lambda data, settings: self._TE("BivariateTE", data, settings),
            "MultivariateMI":       lambda data, settings: self._TE("MultivariateMI", data, settings),
            "MultivariateTE":       lambda data, settings: self._TE("MultivariateTE", data, settings),
            "BivariatePID":         bivariate_pid_3D,

            # TEMPORAL METRICS
            "autocorr":             autocorr_3D,
            "ord_mean":             sequence.temporal_mean_3D,
            "ord_moments":          sequence.temporal_moments_3D,
            "ord_binary":           sequence.bivariate_binary_orderability_3D,
            "ord_binary_avg":       sequence.avg_bivariate_binary_orderability_3D,
            "ord_student":          sequence.bivariate_student_orderability_3D,
            "ord_student_avg":      sequence.avg_bivariate_student_orderability_3D,
            "temporal_basis":       stretch_basis_projection,

            "generic_metric":       self._generic_metric
        }

        # Initialize composite metrics library
        self.compositeMetricDict = {
            "avg_entropy_1D" :      self._avg_entropy_1D,
            "avg_PI_1D":            self._avg_PI_1D,
            "avg_TC":               self._avg_TC,
            "avg_TPI":              self._avg_TPI,
        }

        # Initialize canonical order
        self.dimOrderCanon = 'rps'

    def set_data(self, data, dimOrderSrc, timeWindow=None, zscoreDim=None):
        # Convert data to standard format, add fake missing dimensions
        dataCanon = numpy_transpose_byorder(data, dimOrderSrc, self.dimOrderCanon, augment=True)

        # extract params
        self.timeWindow = timeWindow

        # zscore whole data array if requested
        self.data = zscore_dim_ord(dataCanon, self.dimOrderCanon, zscoreDim)
        # if zscoreDim is not None:
        #     axisZScore = tuple([i for i, e in enumerate(self.dimOrderCanon) if e in zscoreDim])
        #     self.data = zscore(dataCanon, axisZScore)
        # else:
        #     self.data = dataCanon

    def _nansum(self, data, settings):
        return np.nansum(data)

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

    # Calculate channel-wise entropy averaged over all channels
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

    def _is_metric_idtxl_mi_te(self, metricName):
        return metricName in ["BivariateMI", "BivariateTE", "MultivariateMI", "MultivariateTE"]

    def _sweep_param_idx(self, sweepSettings, param):
        assert param in sweepSettings.keys()
        idxParamDim = list(sweepSettings.keys()).index(param)
        if self.timeWindow is not None:
            idxParamDim += 1  # Window sweep happens before target sweep
        return idxParamDim

    # Transposes array so that dimension iDim is at the last place
    def _push_back_dim(self, data, iDim):
        dimSeq = list(range(data.ndim))
        dimSeq = dimSeq[:iDim] + dimSeq[iDim + 1:] + [dimSeq[iDim]]
        return data.transpose(tuple(dimSeq))

    def _preprocess(self, metricName, metricSettings, sweepSettings):
        nProcess = self.data.shape[self.dimOrderCanon.index("p")]

        # Wrapper for IDTxl - allow to parallelize over targets
        if "parallelTrg" in metricSettings and metricSettings["parallelTrg"]:
            if self._is_metric_idtxl_mi_te(metricName):
                if sweepSettings is None:
                    sweepSettings = {}
                sweepSettings["iTrg"] = list(range(nProcess))
            else:
                raise ValueError("Attempting to parallelize over target for unexpected metric", metricName)
        elif "parallelSrc3D" in metricSettings and metricSettings["parallelSrc3D"]:
            if metricName == "BivariatePID":
                trg = metricSettings['trg']
                src = []
                for i in range(nProcess):
                    if i != trg:
                        for j in range(i+1, nProcess):
                            if j != trg:
                                src += [(i, j)]
                sweepSettings['src'] = src
            else:
                raise ValueError("Attempting to parallelize over sources for unexpected metric", metricName)

        return sweepSettings

    def _postprocess(self, result, metricName, metricSettings, sweepSettings):
        # Wrapper for IDTxl - allow to parallelize over targets

        if "parallelTrg" in metricSettings and metricSettings["parallelTrg"]:
            if self._is_metric_idtxl_mi_te(metricName):
                # 1. Find the dimOrder of the target sweep
                idxTrgDim = self._sweep_param_idx(sweepSettings, 'iTrg')

                # 2. Transpose s.t. target goes immediately after source
                # Note: Sources and targets are always last, so it suffices to put targets last
                return self._push_back_dim(result, idxTrgDim)
        elif "parallelSrc3D" in metricSettings and metricSettings["parallelSrc3D"]:
            if metricName == "BivariatePID":
                # 1. Find the dimOrder of the source sweep
                idxSrcDim = self._sweep_param_idx(sweepSettings, 'iTrg')

                # Construct new shape where there is no source loop, but there is a 2D [nCh x nCh] array at the end
                nProcess = self.data.shape[self.dimOrderCanon.index("p")]
                newShape = result.shape[:idxSrcDim] + result.shape[idxSrcDim+1:] + (nProcess, nProcess)
                resultNew = np.array(newShape)

                # Fill it in
                for iSrcPair, (iSrc1, iSrc2) in enumerate(sweepSettings['src']):
                    resultNew[..., iSrc1, iSrc2] = np.take(result, iSrcPair, axis=idxSrcDim)

                # Push the PID components into last axis
                return self._push_back_dim(resultNew, -3)

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

        if metricName in self.compositeMetricDict.keys():
            return self.compositeMetricDict[metricName](dimOrderTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)

        # Pass additional fixed settings to the metric function
        if metricSettings is None:
            metricSettings = {}
        metricFunc = self.metricDict[metricName]
        wrappedFunc = lambda data, settings: metricFunc(data, {**settings, **metricSettings})

        sweepGen = SweepGenerator(self.data, self.dimOrderCanon, dimOrderTrg, timeWindow=self.timeWindow, settingsSweep=sweepSettings)
        rez = sweepGen.unpack(self.mapper.mapMultiArg(wrappedFunc, sweepGen.iterator()))

        return self._postprocess(rez, metricName, metricSettings, sweepSettings)
