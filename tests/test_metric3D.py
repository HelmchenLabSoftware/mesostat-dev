import numpy as np
import unittest

from mesostat.metric.metric import MetricCalculator


class TestMetrics3D(unittest.TestCase):

    # Computing rel. error of approximate number of combinations for small values
    def test_shapes(self):
        # Create dummy data
        nProcess = 12
        nSample = 200
        nRepetition = 100

        dataShape = (nProcess, nSample, nRepetition)
        dataDimOrd = "psr"
        data = np.random.uniform(0, 1, dataShape)

        # Settings for some of the metrics
        metricSettings = {"max_lag" : 3, "lag" : 5, "hist" : 3, "parallelTrg" : True,
                          'min_lag_sources': 1,
                          'max_lag_sources': 1,
                          'cmi_estimator': "JidtGaussianCMI"
                          }

        mc = MetricCalculator(serial=False)
        mc.set_data(data, dataDimOrd, timeWindow=None, zscoreDim="sr")

        # Test if result of the metric has expected shape, based on source and target axis
        def testAxis(metrics, srcDimOrder, trgDimOrder, extraAxis=None, metricSettings=None):
            trgShape = tuple([dataShape[srcDimOrder.index(e)] for e in trgDimOrder])

            for iMetric, metricName in enumerate(metrics):
                print(metricName, trgDimOrder)
                trgShapeEff = trgShape if extraAxis is None else trgShape + extraAxis[iMetric]
                rez = mc.metric3D(metricName, trgDimOrder, metricSettings=metricSettings)
                self.assertEqual(rez.shape, trgShapeEff, msg="Error in "+metricName)

        # Test if metrics throw ValueError exception
        def testRaises(metrics, trgDimOrder):
            for iMetric, metricName in enumerate(metrics):
                print(metricName, trgDimOrder)
                with self.assertRaises(ValueError):
                    mc.metric3D(metricName, trgDimOrder, metricSettings=metricSettings)

        ########################################
        # TEST 1.1: Shape: Full projections
        ########################################

            # "mean":                 self._nanmean,
            # "std":                  self._nanstd,
            # "autocorr":             autocorr_3D,
            # "corr":                 corr_3D,
            # "crosscorr":            cross_corr_3D,
            # "BivariateMI":          lambda data, settings : self._TE("BivariateMI", data, settings),
            # "BivariateTE":          lambda data, settings: self._TE("BivariateTE", data, settings),
            # "MultivariateMI":       lambda data, settings: self._TE("MultivariateMI", data, settings),
            # "MultivariateTE":       lambda data, settings: self._TE("MultivariateTE", data, settings),
            # "autocorr_d1":          autocorr_d1_3D,
            # "ar1_coeff":            autoregression.ar1_coeff,
            # "ar1_testerr":          autoregression.ar1_testerr,
            # "ar_testerr":           autoregression.ar_testerr,
            # "mar1_coeff":           autoregression.mar1_coeff,
            # "mar1_testerr":         autoregression.mar1_testerr,
            # "mar_testerr":          autoregression.mar_testerr,
            # "mar_inp_testerr":      autoregression.mar_inp_testerr,
            # "avgcorr":              avg_corr_3D,
            # "avg_entropy":          average_entropy_3D,
            # "avg_PI":               average_predictive_info,
            # "ord_mean":             sequence.temporal_mean_3D,
            # "ord_moments":          sequence.temporal_moments_3D,
            # "ord_binary":           sequence.bivariate_binary_orderability_3D,
            # "ord_binary_avg":       sequence.avg_bivariate_binary_orderability_3D,
            # "ord_student":          sequence.bivariate_student_orderability_3D,
            # "ord_student_avg":      sequence.avg_bivariate_student_orderability_3D,
            # "temporal_basis":       stretch_basis_projection,
            # "generic_metric":       self._generic_metric
            # "avg_entropy_1D" :      self._avg_entropy_1D,
            # "avg_PI_1D":            self._avg_PI_1D,
            # "avg_TC_1D":            self._avg_TC,
            # "avg_TPI_1D":           self._avg_TPI,

        # # Scalar
        # metricScalarFull = [
        #     "mean", "std",
        #     "autocorr_d1",
        #     "ar1_coeff", "ar1_testerr", "ar_testerr", "mar1_testerr", "mar_testerr",
        #     "avgcorr",
        #     "avg_entropy", "avg_PI", "avg_entropy_1D", "avg_PI_1D", "avg_TC_1D", "avg_TPI_1D",
        #     "ord_mean", "ord_binary_avg",
        #     ]
        # for metricName in metricScalarFull:
        #     print("Testing full projection for", metricName)
        #     rez = mc.metric3D(metricName, "", metricSettings=metricSettings)
        #     self.assertTrue(np.isscalar(rez))
        #     self.assertTrue(~np.isnan(rez))

        # Vector
        metricVectorFull = [
            "autocorr",
            "corr", "crosscorr",
            "BivariateMI", "BivariateTE", "MultivariateMI", "MultivariateTE",
            "ord_binary",
            "temporal_basis"]
        extraAxisVectorFull = [
            (nSample,),
            (nProcess, nProcess), (nProcess, nProcess),
            (3, nProcess, nProcess), (3, nProcess, nProcess), (3, nProcess, nProcess), (3, nProcess, nProcess),
            (nProcess, nProcess),
            (5, )]
        testAxis(metricVectorFull, dataDimOrd, "", extraAxis=extraAxisVectorFull, metricSettings=metricSettings)
        #
        # #######################################
        # # TEST 1.2: Shape: scalar: 1D projections
        # #######################################
        # # Scalar
        # metricScalarSample = ["mean", "std", "avgcorr", "avg_entropy"]
        # testAxis(metricScalarSample, dataDimOrd, "s")
        #
        # metricScalarProcess = ["mean", "std", "autocorr_d1", "ar1_coeff", "ar1_testerr", "avg_entropy", "avg_PI"]
        # testAxis(metricScalarProcess, dataDimOrd, "p", metricSettings=metricSettings)
        #
        # # Vector
        # metricVectorSample = ["corr"]
        # extraAxisVectorSample = [(nProcess, nProcess)]
        # testAxis(metricVectorSample, dataDimOrd, "s", extraAxis=extraAxisVectorSample, metricSettings=metricSettings)
        #
        # metricVectorProcess = ["autocorr", "temporal_basis"]
        # extraAxisVectorProcess = [(nSample,), (5,)]
        # testAxis(metricVectorProcess, dataDimOrd, "p", extraAxis=extraAxisVectorProcess, metricSettings=metricSettings)
        #
        # ########################################
        # # TEST 1.1.3: Shape: scalar: 2D projections
        # ########################################
        # # Scalar
        # metricScalarProcessSample = ["mean", "std", "avg_entropy"]
        # testAxis(metricScalarProcessSample, dataDimOrd, "ps")
        #
        # metricScalarProcessRepetition = ["mean", "std", "autocorr_d1", "ar1_coeff", "ar1_testerr", "avg_entropy", "avg_PI"]
        # testAxis(metricScalarProcessRepetition, dataDimOrd, "pr", metricSettings=metricSettings)
        #
        # # Vector
        # metricVectorProcessRepetition = ["autocorr", "temporal_basis"]
        # extraAxisVectorProcessRepetition = [(nSample,), (5,)]
        # testAxis(metricVectorProcessRepetition, dataDimOrd, "pr", extraAxis=extraAxisVectorProcessRepetition, metricSettings=metricSettings)
        #
        # ########################################
        # # TEST 2.1: Required Dim: test raises error if required dimension missing
        # ########################################
        # metricMustHaveSamples = ["autocorr", "autocorr_d1", "ar1_coeff", "ar1_testerr", "cumul_ord", "cumul_ord_coeff", "avg_PI", "temporal_basis"]
        # testRaises(metricMustHaveSamples, "s")
        #
        # metricMustHaveProcesses = ["corr", "avgcorr", "crosscorr", "cumul_ord", "cumul_ord_coeff"]
        # testRaises(metricMustHaveProcesses, "p")
