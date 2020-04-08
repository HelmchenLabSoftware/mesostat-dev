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
        metricSettings = {"max_lag" : 3, "lag" : 5}

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
        # Scalar
        metricScalarFull = ["mean", "std", "autocorr_d1", "ar1_coeff", "ar1_testerr", "avgcorr", "avg_entropy", "cumul_ord_coeff", "avg_PI"]
        for metricName in metricScalarFull:
            rez = mc.metric3D(metricName, "", metricSettings=metricSettings)
            self.assertTrue(np.isscalar(rez))
            self.assertTrue(~np.isnan(rez))

        # Vector
        metricVectorFull = ["autocorr", "corr", "crosscorr", "cumul_ord", "temporal_basis"]
        extraAxisVectorFull = [(nSample,), (nProcess, nProcess), (nProcess, nProcess), (nProcess, nProcess), (5, )]
        testAxis(metricVectorFull, dataDimOrd, "", extraAxis=extraAxisVectorFull, metricSettings=metricSettings)

        #######################################
        # TEST 1.2: Shape: scalar: 1D projections
        #######################################
        # Scalar
        metricScalarSample = ["mean", "std", "avgcorr", "avg_entropy"]
        testAxis(metricScalarSample, dataDimOrd, "s")

        metricScalarProcess = ["mean", "std", "autocorr_d1", "ar1_coeff", "ar1_testerr", "avg_entropy", "avg_PI"]
        testAxis(metricScalarProcess, dataDimOrd, "p", metricSettings=metricSettings)

        # Vector
        metricVectorSample = ["corr"]
        extraAxisVectorSample = [(nProcess, nProcess)]
        testAxis(metricVectorSample, dataDimOrd, "s", extraAxis=extraAxisVectorSample, metricSettings=metricSettings)

        metricVectorProcess = ["autocorr", "temporal_basis"]
        extraAxisVectorProcess = [(nSample,), (5,)]
        testAxis(metricVectorProcess, dataDimOrd, "p", extraAxis=extraAxisVectorProcess, metricSettings=metricSettings)

        ########################################
        # TEST 1.1.3: Shape: scalar: 2D projections
        ########################################
        # Scalar
        metricScalarProcessSample = ["mean", "std", "avg_entropy"]
        testAxis(metricScalarProcessSample, dataDimOrd, "ps")

        metricScalarProcessRepetition = ["mean", "std", "autocorr_d1", "ar1_coeff", "ar1_testerr", "avg_entropy", "avg_PI"]
        testAxis(metricScalarProcessRepetition, dataDimOrd, "pr", metricSettings=metricSettings)

        # Vector
        metricVectorProcessRepetition = ["autocorr", "temporal_basis"]
        extraAxisVectorProcessRepetition = [(nSample,), (5,)]
        testAxis(metricVectorProcessRepetition, dataDimOrd, "pr", extraAxis=extraAxisVectorProcessRepetition, metricSettings=metricSettings)

        ########################################
        # TEST 2.1: Required Dim: test raises error if required dimension missing
        ########################################
        metricMustHaveSamples = ["autocorr", "autocorr_d1", "ar1_coeff", "ar1_testerr", "cumul_ord", "cumul_ord_coeff", "avg_PI", "temporal_basis"]
        testRaises(metricMustHaveSamples, "s")

        metricMustHaveProcesses = ["corr", "avgcorr", "crosscorr", "cumul_ord", "cumul_ord_coeff"]
        testRaises(metricMustHaveProcesses, "p")
