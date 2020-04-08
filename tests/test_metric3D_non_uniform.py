import numpy as np
import unittest

from mesostat.metric.metric_non_uniform import MetricCalculatorNonUniform


class TestMetrics3DNonUniform(unittest.TestCase):

    # Computing rel. error of approximate number of combinations for small values
    def test_shapes(self):
        # Create dummy data
        nProcess = 12
        nSampleMin = 150
        nSampleMax = 250
        nRepetition = 100

        dataDimOrd = "ps"
        dataLst = []
        for iRep in range(nRepetition):
            nSampleThis = np.random.randint(nSampleMin, nSampleMax)
            dataLst += [np.random.uniform(0, 1, (nProcess, nSampleThis))]

        # Find out actual minimal number of samples to plug into truncated autoregression
        nSampleMinEff = np.min([data.shape[1] for data in dataLst])

        # Settings for some of the metrics
        metricSettings = {"max_lag" : 3, "lag" : 5}

        mc = MetricCalculatorNonUniform(serial=False)
        mc.set_data(dataLst, zscoreChannel=True)

        # Test if result of the metric has expected shape, based on source and target axis
        def testAxis(metrics, srcDimOrder, trgDimOrder, extraAxis=None, metricSettings=None):
            shapeDict = {"p" : nProcess, "r" : nRepetition}
            trgShape = tuple([shapeDict[e] for e in trgDimOrder])

            for iMetric, metricName in enumerate(metrics):
                trgShapeEff = trgShape if extraAxis is None else trgShape + extraAxis[iMetric]
                rez = mc.metric3D(metricName, trgDimOrder, metricSettings=metricSettings)
                print(metricName, trgDimOrder, rez.shape, trgShapeEff)
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
        metricScalarFull = ["mean", "std", "autocorr_d1", "avgcorr", "avg_entropy", "cumul_ord_coeff", "avg_PI"]
        for metricName in metricScalarFull:
            rez = mc.metric3D(metricName, "", metricSettings=metricSettings)
            print(metricName, rez)
            self.assertTrue(np.isscalar(rez))
            self.assertTrue(~np.isnan(rez))

        # Vector
        metricVectorFull = ["autocorr", "corr", "crosscorr", "cumul_ord", "temporal_basis"]
        extraAxisVectorFull = [(nSampleMinEff,), (nProcess, nProcess), (nProcess, nProcess), (nProcess, nProcess), (5, )]
        testAxis(metricVectorFull, dataDimOrd, "", extraAxis=extraAxisVectorFull, metricSettings=metricSettings)

        ######################################
        # TEST 1.2: Shape: scalar: 1D projections
        ######################################
        # Scalar
        metricScalarProcess = ["mean", "std", "autocorr_d1", "avg_entropy", "avg_PI"]
        testAxis(metricScalarProcess, dataDimOrd, "p", metricSettings=metricSettings)

        # Vector
        metricVectorProcess = ["autocorr", "temporal_basis"]
        extraAxisVectorProcess = [(nSampleMinEff,), (5,)]
        testAxis(metricVectorProcess, dataDimOrd, "p", extraAxis=extraAxisVectorProcess, metricSettings=metricSettings)

        #######################################
        # TEST 1.1.3: Shape: scalar: 2D projections
        #######################################
        # Scalar
        metricScalarProcessRepetition = ["mean", "std", "autocorr_d1", "avg_entropy", "avg_PI"]
        testAxis(metricScalarProcessRepetition, dataDimOrd, "pr", metricSettings=metricSettings)

        # Vector
        metricVectorProcessRepetition = ["temporal_basis"]
        extraAxisVectorProcessRepetition = [(5,)]
        testAxis(metricVectorProcessRepetition, dataDimOrd, "pr", extraAxis=extraAxisVectorProcessRepetition, metricSettings=metricSettings)

        ########################################
        # TEST 2.1: Required Dim: test raises error if required dimension missing
        ########################################
        metricMustHaveSamples = list(mc.metricDict.keys())  # Looping over samples is completely disallowed for this iterator
        testRaises(metricMustHaveSamples, "s")

        metricMustHaveProcesses = ["corr", "avgcorr", "crosscorr", "cumul_ord", "cumul_ord_coeff"]
        testRaises(metricMustHaveProcesses, "p")
