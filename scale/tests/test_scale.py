"""
Copyright 2018 CS Systèmes d'Information

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import logging
import unittest
import numpy as np

from ikats.algo.scale.scale import AvailableScaler, Scaler, scale_ts_list, SCALER_DICT
from ikats.core.resource.api import IkatsApi

import sklearn.preprocessing

# Set LOGGER
LOGGER = logging.getLogger()
# Log format
LOGGER.setLevel(logging.DEBUG)
FORMATTER = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
# Create another handler that will redirect log entries to STDOUT
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setLevel(logging.DEBUG)
STREAM_HANDLER.setFormatter(FORMATTER)
LOGGER.addHandler(STREAM_HANDLER)

# All use case tested during this script.
# Keys are argument `ts_id` of function `gen_ts`
USE_CASE = {
    1: "Null average",
    2: "Linear curve",
    3: "Constant value"
}


def gen_ts(ts_id):
    """
    Generate a TS in database used for test bench where id is defined

    :param ts_id: Identifier of the TS to generate (see content below for the structure)
    :type ts_id: int

    :return: the TSUID, funcId and all expectd result (one per scaling)
    :rtype: dict
    """

    # Build TS identifier
    fid = "UNIT_TEST_Scale_%s" % ts_id

    # Create np.array with shape (n_row, 2) ([time, value])

    if ts_id == 1:
        # CASE: avg=0
        # ----------------
        ts_content = np.array([list(range(14879030000, 14879039000, 1000)),
                               [-1., -2.,  1.,  2.,  0.,  3., -3.,  4., -4.]],
                              np.float64).T
        # Average: 0, Standard deviation: 2.58198890

        # Expected result
        # ----------------
        # scaled with Z-Norm (X - mean / correct_std)
        ts_content_znorm = np.array(ts_content[:, 1]) / np.std(ts_content[:, 1], ddof=1)
        # Correct_std = np.std(X, doof=1)
        # Correct std = sqrt(1/(N-1) sum(x - mean)² )

        # scaled with MinMax scaler (X - X.min) / (X.max - X.min)
        ts_content_minmax = (np.array(ts_content[:, 1]) - -4) / (4 - -4)
        # scaled with MaxAbs scaler X / max( abs(X.max), abs(X.min)) )
        ts_content_maxabs = np.array(ts_content[:, 1]) / 4

    elif ts_id == 2:
        # CASE: linear curve
        ts_content = np.array([list(range(14879030000, 14879039000, 1000)),
                               list(range(9))],
                              np.float64).T
        # Average = 4, Standard deviation = 2.58198890e+00

        # Expected result
        # ----------------
        # scaled with Z-Norm (X - mean / correct_std)
        ts_content_znorm = (np.arange(9) - 4) / np.std(np.arange(9), ddof=1)
        # Correct_std = np.std(X, doof=1)
        # Correct std = sqrt(1/(N-1) sum(x - mean)² )
        # scaled with MinMax scaler (X - X.min) / (X.max - X.min)
        ts_content_minmax = np.arange(9) / 8.
        # scaled with MaxAbs scaler X / max( abs(X.max), abs(X.min)) )
        ts_content_maxabs = ts_content_minmax  # same result than previous: min=0

    elif ts_id == 3:
        # CASE: Constant value
        ts_content = np.array([list(range(14879030000, 14879039000, 1000)),
                               [1.]*9],
                              np.float64).T
        # Average = 0., Standard deviation = 0.

        # Expected result
        # ----------------
        # scaled with Z-Norm (X - mean / correct_std)
        ts_content_znorm = np.array([0] * 9)
        # scaled with MinMax scaler (X - X.min) / (X.max - X.min), 0.5 * max if min=max
        ts_content_minmax = np.array([0.5] * 9)
        # scaled with MaxAbs scaler X / max( abs(X.max), abs(X.min)) )
        ts_content_maxabs = np.array([1.] * 9)  # same result than previous: min=0

    else:
        raise NotImplementedError

    # Create the timeseries
    result = IkatsApi.ts.create(fid=fid, data=np.array(ts_content))
    # TODO: modify this, when pb with period is fix
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_ref_period", value=1000, force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="qual_nb_points", value=len(ts_content), force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="metric", value="metric_%s" % ts_id, force_update=True)
    IkatsApi.md.create(tsuid=result['tsuid'], name="funcId", value="fid_%s" % ts_id, force_update=True)
    if not result['status']:
        raise SystemError("Error while creating TS %s" % ts_id)

    return {"tsuid": result['tsuid'],
            "funcId": fid,
            "ts_content": ts_content,
            "expected_" + AvailableScaler.ZNorm: ts_content_znorm,     # Store the 3 expected results (3 scalers)
            "expected_" + AvailableScaler.MinMax: ts_content_minmax,
            "expected_" + AvailableScaler.MaxAbs: ts_content_maxabs}


class TesScale(unittest.TestCase):
    """
    Test the scale algorithm (results are rounded with 6 digits)
    """

    @staticmethod
    def round_result(np_array, digits=6):
        """
        Round numpy array elements, and transform result into list.

        :param np_array: The array to round
        :type np_array: np.array

        :param digits: umber of digits used after rounding (here 6)
        :type digits: int

        :return: List containing `np_array` input array, rounded.
        :rtype: list
        """
        return [round(i, digits) for i in np_array]

    def test_Scaler(self):
        """
        Testing class `Scaler`
        """
        # Test default implementation (Z-norm, no spark)
        # ----------------------------------------------
        # -> Should be object sklearn.preprocessing.StandardScaler
        value = Scaler().scaler
        expected_type = sklearn.preprocessing.StandardScaler
        msg="Error in init `Scaler` object, get type {}, expected type {}"

        self.assertEqual(type(value), expected_type, msg=msg.format(type(value), expected_type))

        # -> Arg copy` should be set to `False`
        msg = "Error in init `Scaler`, arg `copy` is {}, should be set to `False` "
        self.assertFalse(value.copy, msg=msg.format(value.copy))

        # Test implementation Z-norm with spark
        # ----------------------------------------------
        # -> Arg `WithMean` should be set to `True`
        msg = "Error in init `Scaler` object (Z-Norm with spark), arg `WithMean` is {}, expected `True`"
        result = Scaler(spark=True).scaler.getWithMean()
        self.assertTrue(result, msg=msg.format(result))

    def test_arguments_scale_ts_list(self):
        """
        Testing behaviour when wrong arguments on function `scale_ts_list`.
        """
        # Get the TSUID of the saved TS
        tsuid = gen_ts(1)['tsuid']

        try:

            # TS list
            # ----------------------------
            # Wrong type ((not list)
            msg = "Testing arguments : Error in testing `ts_list` type"
            with self.assertRaises(TypeError, msg=msg):
                scale_ts_list(ts_list=0.5)

            # empty TS list
            msg = "Testing arguments : Error in testing `ts_list` as empty list"
            with self.assertRaises(ValueError, msg=msg):
                scale_ts_list(ts_list=[])

            # Un-existant TS
            msg = "Testing arguments : Error in testing un-existant `ts_list`"
            with self.assertRaises(ValueError, msg=msg):
                scale_ts_list(ts_list=['TS which does not exist'])

            # scaler
            # ----------------------------
            # wrong type (not str)
            msg = "Testing arguments : Error in testing `scale` type"
            with self.assertRaises(TypeError, msg=msg):
                scale_ts_list(ts_list=[tsuid], scaler=1.0)

            # wrong element (not in SCALER_DICT
            msg = "Testing arguments : Error in testing `scale` unexpected value"
            with self.assertRaises(ValueError, msg=msg):
                scale_ts_list(ts_list=[tsuid], scaler="Scaler which does not exist")

            # nb_points_by_chunk
            # ----------------------------
            # Wrong type (not int)
            msg = "Testing arguments : Error in testing `nb_point_by_chunk` type"
            with self.assertRaises(TypeError, msg=msg):
                scale_ts_list(ts_list=[tsuid], nb_points_by_chunk="a")

            # Not > 0
            msg = "Testing arguments : Error in testing `nb_point_by_chunk` negative value"
            with self.assertRaises(TypeError, msg=msg):
                scale_ts_list(ts_list=[tsuid], nb_points_by_chunk=-100)

            # spark
            # ----------------------------
            # Wrong type (not NoneType or bool)
            msg = "Testing arguments : Error in testing `spark` type"
            with self.assertRaises(TypeError, msg=msg):
                scale_ts_list(ts_list=[tsuid], spark="True")

        finally:
            # Clean up database
            IkatsApi.ts.delete(tsuid, True)

    def test_scale_value(self):
        """
        Testing the result values of the scale algorithm.
        """
        # For each Available scaler
        for scaler in list(SCALER_DICT.keys()):

            # For each use-case
            for case in list(USE_CASE.keys()):
                # CASE 1:avg=0
                # CASE 2: Linear curve
                # CASE 3: Constant value

                result = gen_ts(case)
                tsuid = result['tsuid']
                # Expected result (rounded with k digits)
                expected = self.round_result(result['expected_' + scaler])
                try:

                    # Perform scaling, and get the resulting tsuid
                    result_tsuid = scale_ts_list([tsuid], scaler=scaler, spark=False)[0]['tsuid']

                    # Get results (column "Value" is [:, 1])
                    result_values = IkatsApi.ts.read(result_tsuid)[0][:, 1]

                    # Round result (default k digits): else, raise error
                    result_values = self.round_result(result_values)

                    # Standard Scaler on constant data, result = list of 0.
                    msg = "Error in result of {} 'no spark' mode (case {}):\n" \
                          " get: {},\nexpected: {}, \ndiff: {}".format(scaler,
                                                                       case,
                                                                       result_values,
                                                                       expected,
                                                                       [result_values[i] - expected[i] for i in range(len(expected))])

                    self.assertEqual(result_values, expected, msg=msg)

                finally:
                    IkatsApi.ts.delete(tsuid, True)

    def test_spark(self):
        """
        Testing the result values of the scale algorithm, when spark is forced true.
        """
        # For each Available scaler
        for scaler in list(SCALER_DICT.keys()):
            # For each use-case
            for case in list(USE_CASE.keys()):
                # CASE 1:avg=0
                # CASE 2: Linear curve
                # CASE 3: Constant value

                result = gen_ts(case)
                tsuid = result['tsuid']
                # Expected result (rounded with k digits)
                expected = self.round_result(result['expected_' + scaler])

                try:

                    # Force spark usage
                    result_tsuid = scale_ts_list([tsuid], scaler=scaler, spark=True)[0]['tsuid']

                    # Get results (column "Value" is [:, 1])
                    result_values = IkatsApi.ts.read(result_tsuid)[0][:, 1]
                    # Round result (default k digits): else, raise error
                    result_values = self.round_result(result_values)

                    # Standard Scaler on constant data, result = list of 0.
                    msg = "Error in result of {} 'Spark' mode (case {}):\n" \
                          " get {},\n expected {}\n" \
                          "difference: {}".format(scaler,
                                                  case,
                                                  result_values,
                                                  expected,
                                                  [result_values[i] - expected[i] for i in range(len(expected))])

                    # Standard Scaler on constant data, result = list of 0.
                    self.assertEqual(result_values, expected, msg=msg)

                finally:
                    IkatsApi.ts.delete(tsuid, True)

    def test_diff_spark(self):
        """
        Testing difference of result between "Spark" and "No Spark"
        """
        # For each Available scaler
        for scaler in list(SCALER_DICT.keys()):

            # For each use-case
            for case in list(USE_CASE.keys()):
                # CASE 1:avg=0
                # CASE 2: Linear curve
                # CASE 3: Constant value

                tsuid = gen_ts(case)['tsuid']

                try:
                    # Result with spark FORCED
                    # TODO: Perhaps here, modify here nb_pt_by_chunks (?)
                    # Force spark usage
                    result_tsuid_spark = scale_ts_list([tsuid], scaler=scaler, spark=True)[0]['tsuid']

                    # Get results (column "Value" is [:, 1])
                    result_spark = IkatsApi.ts.read(result_tsuid_spark)[0][:, 1]
                    # Round result (default k digits): else, raise error
                    result_spark = self.round_result(result_spark)

                    # Result with NO Spark Forced
                    result_tsuid_no_spark = scale_ts_list([tsuid], scaler=scaler, spark=False)[0]['tsuid']

                    # Get results (column "Value" is [:, 1])
                    result_no_spark = IkatsApi.ts.read(result_tsuid_no_spark)[0][:, 1]
                    # Round result (default k digits): else, raise error
                    result_no_spark = self.round_result(result_no_spark)

                    msg = "Error in compare Spark/no spark: case {} ({}) \n" \
                          "Result Spark: {} \n" \
                          "Result no spark {}.\n" \
                          "Difference: {}".format(case,
                                                  USE_CASE[case],
                                                  result_spark,
                                                  result_no_spark,
                                                  [result_spark[i] - result_no_spark[i] for i
                                                   in range(len(result_spark))])

                    self.assertEqual(result_spark, result_no_spark, msg=msg)

                finally:
                    IkatsApi.ts.delete(tsuid, True)
