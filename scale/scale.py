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
import time
import numpy as np

# All scalers used in case SPARK=False
import sklearn.preprocessing

# All scalers used in case SPARK=True
import pyspark.ml.feature

# Spark utils
from ikats.core.library.spark import SSessionManager, SparkUtils
from pyspark.ml.feature import VectorAssembler

# Ikats utils
from ikats.core.resource.api import IkatsApi
from ikats.core.data.ts import TimestampedMonoVal
from ikats.core.library.exception import IkatsException, IkatsConflictError

"""
Scale Algorithm (also named Normalize):
For now, scaler used are:
- Standard Scaler (also called Z-Norm): (X - mean) / std
- MinMax Scaler : (X - X.min) / (X.max - X.min)
- MaxAbs Scaler: X / max( abs(X.max), abs(X.min) )

Each scaler need to have an implementation in sklearn AND pyspark.
If you want to add a scaler, please just update classes `AvailableScaler` and `ScalerName`.
"""

# Define a logger for this algorithm
LOGGER = logging.getLogger(__name__)


class AvailableScaler(enumerate):
    """
    Class containing list of (unique) scaler names to propose to user.
    """
    # Standard scaler: (X - mean) / std
    ZNorm = "Z-Norm"
    # Min Max scaler: (X - min) / (max - min)
    MinMax = "MinMax"
    # Max absolute scaler: X / abs(max)
    MaxAbs = "MaxAbs"


# Dict containing list of scaler used in algo.
# Structure of the result: dict
# {'no_spark': sklearn.preprocessing scaler,
# 'spark': pyspark.ml.feature scaler}
#
# These objects are not init (for 'spark' scalers, need to init a Spark Context !)
SCALER_DICT = {
    AvailableScaler.ZNorm: {'no_spark': sklearn.preprocessing.StandardScaler,
                            'spark': pyspark.ml.feature.StandardScaler},
    # 'no_spark': need to set `copy=False`
    # 'spark': need to set `withMean=True` (center data before scaling: X-mean)
    AvailableScaler.MinMax: {'no_spark': sklearn.preprocessing.MinMaxScaler,
                             'spark': pyspark.ml.feature.MinMaxScaler},
    # 'no_spark': need to set `copy=False`
    AvailableScaler.MaxAbs: {'no_spark': sklearn.preprocessing.MaxAbsScaler,
                             'spark': pyspark.ml.feature.MaxAbsScaler}
    # 'no_spark': need to set `copy=False`
}
# Example: To init sklearn (-> no spark)  Standard Scaler :
# SCALER_DICT[AvailableScaler.ZNorm]['no_spark']()

# (Intermediate) Names of created columns (during spark operation only)
_INPUT_COL = "features"
_OUTPUT_COL = "scaledFeatures"

class Scaler(object):
    """
    Wraper of sklearn / pyspark.ml scalers.
    Usefull for init `ScalerName` content
    """

    def __init__(self, scaler=AvailableScaler.ZNorm, spark=False):
        """
        Init `Scaler` object.

        :param scaler: The name of the scaler used. Default: Z-Norm
        :type scaler: str

        :param spark: Use spark scaler (case True) or not. Default False.
        :type spark: Bool
        """
        self.spark = spark

        # Init self.scaler
        self._get_scaler(scaler)

    def _get_scaler(self, scaler=AvailableScaler.ZNorm):
        """
        Init scaler.
        Note that if self.spark=True, a SparkContext need to be init !

        :param scaler: The name of the scaler used. Default: Z-Norm
        :type scaler: str

        :return: A Scaler
        """
        # CASE Spark=True (pyspark scaler)
        # -----------------------------
        if self.spark:
            # Init SparkSession (necessary for init Spark Scaler)
            SSessionManager.get()

            # Init pyspark.ml.feature scaler object
            self.scaler = SCALER_DICT[scaler]['spark']()

            # Additional arguments to set
            # --------------------------------
            # Set input / output columns names (necessary for Spark functions)
            self.scaler.inputCol = _INPUT_COL
            self.scaler.outputCol = _OUTPUT_COL

            if scaler == AvailableScaler.ZNorm:
                # By default, spark's Standard scaler does not center the data (X-mean)
                self.scaler.setWithMean(True)

        # CASE Spark=False (sklearn scaler)
        # -------------------------------
        else:
            # Init sklearn.preprocessing scaler object
            self.scaler = SCALER_DICT[scaler]['no_spark']()

            # TODO: test diff with copy=True
            # Additional arguments to set
            # --------------------------------
            # Perform an inplace scaling (for performance)
            self.scaler.copy = False

    def perform_scaling(self, X):
        """
        Perform scaler.
            - replace the sklearn's `fit_transform()`
            - replace le pyspark's `fit().transform()`

        :param X: The data to scale
        :type X: np.array, pyspark.sql.DataFrame

        :return: Object X scaled with `self.scaler`
        :rtype: type(X)
        """
        # CASE : Use spark = True: use lib `pyspark.ml.feature`
        if self.spark:
            return self.scaler.fit(X).transform(X)

        # CASE : Use spark = False: use lib `sklearn.preprocessing`
        else:
            return self.scaler.fit_transform(X)


def scale(ts_list, scaler=AvailableScaler.ZNorm):
    """
    Compute a scaling on a provided ts list ("no spark" mode).

    :param ts_list: List of TS to scale
    :type ts_list: list of str

    :param scaler: The scaler used, should be one of the AvailableScaler...
    :type scaler: str

    :return: A list of dict composed of original TSUID and the information about the new TS
    :rtype: list

    ..Example: result=[{"tsuid": new_tsuid,
                        "funcId": new_fid
                        "origin": tsuid
                        }, ...]
    """
    # Init result, list of dict
    result = []

    # 0/ Init Scaler object
    # ------------------------------------------------
    # Init Spark Scaler
    current_scaler = Scaler(scaler=scaler, spark=False)

    # Perform operation iteratively on each TS
    for tsuid in ts_list:

        # 1/ Load TS content
        # ------------------------------------------------
        start_loading_time = time.time()

        # Read TS from it's ID
        ts_data = IkatsApi.ts.read([tsuid])[0]
        # ts_data is np.array, shape = (2, nrow)

        LOGGER.debug("TSUID: %s, Gathering time: %.3f seconds", tsuid, time.time() - start_loading_time)

        # 2/ Perform scaling
        # ------------------------------------------------
        start_computing_time = time.time()

        # ts_data is np.array [Time, Value]: apply scaler on col `Value` ([:, 1])
        # Need to reshape this col into a (1, n_row) dataset (sklearn request)
        scaled_data = current_scaler.perform_scaling(X=ts_data[:, 1].reshape(-1, 1))

        LOGGER.debug("TSUID: %s, Computing time: %.3f seconds", tsuid, time.time() - start_computing_time)

        # 3/ Merge [Dates + new_values] and save
        # ------------------------------------------------
        ts_result = TimestampedMonoVal(np.dstack((ts_data[:, 0], scaled_data.flat))[0])

        # 4/ Save result
        # ------------------------------------------------
        # Save the result
        start_saving_time = time.time()
        short_name = "scaled"
        new_tsuid, new_fid = save(tsuid=tsuid,
                                  ts_result=ts_result,
                                  short_name=short_name,
                                  sparkified=False)

        # Inherit from parent
        IkatsApi.ts.inherit(new_tsuid, tsuid)

        LOGGER.debug("TSUID: %s(%s), Result import time: %.3f seconds", new_fid, new_tsuid,
                     time.time() - start_saving_time)

        # 4/ Update result
        # ------------------------------------------------
        result.append({
            "tsuid": new_tsuid,
            "funcId": new_fid,
            "origin": tsuid
        })

    return result


def spark_scale(ts_list, scaler=AvailableScaler.ZNorm, nb_points_by_chunk=50000):
    """
    Compute a scaling on a provided ts list ("spark" mode).

    :param ts_list: List of TS to scale
    :type ts_list: list of str

    :param scaler: The scaler used, should be one of the AvailableScaler...
    :type scaler: str

    :param nb_points_by_chunk: size of chunks in number of points
    (assuming points of the time series are in same frequency and without holes)
    :type nb_points_by_chunk: int

    :return: A list of dict composed of original TSUID and the information about the new TS
    :rtype: list

    ..Example: result=[{"tsuid": new_tsuid,
                        "funcId": new_fid
                        "origin": tsuid
                        }, ...]
    """
    # 0/ Init Scaler object
    # ------------------------------------------------
    current_scaler = Scaler(scaler=scaler, spark=True)

    # Init result, list of dict
    result = []

    # Checking metadata availability before starting computation
    meta_list = IkatsApi.md.read(ts_list)

    # Perform operation iteratively on each TS
    for tsuid in ts_list:

        try:
            # 1/ Retrieve meta data
            # --------------------------------------------------------------------------
            md = meta_list[tsuid]
            period = int(float(md['qual_ref_period']))
            sd = int(md['ikats_start_date'])
            ed = int(md['ikats_end_date'])

            # 2/ Get data
            # --------------------------------------------------------------------------
            # Import data into dataframe (["Index", "Timestamp", "Value"])
            df, size = SSessionManager.get_ts_by_chunks_as_df(tsuid=tsuid,
                                                              sd=sd,
                                                              ed=ed,
                                                              period=period,
                                                              nb_points_by_chunk=nb_points_by_chunk)

            # 3/ Calculate scaling
            # -------------------------------
            # TODO: from HERE, put this code into `Scale` class
            # from
            # https://stackoverflow.com/questions/47705919/spark-get-the-actual-cluster-centeroids-with-standardscaler

            # OPERATION: Transform into `Vector` object (necessary for scaling with spark)
            # INPUT: Spark DataFrame, with column ['Time', 'Value']
            # OUTPUT: Spark Dataframe with columns ['Time', 'Value', _INPUT_COL]
            # New col `_INPUT_COL` containing data to scale
            assembler = VectorAssembler(inputCols=["Value"], outputCol=_INPUT_COL)

            # Perform operation
            data = assembler.transform(df)
            # TODO: END

            # OPERATION: Perform scaling on data frame
            # INPUT: Spark Dataframe with columns ['Time', 'Value', _INPUT_COL]
            # OUTPUT: Spark Dataframe with columns ['Time', 'Value', _INPUT_COL, _OUTPUT_COL]
            # New col `_OUTPUT_COL` containing scaled data
            scaledData = current_scaler.perform_scaling(data)

            # OPERATION: Reformat dataframe to obtain ['Time', 'Value'] with proper data
            # INPUT: Spark Dataframe with columns ['Time', 'Value', _INPUT_COL, _OUTPUT_COL]
            # OUTPUT: Spark DataFrame, with column ['Time', 'Value']
            # Column 'Value' contains scaled data
            scaledData = scaledData.drop(['Value', _INPUT_COL]).\
                withColumnRenamed(_OUTPUT_COL, 'Value')

            # 4/ Save result
            # ------------------------------------------------
            # Save the result
            start_saving_time = time.time()
            short_name = "scaled"

            # Generate the new functional identifier (fid) for the current TS (ts_uid)
            new_fid = gen_fid(tsuid=tsuid, short_name=short_name)

            # OPERATION: Import result by partition into database, and collect
            # INPUT: [Timestamp, rollmean]
            # OUTPUT: the new tsuid of the rollmean ts (not used)
            scaledData.rdd.mapPartitions(lambda x: SparkUtils.save_data(fid=new_fid,
                                                                        data=list(x))).collect()

            # Retrieve tsuid of the saved TS
            new_tsuid = IkatsApi.fid.tsuid(new_fid)
            LOGGER.debug("TSUID: %s(%s), Result import time: %.3f seconds", new_fid, new_tsuid,
                         time.time() - start_saving_time)

            # TODO: Here, meta data are un-changed (?)
            # # Inherit from parent
            # IkatsApi.ts.inherit(new_tsuid, tsuid)
            #
            # # store metadata ikats_start_date, ikats_end_date and qual_nb_points
            # if not IkatsApi.md.create(
            #         tsuid=new_tsuid,
            #         name='ikats_start_date',
            #         value=sd + time_gap,
            #         data_type=DTYPE.date,
            #         force_update=True):
            #     LOGGER.error("Metadata ikats_start_date couldn't be saved for TS %s", new_tsuid)
            #
            # if not IkatsApi.md.create(
            #         tsuid=new_tsuid,
            #         name='ikats_end_date',
            #         value=ed + time_gap,
            #         data_type=DTYPE.date,
            #         force_update=True):
            #     LOGGER.error("Metadata ikats_end_date couldn't be saved for TS %s", new_tsuid)
            #
            # # Retrieve imported number of points from database
            # qual_nb_points = IkatsApi.ts.nb_points(tsuid=new_tsuid)
            # if not IkatsApi.md.create(
            #         tsuid=new_tsuid,
            #         name='qual_nb_points',
            #         value=qual_nb_points,
            #         data_type=DTYPE.number,
            #         force_update=True):
            #     LOGGER.error("Metadata qual_nb_points couldn't be saved for TS %s", new_tsuid)

        except Exception:
            raise
        finally:
            SSessionManager.stop()

        result.append({
            "tsuid": new_tsuid,
            "funcId": new_fid,
            "origin": tsuid
        })

    # END FOR (op. on all TS performed)

    return result


def scale_ts_list(ts_list, scaler=AvailableScaler.ZNorm, nb_points_by_chunk=50000, spark=None):
    """
    Wrapper
    Compute a scaling on a provided ts_list

    :param ts_list: List of TS to scale
    :type ts_list: list of str

    :param scaler: The scaler used, should be one of the AvailableScaler...
    :type scaler: str

    :param nb_points_by_chunk: size of chunks in number of points (assuming time series is periodic and without holes)
    :type nb_points_by_chunk: int

    :param spark: Flag indicating if Spark usage is:
        * forced (case True),
        * forced to be not used (case False)
        * case None: Spark usage is checked (function of amount of data)
    For TU only ! default None
    :type spark: bool or NoneType

    :return: A list of dict composed of original TSUID and the information about the new TS
    :rtype: list

    ..Example: result=[{"tsuid": new_tsuid,
                        "funcId": new_fid
                        "origin": tsuid
                        }, ...]
    """
    # 0/ Check inputs
    # ----------------------------------------------------------
    # TS list
    if type(ts_list) is not list:
        raise TypeError("Arg. type `ts_list` is {}, expected `list`".format(type(ts_list)))
    if len(ts_list) == 0:
        raise ValueError("`ts_list` provided is empty !")

    # Scaler
    if type(scaler) is not str:
        raise TypeError("Arg. type `scaler` is {}, expected `str`".format(type(scaler)))
    if scaler not in list(SCALER_DICT.keys()):
        raise ValueError("Arg. `scale` is {}, expected element in {}".format(scaler, list(SCALER_DICT.keys())))

    # Nb points by chunk
    if type(nb_points_by_chunk) is not int or nb_points_by_chunk < 0:
        raise TypeError("Arg. `nb_points_by_chunk` must be an integer > 0, get {}".format(nb_points_by_chunk))

    # Spark
    if type(spark) is not bool and spark is not None:
        raise TypeError("Arg. type `spark` is {}, expected `bool` or `NoneType`".format(type(spark)))

    # 1/ Check for spark usage and run
    # ----------------------------------------------------------
    if spark is True or (spark is None and SparkUtils.check_spark_usage(tsuid_list=ts_list,
                                                                        nb_ts_criteria=100,
                                                                        nb_points_by_chunk=nb_points_by_chunk)):
        # Arg `spark=True`: spark usage forced
        # Arg `spark`=None`: Check using criteria (nb_points and number of ts)
        return spark_scale(ts_list=ts_list, scaler=scaler, nb_points_by_chunk=nb_points_by_chunk)
    else:
        return scale(ts_list=ts_list, scaler=scaler)


# TODO: put these two functions into module
def save(tsuid, ts_result, short_name="scaled", sparkified=False):
    """
    Saves the TS to database
    It copies some attributes from the original TSUID, that is why it needs the tsuid

    :param tsuid: original TSUID used for computation
    :type tsuid: str

    :param ts_result: TS resulting of the operation
    :type ts_result: TimestampedMonoVal

    :param short_name: Name used as short name for Functional identifier
    :type short_name: str

    :param sparkified: set to True to prevent from having multi-processing,
                       and to handle correctly the creation of TS by chunk
    :type sparkified: bool

    :return: the created TSUID and its associated FID
    :rtype: str, str

    :raise IOError: if an issue occurs during the import
    """
    if type(ts_result) is not TimestampedMonoVal:
        raise TypeError('Arg `ts_result` is {}, expected TimestampedMonoVal'.format(type(ts_result)))

    try:
        # Generate new FID
        new_fid = gen_fid(tsuid=tsuid, short_name=short_name)

        # Import time series result in database
        res_import = IkatsApi.ts.create(fid=new_fid,
                                        data=ts_result.data,
                                        generate_metadata=True,
                                        parent=tsuid,
                                        sparkified=sparkified)
        return res_import['tsuid'], new_fid

    except Exception:
        raise IkatsException("save_rollmean() failed")


def gen_fid(tsuid, short_name="scaled"):
    """
    Generate a new functional identifier (fid) for current TS (`tsuid`).
    Return new fid (`original_fid`_`short_name`). If already exist, create new fid
    (`original_fid`_`short_name`_`time * 1000`).

    :param tsuid: original TSUID used for computation
    :type tsuid: str

    :param short_name: Name used as short name for Functional identifier
    :type short_name: str

    :return: The new fid of the TS to create.
    :rtype: str
    """
    # Retrieve time series information (funcId)
    original_fid = IkatsApi.ts.fid(tsuid=tsuid)

    # Generate unique functional id for resulting time series
    new_fid = '%s_%s' % (str(original_fid), short_name)
    try:
        IkatsApi.ts.create_ref(new_fid)
    except IkatsConflictError:
        # TS already exist, append timestamp to be unique
        new_fid = '%s_%s_%s' % (str(original_fid), short_name, int(time.time() * 1000))
        IkatsApi.ts.create_ref(new_fid)

    return new_fid


