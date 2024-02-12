# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
"""Data access module.

Superclass containing some common attributes and helper functions used in multiple data access classes

"""

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Union

import pandas as pd

from pyelq.meteorology import Meteorology, MeteorologyGroup
from pyelq.sensor.sensor import Sensor, SensorGroup


@dataclass
class DataAccess(ABC):
    """DataAccess superclass containing some common attributes and functionalities.

    This superclass is used to show the type of methods to implement when creating a new data access class. The data
    access classes are used to convert raw data into well-defined classes and objects which can be used by the rest of
    the package.

    Attributes:
        latitude_bounds (tuple, optional): Tuple specifying (latitude_min, latitude_max)
        longitude_bounds (tuple, optional): Tuple specifying (longitude_min, longitude_max)
        date_bounds (tuple, optional): Tuple specifying (datetime_min, datetime_max)

    """

    latitude_bounds: tuple = (None, None)
    longitude_bounds: tuple = (None, None)
    date_bounds: tuple = (None, None)

    @abstractmethod
    def to_sensor(self, *args: Any, **kwargs: dict) -> Union[Sensor, SensorGroup]:
        """Abstract method to convert raw data into a Sensor or SensorGroup object.

        This method should be implemented to convert the raw data into a Sensor or SensorGroup object.

        Args:
            *args (Any): Variable length argument list of any type.
            **kwargs (dict): Arbitrary keyword arguments

        """

    @abstractmethod
    def to_meteorology(self, *args: Any, **kwargs: dict) -> Union[Meteorology, MeteorologyGroup]:
        """Abstract method to convert raw data into a Meteorology or MeteorologyGroup object.

        This method should be implemented to convert the raw data into a Meteorology or MeteorologyGroup object.

        Args:
            *args (Any): Variable length argument list of any type.
            **kwargs (dict): Arbitrary keyword arguments

        """

    def _query_aoi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Helper function to perform area of interest query on data.

        Args:
            data (pd.Dataframe): Pandas dataframe to perform the query on

        """
        aoi_query_string = ""
        if self.latitude_bounds[0] is not None:
            aoi_query_string += f" & latitude>={self.latitude_bounds[0]}"
        if self.latitude_bounds[1] is not None:
            aoi_query_string += f" & latitude<={self.latitude_bounds[1]}"
        if self.longitude_bounds[0] is not None:
            aoi_query_string += f" & longitude>={self.longitude_bounds[0]}"
        if self.longitude_bounds[1] is not None:
            aoi_query_string += f" & longitude<={self.longitude_bounds[1]}"
        if len(aoi_query_string) > 0:
            aoi_query_string = aoi_query_string[3:]
            return data.query(aoi_query_string).copy()
        return data

    def _query_time(self, data: pd.DataFrame) -> pd.DataFrame:
        """Helper function to perform time query on data.

        Args:
            data (pd.Dataframe): Pandas dataframe to perform the query on

        """
        time_query_string = ""
        if self.date_bounds[0] is not None:
            timestamp_min = dt.datetime.timestamp(self.date_bounds[0])
            time_query_string += f" & timestamp>={timestamp_min}"
        if self.date_bounds[1] is not None:
            timestamp_max = dt.datetime.timestamp(self.date_bounds[1])
            time_query_string += f" & timestamp<={timestamp_max}"
        if len(time_query_string) > 0:
            time_query_string = time_query_string[3:]
            return data.query(time_query_string).copy()
        return data
