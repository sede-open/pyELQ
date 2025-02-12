# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Main pyELQ module."""
__all__ = [
    "component",
    "data_access",
    "dispersion_model",
    "plotting",
    "sensor",
    "support_functions",
    "coordinate_system",
    "dlm",
    "gas_species",
    "meteorology",
    "model",
    "preprocessing",
    "source_map",
]

from warnings import warn

warn(
    "The pyELQ package will move from the pyelq-sdk project to the pyelq project on PyPi in future version. Please condider installing this package through https://pypi.org/project/pyelq/.",
    FutureWarning,
    stacklevel=2,
)
