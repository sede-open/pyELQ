<!--
SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.

SPDX-License-Identifier: Apache-2.0
-->

# pyELQ
This repository contains the Python Emission Localization and Quantification software we call pyELQ. It is code used 
for gas dispersion modelling, in particular methane emissions detection, localization and quantification. 

***
# Background
The **py**thon **E**mission **L**ocalization and **Q**uantification (pyELQ) code aims to maximize effective use of 
existing measurement data, especially from continuous monitoring solutions. The code has been developed to detect, 
localize, and quantify methane emissions from concentration and wind measurements. It can be used in combination with 
point or beam sensors that are placed strategically on an area of interest.

The algorithms in the pyELQ code are based a Bayesian statistics framework. pyELQ can ingest long-term concentration 
and wind data, and it performs an inversion to predict the likely strengths and locations of persistent methane sources. 
The goal is to arrive at a plausible estimate of methane emissions from an area of interest that matches the measured 
data. The predictions from pyELQ come with uncertainty ranges that are representative of probability density functions 
sampled by a Markov Chain Monte Carlo method. Time series of varying length can be processed by pyELQ: in general, 
the Bayesian inversion leads to a more constrained solution if more high-precision measurement data is available. 
We have tested our code under controlled conditions as well as in operating oil and gas facilities.

The information on the strength and the approximate location of methane emission sources provided by pyELQ can help 
operators with more efficient identification and quantification of (unexpected) methane sources, in order to start 
appropriate mitigating actions accordingly. The pyELQ code is being made available in an open-source environment, 
to support various assets in their quest to reduce methane emissions.

Use cases where the pyELQ code has been applied are described in the following papers:

* IJzermans, R., Jones, M., Weidmann, D. et al. "Long-term continuous monitoring of methane emissions at an oil and gas facility using a multi-open-path laser dispersion spectrometer." Sci Rep 14, 623 (2024). (https://doi.org/10.1038/s41598-023-50081-9)

* Weidmann, D., Hirst, B. et al. "Locating and Quantifying Methane Emissions by Inverse Analysis of Path-Integrated Concentration Data Using a Markov-Chain Monte Carlo Approach." ACS Earth and Space Chemistry 2022 6 (9), 2190-2198  (https://doi.org/10.1021/acsearthspacechem.2c00093)
***
# Installing pyELQ as a package
Suppose you want to use this pyELQ package in a different project.
You can install it from [PyPi](https://pypi.org/project/pyelq-sdk/) through pip 
`pip install pyelq-sdk`.
Or you could clone the repository and install it from the source code. 
After activating the environment you want to install pyELQ in, open a terminal, move to the main pyELQ folder
where pyproject.toml is located and run `pip install .`, optionally you can pass the `-e` flag is for editable mode.
All the main options, info and settings for the package are found in the pyproject.toml file which sits in this repo
as well.

***

# Examples
For some examples on how to use this package please check out these [Examples](https://github.com/sede-open/pyELQ/blob/main/examples)

***

# Contribution
This project welcomes contributions and suggestions. If you have a suggestion that would make this better you can simply open an issue with a relevant title. Don't forget to give the project a star! Thanks again!

For more details on contributing to this repository, see the [Contributing guide](https://github.com/sede-open/pyELQ/blob/main/CONTRIBUTING.md).

***
# Licensing

Distributed under the Apache License Version 2.0. See the [license file](https://github.com/sede-open/pyELQ/blob/main/LICENSE.md) for more information.