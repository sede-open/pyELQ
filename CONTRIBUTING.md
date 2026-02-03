<!--
SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.

SPDX-License-Identifier: Apache-2.0
-->

# Getting started with contributing
We're happy for everyone to contribute to the package by proposing new features, implementing them in a fork and
creating a pull request. In order to keep the codebase consistent we use some common standards and tools for formatting
of the code. We are using poetry to keep our development environment up to date. Please follow the instructions here
https://python-poetry.org/docs/ to install poetry. Next, pull the repo to your local machine, open a terminal window
and navigate to the top directory of this package. Run the commands `poetry install --all-extras` and
`poetry install --with contributor` to install all required tools and dependencies for contributing to this package.

We list the various tools below:
- pylint: Tool to help with the formatting of the code, can be used as a linter in most IDEs, all relevant settings are
contained in the .pylintrc file and additionally controlled through the pyproject.toml file.
- isort: Sorts the inputs, can be used from the command line  `isort .`, use the `--check` flag if you do not want to
reformat the import statements in place but just want to check if imports need to be reformatted.
- black: Formats the code based on PEP standards, can be used from the command line: `black .`, use the `--check` flag
if you do not want to reformat the code in place but just check if files need to be reformatted.
- pydocstyle: Checks if the docstrings for all files and functions are present and follow the same style as specified
in the pyproject.toml file. Used in order to get consistent documentation, can be used as a check from the command line
but will not be able to replace any text, `pydocstyle .`

In case you're unfamiliar with the tools, don't worry we have set up GitHub actions accordingly to format the code before
a push to main.

When you implement a new feature you also need to write additional (unit) tests to show the feature you've implemented
is also working as it should. Do so by creating a file in the appropriate test folder and call that file
test_<new_feature_name>.py. Use pytest to see if your test is passing and use pytest-cov to check the coverage of your
test. The settings in the pyproject.toml file are such that we automatically test for coverage. You can run all tests
through the command line `pytest .`, use the `--cov-report term-missing` flag to show which lines are missing in the
coverage. All test are required to pass before merging into main.

Whenever we merge new code into main, we increase the version of the package manually.
Version release convention used is major.minor.micro.

# Notice

The [codeowners](https://github.com/sede-open/pyELQ//blob/main/CODEOWNERS.md) reserve the right to deny applications
for ‘maintainer’ status or contributions if
the prospective maintainer or contributor is a national of and/or located in a ‘Restricted Jurisdiction’.
(A Restricted Jurisdiction is defined as a country, state, territory or region which is subject to comprehensive
trade sanctions or embargoes namely: Iran, Cuba, North Korea, Syria, the Crimea region of Ukraine (including
Sevastopol) and non-Government controlled areas of Donetsk and Luhansk). For anyone to be promoted to 'maintainer'
status, the prospective maintainer will be required to provide information on their nationality, location, and
affiliated organizations
