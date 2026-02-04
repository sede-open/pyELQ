# SPDX-FileCopyrightText: 2024 Shell Global Solutions International B.V. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re

with open("pyproject.toml", "r") as file:
    version_content = file.read()
# Match regex for <version="0.0.0a",> pattern
current_semantic_version = re.findall(r'version = "(\d+\.\d+\.[a-zA-Z0-9]+)"', version_content)
major_version, minor_version, patch_version = current_semantic_version[0].split(".")
patch_version = int(re.findall(r"\d+", patch_version)[0])
output_semantic_version = f"{major_version}.{minor_version}.{patch_version}"
print(output_semantic_version)  # Print is required for release in GitHub action
