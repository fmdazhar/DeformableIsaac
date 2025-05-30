


"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "numpy==1.23.5",
    "protobuf==3.20.2",
    "omegaconf==2.3.0",
    "hydra-core==1.3.2",
    "urllib3==1.26.16",
    "moviepy==1.0.3",
    "gymnasium==0.28.1"
]

# Installation operation
setup(
    name="leggeedisaacgymenvs",
    author="NVIDIA",
    version="4.0.0",
    description="RL environments for robot learning in NVIDIA Isaac Sim.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7, 3.8"],
    zip_safe=False,
)

# EOF
