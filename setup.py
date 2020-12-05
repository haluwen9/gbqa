#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
from setuptools import setup

# Use the VERSION file to get version
# version_file = os.path.join(os.path.dirname(__file__), 'underthesea', 'VERSION')
# with open(version_file) as fh:
#     version = fh.read().strip()
version = "0.0.1"

with io.open('requirements.txt') as requirements_file:
    install_requires = requirements_file.read().splitlines(False)

tests_require = [
]

setup_requires = [
]

setup(
    name='gbqa',
    version=version,
    description="Graph Base Question Answering - Text Mining",
    author="NiceGuys",
    url='https://github.com/haluwen9/gbqa',
    packages=[
        'gbqa',
    ],
    package_dir={'gbqa': 'gbqa'},
    entry_points={
        'console_scripts': [
            'gbqa=gbqa.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=install_requires,
    license="GNU General Public License v3",
    zip_safe=False,
    keywords=['gbqa', 'question answering'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    tests_require=tests_require,
    setup_requires=setup_requires
)