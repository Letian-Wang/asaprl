""" Module setuptools script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
meta = {}
with open(os.path.join(here, 'asaprl', '__init__.py'), 'r') as f:
    exec(f.read(), meta)

description = """ASAP-RL"""

setup(
    name=meta['__TITLE__'],
    version=meta['__VERSION__'],
    description=meta['__DESCRIPTION__'],
    long_description=description,
    author=meta['__AUTHOR__'],
    license='Apache License, Version 2.0',
    keywords='DL RL AD Platform',
    packages=[
        *find_packages(include=('asaprl', 'asaprl.*')),
    ],
    python_requires=">=3.6",
    install_requires=[
        'ephem',
        'h5py',
        'imageio',
        'imgaug',
        'lmdb',
        'loguru==0.3.0',
        'networkx',
        'pandas',
        'py-trees==0.8.3',
        'pygame',
        'torchvision',
        'di-engine==0.2.3',
        'scikit-image',
        'setuptools==49.6.0',
        'shapely',
        'terminaltables',
        'tqdm',
        'xmlschema',
        'metadrive-simulator==0.2.4'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research/Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
