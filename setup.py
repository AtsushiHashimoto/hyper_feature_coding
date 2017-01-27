#!/usr/bin/env python
# coding: utf-8
from setuptools import setup, find_packages, Extension
from hyper_feature_coding import __author__, __version__, __license__
 
setup(
        name             = 'hyper_feature_coding',
        version          = __version__,
        description      = '.',
        license          = __license__,
        author           = __author__,
        author_email     = 'ahasimoto@mm.media.kyoto-u.ac.jp',
        url              = 'https://github.com/AtsushiHashimoto/hyper_feature_coding.git',
        keywords         = 'hierarchical clustering, hyper feature coding, histogram',
        packages         = find_packages(),
	include_package_data = True,
        install_requires = ['numpy','sklearn'],
        )
 
