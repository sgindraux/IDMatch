# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='IDMatch',
    version='1.1.0',
    description='Image and DSM matching software',
    long_description=readme,
    author='Saskia Gindraux',
    author_email='saskia.gindraux@gmail.ch',
    url='https://github.com/sgindraux/idmatch',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

