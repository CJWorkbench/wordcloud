#!/usr/bin/env python

from setuptools import setup

setup(
    name='wordcloud',
    version='0.0.1',
    description='Show most-common words in a text column',
    author='Adam Hooper',
    author_email='adam@adamhooper.com',
    url='https://github.com/CJWorkbench/wordcloud',
    packages=[''],
    py_modules=['wordcloud'],
    install_requires=['pandas==0.23.4', 'nltk==3.3.0']
)