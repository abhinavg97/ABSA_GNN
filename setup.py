"""
https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
https://docs.pytest.org/en/latest/goodpractices.html
"""
import io
from os import path
from setuptools import setup, find_packages

requirements = [
    'torch==1.6.0',
    'pytorch_lightning==0.10.0',
    'numpy==1.19.1',
    'dgl==0.5.2',
    'spacy==2.3.2',
    'pandas==1.1.0',
    'networkx==2.4',
    'scikit-learn==0.23.2',
    'scikit-multilearn==0.2.0',
    'nltk==3.5',
    'editdistance==0.5.3',          # required for contextualspellcheck
    'contextualSpellCheck==0.2.0',
    'word2number==1.1',
    'pycontractions==2.0.1',
    'unidecode==1.1.1'
    'simpletransformers==0.48.6',
    'matplotlib'
]

extra_requirements = [
        'matplotlib'
]

VERSION = '0.0.1'

this_directory = path.abspath(path.dirname(__file__))
with io.open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    README = f.read()

setup(
    name='text_gcn',
    version=VERSION,
    url='https://github.com/abhinavg97/GCN',
    description='Python module designed for text GCN',
    long_description=README,
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
    extras_require={
        'interactive': extra_requirements
    },
    setup_requires=['pytest-runner', 'flake8'],
    tests_require=['pytest'],
    package_data={'': ['*.json']}
)

# run pip install -e .[interactive] to install the package
