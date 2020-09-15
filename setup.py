"""
https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
https://docs.pytest.org/en/latest/goodpractices.html
"""
from setuptools import setup, find_packages

requirements = [
    'torch',
    'pytorch_lightning',
    'numpy',
    'dgl',
    'spacy',
    'pandas',
    'networkx',
    'scikit-learn',
    'scikit-multilearn',
    'nltk',
    'editdistance',          # required for contextualspellcheck
    'transformers',          # required for contextualspellcheck
    'contextualSpellCheck',
    'word2number',
    'pycontractions',
    'unidecode',
]

extra_requirements = [
        'matplotlib'
]

VERSION = '0.0.1'

setup(
    name='text_gcn',
    version=VERSION,
    url='https://github.com/abhinavg97/GCN',
    description='Python module designed for text GCN',
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
