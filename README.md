# [Text GCN](https://github.com/abhinavg97/GCN)

This project is under works. It aims to do sentiment analysis using text GCN.

## Backlog:

1. Create GCN

## Setup

### Prerequisites

- Python3

#### Install the virtual Environment for python
```
sudo apt install python3-venv
```

#### Install the Java dependancy
```
sudo apt install openjdk-8-jre-headless
```
This is required for language check package

#### Install the spacy Dependancies
```
sudo apt install python3-tk
python - m spacy download en_core_web_lg en_core_web_sm
```

#### Clone the repository
```bash
$ git clone https://github.com/abhinavg97/GCN.git
$ cd gcn
```

#### Activate the virtual environment
```
source venv/bin/activate
```

#### Install
To visualize the DGL graph
```bash
$ pip install -e .[interactive]
```
For inferring results
```bash
$ pip install -e .
```

## Using the scripts

A description of the options available can be found using:

```bash
$ python gcn.py --help
```

For example:

```bash
$ python gcn.py --data-file data/SemEval16_gold_Laptops/sample.txt --dataset_name SemEval
```