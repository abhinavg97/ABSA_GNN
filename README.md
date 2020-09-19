# [Aspect Based Sentiment Analysis with GCN](https://github.com/abhinavg97/GCN)



This project is under works. It aims to do sentiment analysis using text GCN.

## Currently done:

*   Identified aspects terms from user opinions.
*   Dependency parsing is used to capture syntactical structure.
*   Graph Convolutional Network is used to capture dependencies of aspect and opinions.
*   Stratified split is used to ensure even distribution of aspect classes among train, validation and test data

## Backlog:

1.  Update Adjacency matrix at each iteration

## Folder Structure

Browse into the corresponding folders in the text_gcn module to see the pertaining details

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
python -m spacy download en_core_web_lg en_core_web_sm
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

```bash
$ python DGL_graph_handler.py
```