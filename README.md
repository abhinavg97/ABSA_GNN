# [Aspect Based Sentiment Analysis with GCN](https://github.com/abhinavg97/GCN)



This project is under works. It aims to do sentiment analysis using text GCN.

## Currently done:

*   Identified aspects terms from user opinions.
*   Dependency parsing is used to capture syntactical structure.
*   Graph Convolutional Network is used to capture dependencies of aspect and opinions.
*   Stratified split is used to ensure even distribution of aspect classes among train, validation and test data
*   For predicting the aspect terms, MultilabelClassification from the simpletransformers library is used as the baseline.

## Backlog:

1.  Update Adjacency matrix at each iteration
2.  Correct metrics calculation

## Folder Structure

Browse into the corresponding folders in the text_gcn module to see the pertaining details

## Setup

### Prerequisites

- Python3

#### Install the virtual Environment for python
```bash
$ sudo apt install python3-venv
```

#### Install the Java dependancy
```bash
$ sudo apt install openjdk-8-jre-headless
```

In case pip install gives wheel related errors:
```bash
$ sudo update-alternatives --set java /usr/lib/jvm java-8-openjdk-amd64/jre/bin/java 
```
This is required for language check package

#### Clone the repository
```bash
$ git clone https://github.com/abhinavg97/GCN.git
$ cd gcn
```

#### Create and Activate the virtual environment
```bash
$ python -m venv venv
$ source venv/bin/activate
```

#### Install
```bash
$ pip install -r requirements.txt
```

#### Install the spacy Dependancies
```bash
$ sudo apt install python3-tk
$ python -m spacy download en_core_web_lg 
$ python -m spacy download en_core_web_sm
```

## Using the scripts

```bash
$ python DGL_graph_handler.py
```

## Logging

Logging is done by PyTorch lightning which uses Tensorboard by default.

Visualize the metrics:

```bash
$ tensorboard --logdir lightning_logs/
```
or
```bash
$ python3 -m tensorboard.main --logdir lightning_logs/
```

## Containerize the application

```bash
$ docker image build -t image_name:tag .
$ docker container run --name text_gcn --mount source=volume_name,target=/usr/src/app image_name:tag
```

The mounted directory is present at /var/lib/docker/volumes/

Note: You need sudo permissions to access the above directory
