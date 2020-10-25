# [Aspect Based Sentiment Analysis with GCN](https://github.com/abhinavg97/ABSA_GNN)



This project is under works. It aims to do sentiment analysis using text GCN.

## Currently done:

*   Identified aspects terms from user opinions.
*   Dependency parsing is used to capture syntactical structure.
*   Graph Convolutional Network is used to capture dependencies of aspect and opinions.
*   Stratified split is used to ensure even distribution of aspect classes among train, validation and test data.
*   For predicting the aspect terms, MultilabelClassification from the simpletransformers library is used as the baseline.

## Backlog:

1.  Connect Updating Adjacency matrix code with the main pipeline

## Datasets

Six datasets are used to evaluate our model.

* [FourSquared](https://europe.naverlabs.com/research/natural-language-processing/aspect-based-sentiment-analysis-dataset/)
* [MAMS ACSA](https://github.com/siat-nlp/MAMS-for-ABSA)
* [MAMS ATSA](https://github.com/siat-nlp/MAMS-for-ABSA)
* [Samsung Galaxy](https://github.com/epochx/opinatt)
* [SemEval 2014](http://alt.qcri.org/semeval2014/task4/)
* [SemEval 2016](http://alt.qcri.org/semeval2016/task5/)

All the datasets are cleaned by using the text processing pipeline as mentioned in the paper. The description of the pipeline is given in the utils folder of absa_gnn module in this repository as well.

The cleaned data is stored in the data folder of this repository. The format of the data is [text labels].

Text contains the cleaned text from the datasets mentioned above, labels contain a multi hot vector as described in the paper.

For a detailed information about the files present in each dataset folder, please navigate to the data folder.

Please cite us if you find the the above cleaned datasets helpful in your work.


## Folder Structure

Browse into the corresponding folders in the absa_gnn module to see the pertaining details

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
$ sudo update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java 
```
This is required for language check package

#### Clone the repository
```bash
$ git clone https://github.com/abhinavg97/ABSA_GNN.git
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

Run our model

```bash
$ python main.py
```

Run baseline

```bash
$ python baseline.py
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
$ docker container run --name absa_gnn --mount source=volume_name,target=/usr/src/app image_name:tag
```

The mounted directory is present at /var/lib/docker/volumes/

Note: You need sudo permissions to access the above directory

## Citation

    @misc{
      author = {Gupta, Abhinav and Ghosh, Samujjwal and Konjengbam, Anand},
      title = {ABSA GNN},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/abhinavg97/ABSA_GNN}}
    }
