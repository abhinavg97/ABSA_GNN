# Description

This folder contains the datasets used to evaluate our model.

Six datasets are used to evaluate our model.

* [FourSquared](https://europe.naverlabs.com/research/natural-language-processing/aspect-based-sentiment-analysis-dataset/)
* [MAMS ACSA](https://github.com/siat-nlp/MAMS-for-ABSA)
* [MAMS ATSA](https://github.com/siat-nlp/MAMS-for-ABSA)
* [Samsung Galaxy](https://github.com/epochx/opinatt)
* [SemEval 2014](http://alt.qcri.org/semeval2014/task4/)
* [SemEval 2016](http://alt.qcri.org/semeval2016/task5/)

All the datasets are cleaned by using the text processing pipeline as mentioned in the paper. The description of the pipeline is given in the utils folder of acsa_gnn module in this repository as well.

The cleaned data is stored in the data folder of this repository. The format of the data is [text labels].

Text contains the cleaned text from the datasets mentioned above, labels contain a custom one hot vector as described in the paper.

## Files present

* train.xml

        The raw dataset.

* {dataset}_dataframe.csv

        This file contains the dataframe generated after cleaning the raw data from train.xml.

        The format of the csv is as follows:

        ```
        id,text,labels
        ```

        labels are encoded in a custom one hot vector format.

        -2 -> the label is not present in the sample.
        -1 -> the label has a negative sentiment associated with the sample.
        0  -> the label has a neutral sentiment associated with the sample.
        1  -> the label has a positive sentiment associated with the sample.
        2  -> the label has a ambiguous sentiment associated with the sample.

* {dataset}_label_text_to_label_id.json

        The position of the custom one hot vector represents a label. This position to label mapping is present in this file.

* {dataset}_bag_of_words.csv 

        This file contains the frequencies of the labels in the dataset.

* {dataset}_train_graph.bin

        The saved DGL graph created from the above dataframe. Please use this to fasten your training time.
