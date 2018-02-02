# Document Classification
## Team: volans
#### Members: 
* Ankita Joshi
* Prajay Shetty
* Vyom Shrivastava

Classifier : Naive - Bayes

## Technologies Used:
* Apache Spark
* Scala
* pySpark
* Python 2.7

## Overview:
This repository contains implementaion of a Naive Bayes Classifier build using Apache Spark to classify documents into corrosponding classes. This is done as a Project for the course CSCI 8360: Data Science Practicum.

## Dataset:
The datasets we used to train the classifier is provided by Dr. Shannon Quinn for the course CSCI 8360: Data Science Practicum. We used 3 datasets to to train and tests our model. Each documents belonged to one or more classes
1. Very small dataset with 68 documents to train and 8 documents to test prediction accuracy.
2. Small dataset with 7356 documents to train and 818 documents to test prediction accuracy.
3. Large dataset with 723988 documents to train  and 80435 documents to test prediction accuracy.

## Execution Steps:
The project uses Apache Spark to run. Instructions to download and install Apache Spark can be found [here](https://spark.apache.org/downloads.html).

Navigate to the project folder run the following command to execute and run the classifier program:
```sbt package```
    
```spark-submit Naive-Bayes.scala /path/to/X_train_file /path/to/y_train_file/ /path/to/stopwords_file```


## Pre-processing: 
