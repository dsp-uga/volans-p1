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
`sbt package`
    
`spark-submit Naive-Bayes.scala /path/to/X_train_file.txt /path/to/y_train_file.txt /path/to/stopwords_file.txt`

## Pre-processing: 
* Each document is converted into word vector
* Removed unnecessary strings like &amp and &quot
* Removed punctuation marks
* Removed Stopwords
* Removed words of length less than 3

### Tf-idf to Remove Stopword:
* Calculated the tf-idf value of each word of a document and grouped the words according the their occurence in each class.
* Created our list own list of stopwords which included words of each class with tf-idf value less than 3.
* Using these stopwords improved the accuracy drastically.

## Flow
* Loaded documents and labels 
* Pre-processed the data
* Calculated word count of all unique words in corpus.
* Calculated tfidf of each word in a document and then grouped the word-tfidf pair according to the class the word belongs to
* Calculated term frequency for each word in a class.
* Performed smoothing of data to account for word which are in corpus but not in specific class. 
* Calculated Prior probabilities (Word and Document probabilities)
* Loaded Test Data
* Pre-processed the data similar to the train data.
* Performed smoothing for unseen words.
* Applied Naive-Bayes formula on test document words using the prior probabilities calculated for the words during training.
* Class with highest probability value is assigned the class.

## Accuracy
Following are the best accuracies we got on each dataset:
* Very small: 87.5 %
* small: 92.5 %
* large: 95.06 % 

## Other approaches we tried
* Incorporated tf-idf as the word feature instead of word counts in the prior probabilities and Naive-Bayes formula -- Accuracy was decreased.
* Merged all training datasets into one (very small, small and large) -- no significant effect.
* Not considering unseen words for probability calculation (smoothing during testing) -- increased accuracy slightly for small dataset but no significant effect on large dataset.

## Stuff we planned to try
* Using n-grams instead on 1 word.
* Using combination of multiple binary naive-bayes models to predict confidence level for each class and picking the one class as prediction with highest confidence value
