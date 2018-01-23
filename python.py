
from pyspark import *
from pyspark.sql import SparkSession
from pyspark import SparkContext,SQLContext
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import os;
import math
import json
import re

sc = SparkContext()
sqlContext = SQLContext(sc)

textFile=["X_train_vsmall.txt"]
testFile=["y_train_vsmall.txt"]
path = "/Volumes/OSX-DataDrive/data-distributed/dataset/training_set/"
path_test = "/Volumes/OSX-DataDrive/data-distributed/dataset/label_set/"
stopwords="/Volumes/OSX-DataDrive/data-distributed/stopwords.txts"

stopword_rdd = sc.textFile(stopwords)
stopword_list = stopword_rdd.map(lambda l:l.strip()).collect()
print(stopword_list)
stopwords = sc.broadcast(stopword_list)

def clean(x):
    x = x.strip()
    x = x.lower()
    x = x.replace('.',' ')
    x = x.replace('\\',' ')
    x = x.replace(':',' ')
    x = x.replace('/','')
    x = x.replace('*','')
   
    temp = ['?','!','.','/','’',']','[',',','[',']','@','^','{','}','%','*','#','?--']
    x = replace_all(x,temp)
    if len(x) <=1:
        return None;
    return x

def replace_all(x,dataset):
    for i in dataset:
        x = x.replace(i,'')
    return x
        
def exist(x,dataset):
    for i in dataset:
        if x.startswith(i) or x.endswith(i) or x ==i:
            return True;
    return False;

def round_score(x):
    if x==0:
        return 0
    else:
        return round(x,2)
    
def checkClass(text,class_label='CAT'):
    text = text.strip().split(',');
    return text;

def clean_stopword(x,stopwords):
    if x in stopwords.value:
        return None
    else:
        return x;

    
def build_dict(entries):
    temp = dict();
    for l in entries:
        temp[l[0]] = l[1]
    return temp

def save_dict(entries,filename=""):
    print(len(entries))
    dict_entry = build_dict(entries)
    json.dump(dict_entry,open(filename,"w"))




def max(x):
    if x>=1:
        return 1
    else:
        return 0

def divide_safely(x,y):
    if y == 0:
        return 0
    else:
        return x/y
def calculate_idf(x):
    
    idf_value =max(x[1])+max(x[2])+max(x[3])+max(x[4])+max(x[5])+max(x[6])+max(x[7]) + max(x[8])
    if idf_value !=0:
        idf_value = math.log(8/idf_value)
    else:
        idf_value=1
    return [x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],idf_value]

def calculate_tf_idf(x):
    return [x[0],x[1]*x[-1],x[2]*x[-1],x[3]*x[-1],x[4]*x[-1],x[5]*x[-1],x[6]*x[-1],x[7]*x[-1],x[8]*x[-1]]

def build_ngram(x,length,identifier='::'):
    temp = list();
    text = x[0]
    text = " ".join(text.split())
    text = text.lower().split(' ')
    for i in range(0,len(text)-length):
        result = text[i:i+length]
        temp.append((identifier.join(result),x[1]))

    return temp;

def reverse_ngram(x,identifier='::'):
    temp = x[0].split(identifier)
    return (temp,x[1])
       
def split_row(x):
    text = x[0]
    label = x[1].split(',')
    result  =[(text,i) for i in label ]
    return result;

    
def preprocess(fileName,colname):
    rdd = sc.textFile(fileName).zipWithIndex();
    #rdd = rdd.flatMap(lambda l: build_ngram(l.strip(),3)).reduceByKey(lambda a,b:a+b).map(lambda l: reverse_ngram(l)).sortByKey()
    df = sqlContext.createDataFrame(rdd,schema=[colname,'key']).distinct()
    return df;

#BUILD INDIVIUAL RDD FOR EACH DOCUMENT INORDER TO MERGET TO THE MASTER KEY SET
word_count = list();

df_1 = preprocess(path_test+testFile[0],"label")
df_2 = preprocess(path+textFile[0],"text")
text_table = df_1.join(df_2,df_1.key==df_2.key,"left").select('text','label').rdd.flatMap(lambda l: split_row(l));
text_table.take(40)
text_table = text_table.flatMap(lambda l:build_ngram(l,3)).map(lambda l:reverse_ngram(l));
df = sqlContext.createDataFrame(text_table,schema=['text','label'])
df.registerTempTable("dataset") #establish main table


                    











sqlContext.sql("Select * from dataset where label='ECAT'").registerTempTable("ECAT")
sqlContext.sql("Select * from dataset where label='MCAT'").registerTempTable("MCAT")
sqlContext.sql("Select * from dataset where label='GCAT'").registerTempTable("GCAT")
sqlContext.sql("Select * from dataset where label='GCAT'").registerTempTable("CCAT")
