import pyspark
from operator import add
import json
from string import punctuation
import math


def build_ngram(x,length,identifier='::'):
    temp = list();
    x = " ".join(x.split())
    x = x.lower().split(' ')
    print(x)
    for i in range(0,len(x)-length):
        result = x[i:i+length]
        temp.append((identifier.join(result),1))

    return temp;

def reverse_ngram(x,identifier='::'):
    temp = x[0].split(identifier)
    return (temp,x[1])
       
       

sc = pyspark.SparkContext('local',appName="WordCount")
rdd1 = sc.wholeTextFiles("/Volumes/OSX-DataDrive/data-distributed/dataset/")
rdd2= rdd1.values()
rdd = rdd2.flatMap(lambda l: build_ngram(l.strip(),3)).reduceByKey(lambda a,b:a+b).map(lambda l: reverse_ngram(l)).sortByKey()
print(rdd.take(40))