import findspark
findspark.init()
import pyspark
from operator import add
import json
from string import punctuation
import math

sc = pyspark.SparkContext('local',appName="WordCount")
rdd1 = sc.wholeTextFiles("/home/vyom/UGA/DSP/P1trial/data/")
rdd2= rdd1.values()
splt = rdd2.flatMap(lambda word: word.split())
splt = splt.map(lambda word: word.lower())
word = splt.map(lambda word : (word,1))
reduced = word.reduceByKey(add)
filtered = reduced.filter(lambda x: x[1] > 2)
filtered = filtered.sortBy(lambda x: x[1],False)
print(filtered.take(5))
