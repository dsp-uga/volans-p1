
# coding: utf-8

# In[99]:


from pyspark import *
from pyspark.sql import *
from pyspark import SparkContext,SQLContext
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import *
from pyspark.sql.types import *
from functools import reduce

from pyspark.sql.functions import udf
import numpy as np
import os;
import math
import json
import re


# In[100]:


sc = SparkContext()
sqlContext = SQLContext(sc)

split_size = 10


# In[101]:


textFile=["X_train_vsmall.txt"]
testFile=["y_train_vsmall.txt"]
path = "/Volumes/OSX-DataDrive/data-distributed/dataset/training_set/"
path_test = "/Volumes/OSX-DataDrive/data-distributed/dataset/label_set/"
stopwords="/Volumes/OSX-DataDrive/data-distributed/stopwords.txts"


# In[102]:


stopword_rdd = sc.textFile(stopwords)
stopword_list = stopword_rdd.map(lambda l:l.strip()).collect()
print(stopword_list)
stopwords = sc.broadcast(stopword_list)


# In[119]:


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


def calculate_probability(x,count):
    return float(x/count)
def calculate_prob(df,label,count):
    udf_prob = udf(lambda l:calculate_probability(l,count),FloatType())
    df = df.withColumn(label, udf_prob("count"))
    return df;

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


def naive_bayes(text,total_prob):
    ecat = calculate_class_prob(text,'ecat',final_prob)
    mcat = calculate_class_prob(text,'mcat',final_prob)
    ccat = calculate_class_prob(text,'ccat',final_prob)
    gcat = calculate_class_prob(text,'gcat',final_prob)
    if ecat > mcat and ecat > ccat and ecat > gcat:
        return 0
    elif mcat > ecat and mcat > ccat and mcat > gcat:
        return 1
    elif ccat > mcat and ccat > ecat and ccat > gcat:
        return 2
    elif gcat > mcat and gcat > ccat and gcat > ecat:
        return 3

def calculate_class_prob(text,name,total_prob):
    query = split_query(text,name)
    result = query.collect()
    if name =='ecat':
        result = [i.ecat for i in result]
    elif name=='ccat':
        result = [i.ccat for i in result]
    elif name =='gcat':
        result = [i.gcat for i in result]
    elif name =='mcat':
        result = [i.mcat for i in result]
        
    result = reduce(lambda x, y: x*y, result) * total_prob
    
    return result

def build_ngram(x,length,identifier='::'):
    temp = list();
    text = x[0]
    text = " ".join(text.split())
    text = text.lower().split(' ')
    for i in range(0,len(text)-length):
        result = text[i:i+length]
        result.sort()
        temp.append(((identifier.join(result),x[1]),1))

    return temp;

def split_query(text,name,split_size=10):
    text = text.replace("'",'');
    text = text.replace('"','');
    text = text.split(' ');
    result = None
   
    slots = int(len(text)/split_size)
   
    chunks = np.array_split(text,slots)
    temp = list(chunks)
    df_list = list();
    for i in temp:
        query = build_query(np.array(i).tolist(),name)
        df = sqlContext.sql(query)
        df_list.append(df.rdd)
    result = sc.union(df_list)
    return result;

#taken from stackoverflow

def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)

def build_query(data,name):
    

    query = "Select * from " + name +  " where text='"+data[0]+"'"
    for i in range(1,len(data)):
        query = query + " or text='"+data[i]+"'" 
    print(query)
    return query

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        

        
def reverse_ngram(x,identifier='::'):
    temp = x[0][0].split(identifier)
    return (','.join(temp),x[0][1],x[1])
       
def split_row(x):
    text = x[0]
    label = x[1].split(',')
    result  =[(text,i) for i in label ]
    return result;

def split_word(x):
    label = x[1]
    text = x[0]
    word_list = text.split()
    result = [(i.strip(),label,1) for i in word_list]
    return result;
def preprocess(fileName,colname):
    rdd = sc.textFile(fileName).map(lambda l:l.lower()).zipWithIndex();
    #rdd = rdd.flatMap(lambda l: build_ngram(l.strip(),3)).reduceByKey(lambda a,b:a+b).map(lambda l: reverse_ngram(l)).sortByKey()
    df = sqlContext.createDataFrame(rdd,schema=[colname,'key'])
    return df;


# In[116]:


#BUILD INDIVIUAL RDD FOR EACH DOCUMENT INORDER TO MERGET TO THE MASTER KEY SET
word_count = list();

df_1 = preprocess(path_test+testFile[0],"label")
df_2 = preprocess(path+textFile[0],"text")
text_table = df_1.join(df_2,df_1.key==df_2.key,"left").select('text','label').rdd.flatMap(lambda l: split_row(l)).flatMap(lambda l:split_word(l));
text_table = text_table.map(lambda l:((l[0],l[1]),l[2])).reduceByKey(lambda a,b:a+b).map(lambda l:(str(l[0][0]),l[0][1],l[1]))
#text_table = text_table.flatMap(lambda l:build_ngram(l,3)).reduceByKey(lambda a,b:a+b).map(lambda l:reverse_ngram(l))
#text_table =text_table.flatMap(lambda l:(l[0][0],l[0][1]),l[0][2]).reduceByKey(lambda a,b:a+b)
#print(text_table.take(10))
df = sqlContext.createDataFrame(text_table,schema=['text','label','count'])
df.registerTempTable("dataset") #establish main table


                    











# In[117]:


ecat = sqlContext.sql("Select * from dataset where label='ecat'")
mcat = sqlContext.sql("Select * from dataset where label='mcat'")
gcat = sqlContext.sql("Select * from dataset where label='gcat'")
ccat = sqlContext.sql("Select * from dataset where label='ccat'")

ccat_count = ccat.count();
mcat_count = mcat.count();
ecat_count = ecat.count();
gcat_count = gcat.count();
total = ccat_count + mcat_count + ecat_count + gcat_count
final_prob = (ccat_count/float(total))*(mcat_count/float(total))*(ecat_count/float(total))*(gcat_count/float(total))


prob_ccat = calculate_prob(ccat,"ccat",ccat_count)
prob_ccat = calculate_prob(prob_ccat,"mcat",mcat_count)
prob_ccat = calculate_prob(prob_ccat,"gcat",gcat_count)
prob_ccat = calculate_prob(prob_ccat,"ecat",ecat_count)
prob_ccat = prob_ccat.drop('label')
prob_ccat.registerTempTable('CCAT')
prob_ccat.cache()

prob_mcat = calculate_prob(mcat,"ccat",ccat_count)
prob_mcat = calculate_prob(prob_mcat,"mcat",mcat_count)
prob_mcat = calculate_prob(prob_mcat,"gcat",gcat_count)
prob_mcat = calculate_prob(prob_mcat,"ecat",ecat_count)
prob_mcat = prob_mcat.drop('label')
prob_mcat.registerTempTable('MCAT')
prob_mcat.cache()

prob_gcat = calculate_prob(gcat,"ccat",ccat_count)
prob_gcat = calculate_prob(prob_gcat,"mcat",mcat_count)
prob_gcat = calculate_prob(prob_gcat,"gcat",gcat_count)
prob_gcat = calculate_prob(prob_gcat,"ecat",ecat_count)
prob_gcat = prob_gcat.drop('label')
prob_gcat.registerTempTable('GCAT')
prob_gcat.cache()

prob_ecat = calculate_prob(ecat,"ccat",ccat_count)
prob_ecat = calculate_prob(prob_ecat,"mcat",mcat_count)
prob_ecat = calculate_prob(prob_ecat,"gcat",gcat_count)
prob_ecat = calculate_prob(prob_ecat,"ecat",ecat_count)
prob_ecat = prob_ecat.drop('label')
prob_ecat.registerTempTable('ECAT')
prob_ecat.cache()
prob_ecat.show()


# In[120]:



text = 'if you want multiple formats in your string to print multiple variables, you need to pu s that if you want multiple formats in your string to print multiple variables, you need to pu'
print(naive_bayes(text,final_prob))




