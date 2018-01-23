
# coding: utf-8

# In[16]:


from pyspark import *
from pyspark.sql import SparkSession
from pyspark import SparkContext,SQLContext
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import numpy as np
import os;
import math
import json
import re


# In[17]:


sc = SparkContext()
sqlContext = SQLContext(sc)

split_size = 10


# In[18]:


textFile=["X_train_vsmall.txt"]
testFile=["y_train_vsmall.txt"]
path = "/Volumes/OSX-DataDrive/data-distributed/dataset/training_set/"
path_test = "/Volumes/OSX-DataDrive/data-distributed/dataset/label_set/"
stopwords="/Volumes/OSX-DataDrive/data-distributed/stopwords.txts"


# In[19]:


stopword_rdd = sc.textFile(stopwords)
stopword_list = stopword_rdd.map(lambda l:l.strip()).collect()
print(stopword_list)
stopwords = sc.broadcast(stopword_list)


# In[24]:


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
    text = text.split(' ');
    result = None
    chunk_size = len(text)%split_size
    chunks = np.array_split(text,5)
    temp = list(chunks)
    for i in temp:
        query = build_query(i,name)
        if result  is None:
            result = sqlContext.sql(query).select('text',name)
        else:
            result.union( sqlContext.sql(query).select('text',name))
    return result;
    
        

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
    df = sqlContext.createDataFrame(rdd,schema=[colname,'key']).distinct()
    return df;


# In[25]:


#BUILD INDIVIUAL RDD FOR EACH DOCUMENT INORDER TO MERGET TO THE MASTER KEY SET
word_count = list();

df_1 = preprocess(path_test+testFile[0],"label")
df_2 = preprocess(path+textFile[0],"text")
text_table = df_1.join(df_2,df_1.key==df_2.key,"left").select('text','label').rdd.flatMap(lambda l: split_row(l)).flatMap(lambda l:split_word(l));
text_table = text_table.map(lambda l:((l[0],l[1]),l[2])).reduceByKey(lambda a,b:a+b).map(lambda l:(str(l[0][0]),l[0][1],l[1]))
print(text_table.take(10))
#text_table = text_table.flatMap(lambda l:build_ngram(l,3)).reduceByKey(lambda a,b:a+b).map(lambda l:reverse_ngram(l))
#text_table =text_table.flatMap(lambda l:(l[0][0],l[0][1]),l[0][2]).reduceByKey(lambda a,b:a+b)
#print(text_table.take(10))
df = sqlContext.createDataFrame(text_table,schema=['text','label','count'])
df.registerTempTable("dataset") #establish main table


                    











# In[26]:


ecat = sqlContext.sql("Select * from dataset where label='ecat'")
mcat = sqlContext.sql("Select * from dataset where label='mcat'")
gcat = sqlContext.sql("Select * from dataset where label='gcat'")
ccat = sqlContext.sql("Select * from dataset where label='ccat'")

ccat_count = ccat.count();
mcat_count = mcat.count();
ecat_count = ecat.count();
gcat_count = gcat.count();
total = ccat_count + mcat_count + ecat_count + gcat_count


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


# In[31]:


query = split_query('While you have been writing strings, you still do not know what they do. In this exercise we create a bunch of variables with complex strings so you can see what they are for. First an explanation of strings.A string is usually a bit of text you want to display to someone, or "export" out of the program you are writing. Python knows you want something to be a string when you put either  (double-quotes) or (single-quotes) around the text. You saw this many times with your use of print when you put the text you want to go inside the string inside  or  after the print to print the string.Strings may contain the format characters you have discovered so far. You simply put the formatted variables in the string, and then a % (percent) character, followed by the variable. The only catch is that if you want multiple formats in your string to print multiple variables, you need to put them inside ( ) (parenthesis) separated by , (commas). I s as if you were telling me to buy you a list of items from the store and you said, "I want milk, eggs, bread, and soup. Only as a programmer we say, (milk, eggs, bread, soup).We will now type in a whole bunch of strings, variables, and formats, and print them. You will also practice using short abbreviated variable names. Programmers love saving time at your expense by using annoyingly short and cryptic variable names, so lets get you started reading and writing them early on.','ecat')



print(query.show())


