
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



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-X_train", help="X_training set")
parser.add_argument("-X_test", help="X testing  set")
parser.add_argument("-Y_train", help="Y training set")
parser.add_argument("-stopwords",help="Stopwords file")
args = parser.parse_args()
X_train = args.X_train
X_test = args.X_test
Y_train = args.Y_train
stopwords = args.stopwords
# In[102]:


stopword_rdd = sc.textFile(stopwords)
stopword_list = stopword_rdd.map(lambda l:l.strip()).collect()
print(stopword_list)
stopwords = sc.broadcast(stopword_list)


# In[130]:


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
    return math.log(float(x/count)*1000)
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


def naive_bayes(text):
    ecat = calculate_class_prob(text,'ecat')
    mcat = calculate_class_prob(text,'mcat')
    ccat = calculate_class_prob(text,'ccat')
    gcat = calculate_class_prob(text,'gcat')
    if ecat > mcat and ecat > ccat and ecat > gcat:
        return "ECAT"
    elif mcat > ecat and mcat > ccat and mcat > gcat:
        return "MCAT"
    elif ccat > mcat and ccat > ecat and ccat > gcat:
        return "CCAT"
    elif gcat > mcat and gcat > ccat and gcat > ecat:
        return "GCAT"

def calculate_class_prob(text,name):
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
    try:
        result = reduce(lambda x, y: x + y, result)
    except:
        return 0.2
    
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
    text = list(set(text.split(' ')));
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

#https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
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


# In[131]:

#BUILD INDIVIUAL RDD FOR EACH DOCUMENT INORDER TO MERGET TO THE MASTER KEY SET
word_count = list();

df_1 = preprocess(Y_train,"label")
df_2 = preprocess(X_train,"text")
text_table = df_1.join(df_2,df_1.key==df_2.key,"left").select('text','label').rdd.flatMap(lambda l: split_row(l)).flatMap(lambda l:split_word(l));
text_table = text_table.map(lambda l:((l[0],l[1]),l[2])).reduceByKey(lambda a,b:a+b).map(lambda l:(str(l[0][0]),l[0][1],l[1]))
#text_table = text_table.flatMap(lambda l:build_ngram(l,3)).reduceByKey(lambda a,b:a+b).map(lambda l:reverse_ngram(l))
#text_table =text_table.flatMap(lambda l:(l[0][0],l[0][1]),l[0][2]).reduceByKey(lambda a,b:a+b)
#print(text_table.take(10))
df = sqlContext.createDataFrame(text_table,schema=['text','label','count'])
df.registerTempTable("dataset") #establish main table


                    











# In[135]:


ecat = sqlContext.sql("Select * from dataset where label='ecat'")
mcat = sqlContext.sql("Select * from dataset where label='mcat'")
gcat = sqlContext.sql("Select * from dataset where label='gcat'")
ccat = sqlContext.sql("Select * from dataset where label='ccat'")

ccat_count = ccat.count()*1000;
mcat_count = mcat.count()*1000;
ecat_count = ecat.count()*1000;
gcat_count = gcat.count()*1000;
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
prob_ecat = calculate_prob(ecat,"ecat",ecat_count)
prob_ecat = prob_ecat.drop('label')
prob_ecat.registerTempTable('ECAT')
prob_ecat.cache()


# In[136]:


text = "A dedicated &quot;snow desk&quot; has been set up by the New York and New Jersey Port Authority to monitor and react to harsh weather conditions and help prevent disruption to travellers and cargo moving through key airports this winter. The authority operates New York's John F Kennedy, LaGuardia and Newark airports and carefully tracks weather patterns all year round. Each airport supplements National Weather Service reports with facility-specific forecasts from private companies that are updated a few times a day. &quot;We don't sit and wait for the weather to hit us&quot; said the Port authority chief operations officer David Feeley. &quot;We use the latest technology to anticipate what's coming day's in advance, which allows up to plan for deployment of employees and equipment at each facility.&quot; Each airport has a &quot;snow desk&quot; at which key operations and maintenance personnel analyse the weather reports and deploy staff and equipment accordingly. Each has inground sensors transmitting data such as windspeed and direction, dewpoint, humidity, air and ground temperatures. More than 5,100 tons of salt and sand, special de-icing equipment and 250 pieces of dedicated snow-fighting equipment - including massive snow-melters and snow blowers - is on standby to counter almost any winter blast at JFK, LaGuadria and Newark airports this year. Air Cargo Newsroom Tel+44 171 542 7706 Fax+44 171 542 5017"
print(naive_bayes(text))


f = open(X_test)
f_output = open("output.txt","w")

count = 0
for i in f:
    count = count + 1
    print(count)
    temp = naive_bayes(i)
    f_output.write(temp +os.linesep)
    f_output.flush();
