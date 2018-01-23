import pyspark
from operator import add, itemgetter
import json
from string import punctuation
import math
from pyspark.conf import SparkConf
from string import punctuation


# # Calculating count of words in all docs combined

# In[2]:

sc = pyspark.SparkContext('local',appName="DocClassification")


# In[3]:


documents = sc.textFile("/home/vyom/UGA/DSP/Project1/X_train_vsmall.txt")


# In[4]:


labels = sc.textFile("/home/vyom/UGA/DSP/Project1/y_train_vsmall.txt")


# In[5]:


splt = documents.flatMap(lambda word: word.split())
splt = splt.map(lambda word: word.lower())

#Removing Punctuation
splt = splt.map(lambda word: word.replace("&quot",""))
cleanWords = splt.map(lambda word: word.strip(punctuation))
cleanWords = cleanWords.filter(lambda word:len(word)>0)

#Removing StopWord
stopWordFile = sc.textFile("/home/vyom/UGA/DSP/Project1/stopwords.txt")
stopWord = sc.broadcast(stopWordFile.collect())
lessWords = cleanWords.filter(lambda x: x not in stopWord.value)


# In[6]:


word = lessWords.map(lambda word : (word,1))


# In[7]:


CountinAllDocs = word.reduceByKey(add)


# In[8]:


CountinAllDocs = CountinAllDocs.sortBy(lambda x: x[1],False)


# # Calculating term frequency in each doc

# In[10]:


def stripWord(x):
    tempList =[]
    for word in x:
        word = word.replace("&quot","")
        word = word.strip(punctuation)
        if len(word)>0:
            tempList.append(word)
    return tempList


# In[11]:


def stripStopWord(x):
    tempList =[]
    for word in x:
        if word not in stopWord.value:
            tempList.append(word)
    return tempList


# In[12]:


def countWord(x):
    dictionary={}
    for word in x:
        if word not in dictionary:
            dictionary[word] = 1
        else:
            dictionary[word]+=1
    return dictionary


# In[46]:


bagOfWords = documents.map(lambda word: word.lower().split())
bagOfWords = bagOfWords.map(lambda x: stripWord(x))
bagOfWords = bagOfWords.map(lambda x: stripStopWord(x))
tf = bagOfWords.map(lambda x: countWord(x))


# # Calculating idf

# In[14]:


def uniques(x):
    tempList=[]
    for word in x:
        if word not in tempList:
            tempList.append(word);
    return tempList


# In[15]:


uniqueList = tf.map(lambda x: uniques(x))


# In[16]:


def initializeMap(x):
    tempList = []
    for word in x:
        tempList.append((word,1))
    return tempList


# In[17]:


occurences = sc.parallelize(uniqueList.map(lambda x: initializeMap(x)).reduce(add))
occurences = occurences.reduceByKey(add)

# In[18]:


NumDocs = documents.count()


# In[19]:


idf = occurences.map(lambda x: (x[0],math.log(NumDocs/x[1])))


# In[20]:


idf2 = sc.broadcast(idf.collectAsMap())


# # Calculating Tf-idf

# In[21]:


def getTfidf(x):
    tempDict={}
    idfDict=idf2.value
    for k,v in x.items():
        tempDict[k] = v*idfDict[k]
    return tempDict


# In[22]:


tfidf = tf.map(lambda x: getTfidf(x) )


# In[32]:


tfidfSorted = tfidf.map(lambda x: sorted(x.items(), key=itemgetter(1), reverse = True))
print(tfidfSorted.collect()[0][0:10])

