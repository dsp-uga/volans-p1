
# coding: utf-8

# In[1]:


import pyspark
from operator import add, itemgetter
import json
from string import punctuation
import math
from pyspark.conf import SparkConf
from string import punctuation
#from nltk.stem import LancasterStemmer
import sys

# # Calculating count of words in all docs combined

# In[2]:


sc = pyspark.SparkContext('local[*]',appName="DocClassification")


# In[3]:


documents = sc.textFile(str(sys.argv[1]))


# In[4]:


labels = sc.textFile(str(sys.argv[2]))


# In[6]:


splt = documents.flatMap(lambda word: word.split())
splt = splt.map(lambda word: word.lower())

#Removing Punctuation
splt = splt.map(lambda word: word.replace("&quot;",""))
splt = splt.map(lambda word: word.replace("&amp;",""))
cleanWords = splt.map(lambda word: word.strip(punctuation))
cleanWords = cleanWords.filter(lambda word:len(word)>2)

# In[7]:


word = cleanWords.map(lambda word : (word,1))


# In[8]:


CountinAllDocs = word.reduceByKey(add)


# In[9]:




# In[10]:


numDocs = sc.broadcast(documents.count())
wordList = sc.broadcast(CountinAllDocs.keys().collect())

# # Calculating term frequency in each doc

# In[11]:


def stripWord(x):
    tempList =[]
    for word in x:
        word = word.lower()
        word = word.replace("&quot","")
        word = word.replace("&amp","")
        word = word.strip(punctuation)
        if len(word)>2:
            tempList.append(word)
    return tempList



# In[13]:

def countWord(x):
    dictionary={}
    for word in x:
        if word not in dictionary:
            dictionary[word] = 1
        else:
            dictionary[word]+=1
    return dictionary


# In[14]:


bagOfWords = documents.map(lambda word: word.split())
bagOfWords = bagOfWords.map(lambda x: stripWord(x))
#bagOfWords = bagOfWords.map(lambda x: stripStopWord(x))
tf = bagOfWords.map(lambda x: countWord(x))


# # Calculating idf

# In[15]:


def uniques(x):
    tempList=[]
    for word in x:
        if word not in tempList:
            tempList.append(word);
    return tempList


# In[16]:


uniqueList = tf.map(lambda x: uniques(x))


# In[17]:


def initializeMap(x):
    tempList = []
    for word in x:
        tempList.append((word,1))
    return tempList


# In[18]:


occurences = uniqueList.map(lambda x: initializeMap(x)).flatMap(lambda x: x).reduceByKey(add)


# In[19]:


idf = occurences.map(lambda x: (x[0],math.log(numDocs.value/float(x[1]))))


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


# In[23]:


tfidfSorted = tfidf.map(lambda x: sorted(x.items(), key=itemgetter(1), reverse = True))


# # Implementing Naive Bayes

# ## Calculating word probability

# In[39]:


spltLabels = labels.map(lambda word: word.split(","))


# In[40]:


def removeUnnecessary(label):
    tempList=[]
    for word in label:
        if 'CAT' in word:
            tempList.append(word)
    return tempList


# In[41]:


requiredLabels = spltLabels.map(lambda label: removeUnnecessary(label))


# In[42]:


numberOfDocs = sc.broadcast(requiredLabels.flatMap(lambda x: x).count())


# In[43]:


labelsToUse = sc.broadcast(requiredLabels.collect())


# In[44]:


def dictToList(x):
    tempList=[]
    for k, v in x.items():
        tempList.append((k,v))
    return tempList


# In[45]:


def getProbDoc(x,v):
    tempDict={}
    for label in labelsToUse.value[v]:
        tempDict[label] = x
    return tempDict


# In[56]:


ProbDocIndex = tfidfSorted.zipWithIndex()


# In[34]:


def totalTfidf(x):
    total =0
    for k,v in x.items():
        total = total+v
    return total


# In[59]:


ProbDoc = ProbDocIndex.map(lambda x: getProbDoc(x[0],x[1]))


# In[66]:


ProbDocList= ProbDoc.map(lambda x : dictToList(x))


# In[67]:


def tfidfListToDict(x):
    tempDict={}
    for word in x:
        tempDict[word[0]] = word[1]
    return tempDict


# In[68]:


ProbDocList = ProbDocList.flatMap(lambda x: x).reduceByKey(add)
ProbDocDict= ProbDocList.map(lambda x: (x[0],tfidfListToDict(x[1])))


# In[97]:


ProbDocDictSorted = ProbDocDict.map(lambda x: (x[0],sorted(x[1].items(), key=itemgetter(1), reverse = False)))


# In[99]:


def getOurStopwords(x):
    tempList=[]
    for word in x:
        if(word[1]<3.0):
            tempList.append(word[0])
    return tempList


# In[100]:


ourStopwords = ProbDocDictSorted.map(lambda x: getOurStopwords(x[1]))


# In[111]:


toDownload = ourStopwords.flatMap(lambda x: x).distinct()
toDownload.saveAsTextFile("gs://dspp1/output")
# In[118]:


#rd = toDownload.collect()


# In[117]:

"""
result = open('ourStopwords.txt', 'w')
for td in rd:
    result.write("%s\n" % td)
"""
