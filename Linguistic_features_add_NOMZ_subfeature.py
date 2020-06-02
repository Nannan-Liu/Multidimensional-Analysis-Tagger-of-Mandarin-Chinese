
# coding: utf-8

# In[2]:


#input your txt file folder
folder = '//'


# In[3]:


import os
import nltk 
from nltk.corpus import CategorizedPlaintextCorpusReader
corpus = CategorizedPlaintextCorpusReader(
    folder,
    r'(?!\.).*\.txt',
    cat_pattern=os.path.join(r'(neg|pos)', '.*',),
    encoding='utf-8')
corpus.words()


# In[17]:


#sort text files by their names, if necessary
import fileinput
import fnmatch

files=corpus.fileids()
sample_files=[]

for f in files: 
    #if fnmatch.fnmatch(f, 'sample.txt'):
        print (f)
        #sample_files.append(f)


# In[5]:


import pynlpir
pynlpir.open()


# In[7]:


corpora=[]
for file in files: 
    sub_corpora=corpus.raw(file)
    corpora.append(sub_corpora)


# In[8]:


#notice that here pos_names is set to `parent' because we need to add one sub_feature
tagged_files=[]
for sub_corpora in corpora: 
    tagged_file=pynlpir.segment(sub_corpora, pos_tagging=True, pos_names='parent')
    tagged_files.append(tagged_file)


# In[9]:


none_list=[]
for file in tagged_files: 
    none_list.append([s for s in file if None in s])


# In[10]:


print (none_list)


# In[11]:


#just in case there are new words that ICTCLAS does not recognise
for j in range(len(files)): 
    for n, i in enumerate(tagged_files[j]):
        if i == ('\r新华社', None):
            tagged_files[j][n] = ('\r新华社', 'noun-proper')
        if i == ('新华社', None):
            tagged_files[j][n] = ('新华社', 'noun-proper')
        if i == ('\r新华网', None):
            tagged_files[j][n] = ('\r新华网', 'noun-proper')
        if i == ('新华网', None):
            tagged_files[j][n] = ('新华网', 'noun-proper')
        if i == ('中新网', None):
            tagged_files[j][n] = ('中新网', 'noun-proper')
        if i == ('人民网', None):
            tagged_files[j][n] = ('人民网', 'noun-proper')
        if i == ('\r中国青年网', None):
            tagged_files[j][n] = ('\r中国青年网', 'noun-proper')
        if i == ('中评社', None):
            tagged_files[j][n] = ('中评社', 'noun-proper')
        if i == ('\r中国日报网', None):
            tagged_files[j][n] = ('\r中国日报网', 'noun-proper')
        if i == ('南华早报', None):
            tagged_files[j][n] = ('南华早报', 'noun-proper')
        if i == ('\r国际在线', None):
            tagged_files[j][n] = ('\r国际在线', 'noun-proper')
        if i == ('新华社', None):
            tagged_files[j][n] = ('新华社', 'noun-proper')
        if i == ('派', None): 
            tagged_files[j][n] = ('派', 'noun-verb')
        if i == ('网民', None): 
            tagged_files[j][n] = ('网民', 'noun')
        if i == ('屌丝', None):
            tagged_files[j][n] = ('屌丝', 'noun')
        if i == ('\r屌丝', None):
            tagged_files[j][n] = ('\r屌丝', 'noun')
        if i == ('富帅', None):
            tagged_files[j][n] = ('富帅', 'noun')
        if i == ('解构', None): 
            tagged_files[j][n] = ('解构', 'noun-verb')
        if i == ('身份卑微', None): 
            tagged_files[j][n] = ('身份卑微', 'adjective')
        if i == ('\r南方日报', None): 
            tagged_files[j][n] = ('\r南方日报', 'noun')
        if i == ('法新社', None):
            tagged_files[j][n] = ('法新社', 'noun-proper')
        if i == ('美联社', None):
            tagged_files[j][n] = ('美联社', 'noun-proper')
        if i == ('路透社', None):
            tagged_files[j][n] = ('路透社', 'noun-proper')
        if i == ('环球时报', None):
            tagged_files[j][n] = ('环球时报', 'noun-proper')
        if i == ('飞机', None):
            tagged_files[j][n] = ('飞机', 'noun')
        if i == ('甲', None): 
            tagged_files[j][n] = ('甲', 'numeral')
        if i == ('乙', None): 
            tagged_files[j][n] = ('乙', 'numeral')
        if i == ('丙', None): 
            tagged_files[j][n] = ('丙', 'numeral')
        if i == ('丁', None): 
            tagged_files[j][n] = ('丁', 'numeral')
        if i == ('辰', None): 
            tagged_files[j][n] = ('辰', 'numeral')
        if i == ('癸', None): 
            tagged_files[j][n] = ('癸', 'numeral')  
        if i == ('戊', None): 
            tagged_files[j][n] = ('戊', 'numeral')
        if i == ('巳', None): 
            tagged_files[j][n] = ('巳', 'numeral')
        if i == ('\u3000', None): 
            tagged_files[j][n] = ('\u3000', 'None')
        if i == ('贴吧', None): 
            tagged_files[j][n] = ('贴吧', 'noun')  
        if i == (' ', None): 
            tagged_files[j][n] = (' ', 'empty')  


# In[11]:


#add verb plus genitive de as nominalisation feature
import re
def verb_de(text): 
    verb_list=list(map(list, zip([item for item in text if re.match(r'\bverb\b', item[1])], text[1:])))
    flat_list = [item for sublist in verb_list for item in sublist]
    return round ((flat_list.count(('的', 'particle'))/len(text))*1000, 2)


# In[12]:


import pandas as pd  
df = pd.read_csv(folder + "linguistic_features.csv", header=0)


# In[13]:


verb_de_result=[]
for file in tagged_files: 
    verb_de_result.append(verb_de(file))
    
df['verb_de'] = pd.Series(verb_de_result)
df.to_csv(folder + 'linguistic_features.csv')


# In[14]:


df['NOMZ'] = df['NOMZ'] + df['verb_de']
del df['verb_de']


# In[15]:


df.to_csv(folder + 'linguistic_features.csv')


# In[16]:


#close pynlpir to free allocated memory
pynlpir.close()

