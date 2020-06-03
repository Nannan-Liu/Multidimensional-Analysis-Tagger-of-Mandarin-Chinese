
# coding: utf-8

# In[114]:


#your folder containing linguistic feature file 
folder = '//'


# In[115]:


#read linguistic features stats
import pandas as pd
import csv
stats=pd.read_csv(folder + "linguistic_features.csv", header=0, quoting=csv.QUOTE_NONE)


# In[116]:


stats.head()


# In[117]:


#use txt file names as dataframe index
from os import listdir

import os
files=[]
for file in os.listdir(folder):
    if file.endswith(".txt"):
        files.append(file)
        file_names = sorted([f.replace('.txt', '') for f in files])


# In[118]:


stats.index=file_names
stats.index.name='text'


# In[119]:


#standardisation
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
stats_std = stdsc.fit_transform(stats)


# In[120]:


#create dataframe from standardised feature frequencies
import numpy as np
df = pd.DataFrame(data=np.array(stats_std), index=stats.index, columns=stats.columns)


# In[121]:


text_list=list(df.index.values)


# In[138]:


df.head()


# In[124]:


def dimension1(text): 
    return df.loc[text, 'question']+df.loc[text, 'particle']+df.loc[text, 'interrogative']+df.loc[text, 'exclamation']+df.loc[text, 'SPP']+df.loc[text, 'WH']+df.loc[text, 'PUBV']+df.loc[text, 'mono_negation']+df.loc[text, 'FPP']+df.loc[text, 'honourifics']+df.loc[text, 'INPR']+df.loc[text, 'PRIV']+df.loc[text, 'other_personal']+df.loc[text, 'emotion']-df.loc[text, 'classical_func']


# In[125]:


dimension1_scores=[]
for text in text_list:
    dimension1_scores.append(dimension1(text))


# In[127]:


def dimension2(text): 
    return df.loc[text, 'BE']+df.loc[text, 'modify_adv']+df.loc[text, 'COND']+df.loc[text, 'RB']+df.loc[text, 'EX']+df.loc[text, 'DEMP']+df.loc[text, 'AMP']+df.loc[text, 'HDG']+df.loc[text, 'DWNT']+df.loc[text, 'di_negation']+df.loc[text, 'HSK_1']+df.loc[text, 'HSK_3']-df.loc[text, 'noun']


# In[128]:


dimension2_scores=[]
for text in text_list:
    dimension2_scores.append(dimension2(text))


# In[129]:


def dimension3(text): 
    return df.loc[text, 'lexical_density']+df.loc[text, 'NOMZ']+df.loc[text, 'disyllabic_words']+df.loc[text, 'HSK_6']+df.loc[text, 'abstract']+df.loc[text, 'PHC']+df.loc[text, 'consecutive_nouns']+df.loc[text, 'aux_adj']+df.loc[text, 'AWL']+df.loc[text, 'BPIN']-df.loc[text, 'classifier']-df.loc[text, 'person']-df.loc[text, 'mono_verbs']-df.loc[text, 'unique']


# In[130]:


dimension3_scores=[]
for text in text_list:
    dimension3_scores.append(dimension3(text))


# In[131]:


def dimension4(text): 
    return df.loc[text, 'modify_marker_di']+df.loc[text, 'imperfect']+df.loc[text, 'descriptive']+df.loc[text, 'simile']+df.loc[text, 'PEAS']+df.loc[text, 'TPP']+df.loc[text, 'SMP']+df.loc[text, 'modify_marker_de']


# In[132]:


dimension4_scores=[]
for text in text_list:
    dimension4_scores.append(dimension4(text))


# In[133]:


def dimension5(text): 
    return df.loc[text, 'english']+df.loc[text, 'ACL_std']+df.loc[text, 'ACL']+df.loc[text, 'ASL']


# In[134]:


dimension5_scores=[]
for text in text_list:
    dimension5_scores.append(dimension5(text))


# In[135]:


d = {'text': text_list, 'dimension1':dimension1_scores, 'dimension2':dimension2_scores,    'dimension3':dimension3_scores, 'dimension4':dimension4_scores,    'dimension5':dimension5_scores}


# In[136]:


#dimension scores
df = pd.DataFrame(data=d)


# In[137]:


with open(folder+'dimension_scores.csv', 'a') as f:
             df.to_csv(f, header=True)

