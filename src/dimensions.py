#!/Library/Frameworks/Python.framework/Versions/3.9/bin/python3
# coding: UTF-8
#dependencies
import os
import pandas as pd
import csv
from os import listdir
from sklearn.preprocessing import StandardScaler
import numpy as np

#input your folder containing linguistic feature file 
user_input = input("Enter the path of your linguistic feature file: ")
     
assert os.path.exists(user_input), "MulDi Chinese did not find the file at, "+str(user_input)
f = open(user_input,'r')
print("Found your file")

#read linguistic features stats

stats=pd.read_csv(f, header=0, index_col=0, quoting=csv.QUOTE_NONE)

print(stats.head())


#standardisation
stdsc = StandardScaler()
stats_std = stdsc.fit_transform(stats)


#create dataframe from standardised feature frequencies
df = pd.DataFrame(data=np.array(stats_std), index=stats.index, columns=stats.columns)

text_list=list(df.index.values)


#dimension 1 
def dimension1(text): 
    return df.loc[text, 'question']+df.loc[text, 'particle']+df.loc[text, 'exclamation']+df.loc[text, 'SPP']+df.loc[text, 'interrogative']+df.loc[text, 'PUBV']+df.loc[text, 'WH']+df.loc[text, 'mono_negation']+df.loc[text, 'Chinese_person']+df.loc[text, 'honourifics']+df.loc[text, 'FPP']+df.loc[text, 'INPR']+df.loc[text, 'emotion']-df.loc[text, 'AWL']

dimension1_scores=[]
for text in text_list:
    dimension1_scores.append(dimension1(text))


#dimension 2
def dimension2(text): 
    return df.loc[text, 'descriptive']+df.loc[text, 'imperfect']+df.loc[text, 'adverbial_marker_di']+df.loc[text, 'simile']+df.loc[text, 'PEAS']+df.loc[text, 'onomatopoeia']+df.loc[text, 'TPP']+df.loc[text, 'classifier']+df.loc[text, 'mono_verbs']+df.loc[text, 'SMP']+df.loc[text, 'complement_marker_de']

dimension2_scores=[]
for text in text_list:
    dimension2_scores.append(dimension2(text))

#dimension 3
def dimension3(text): 
    return df.loc[text, 'BE']+df.loc[text, 'COND']+df.loc[text, 'modify_adv']+df.loc[text, 'AMP']+df.loc[text, 'EX']+df.loc[text, 'HDG']+df.loc[text, 'DWNT']+df.loc[text, 'RB']+df.loc[text, 'di_negation']+df.loc[text, 'HSK_3']+df.loc[text, 'DEMP']+df.loc[text, 'PRIV']+df.loc[text, 'HSK_1']+df.loc[text, 'other_personal']-df.loc[text, 'noun']

dimension3_scores=[]
for text in text_list:
    dimension3_scores.append(dimension3(text))


#dimension 4
def dimension4(text): 
    return df.loc[text, 'ACL']+df.loc[text, 'ASL']+df.loc[text, 'ASL_std']+df.loc[text, 'PHC']+df.loc[text, 'BPIN']-df.loc[text, 'unique']-df.loc[text, 'intransitive']-df.loc[text, 'di_verbs']

dimension4_scores=[]
for text in text_list:
    dimension4_scores.append(dimension4(text))


#dimension 5
def dimension5(text): 
    return df.loc[text, 'classical_gram']+df.loc[text, 'classical_syntax']-df.loc[text, 'aux_adj']-df.loc[text, 'lexical_density']-df.loc[text, 'NOMZ']-df.loc[text, 'disyllabic_words']

dimension5_scores=[]
for text in text_list:
    dimension5_scores.append(dimension5(text))

#create dataframe for dimension scores 
d = {'text': text_list, 'dimension1':dimension1_scores, 'dimension2':dimension2_scores,    'dimension3':dimension3_scores, 'dimension4':dimension4_scores,    'dimension5':dimension5_scores}
df = pd.DataFrame(data=d)

folder=os.path.dirname(os.path.abspath(user_input))+'/'

#write dimension scores
df.to_csv(folder + 'dimension_scores.csv', index=False)

print("Completed. Dimension scores written.")