
# coding: utf-8

#input your txt file folder
folder = '//'

#load data, read your txt file folder with NLTK
import os
import nltk 
from nltk.corpus import CategorizedPlaintextCorpusReader
corpus = CategorizedPlaintextCorpusReader(
    folder,
    r'(?!\.).*\.txt',
    cat_pattern=os.path.join(r'(neg|pos)', '.*',),
    encoding='utf-8')
corpus.words()
#example words




#read the output dataframe
import pandas as pd  
df = pd.read_csv(folder + "linguistic_features.csv", header=0)
df.head()

#sort text files by their names, if necessary
import fileinput
import fnmatch

files=corpus.fileids()
#sample_files=[]

for f in files: 
    #if fnmatch.fnmatch(f, 'sample*.txt'):
        print (f)
        #sample_files.append(f)




#create corpora
corpora=[]
for file in files: 
    sub_corpora=corpus.raw(file)
    corpora.append(sub_corpora)



#use ICTCLAS to part-of-speech tag
import pynlpir
pynlpir.open()



#tag corpora, here we set the pos_names to be `child' for more fine-grained deails
tagged_files=[]
for sub_corpora in corpora: 
    tagged_file=pynlpir.segment(sub_corpora, pos_tagging=True, pos_names='child')
    tagged_files.append(tagged_file)



#the error message above indicates ICTCLAS has problems with some proper nouns and new nouns
#they are not tagged, so need to be manually tagged
none_list=[]
for file in tagged_files: 
    none_list.append([s for s in file if None in s])


print (none_list)



#the number in (range) is your number of files 
#the following are new words I found that ICTCLAS does not know
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


#you can double check if they are replaced
print (none_list)




#close pynlpir to free allocated memory
pynlpir.close()



#feature 8 COND 条件连词、副词
#如果……（那么）、只有……（才）、假如、除非、要是、要不是、只要、假使、假如、倘若、倘或、倘、设使、设若、如若、若
def cond(text_type):
    def raw(text_type): 
        return str(text_type).count('如果')+str(text_type).count('只有')    +str(text_type).count('假如')+str(text_type).count('除非')    +str(text_type).count('要是')+str(text_type).count('要不是')    +str(text_type).count('只要')+str(text_type).count('假如')    +str(text_type).count('倘若')+str(text_type).count('倘或')    +str(text_type).count('设使')+str(text_type).count('设若')    +str(text_type).count('如若')+str(text_type).count('若')    +text_type.count(('的话', 'particle 的话'))+    str(text_type).count("('的', 'particle 的/底'), ('时候', 'noun')")
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 


cond_result=[]
for file in tagged_files: 
    cond_result.append(cond(file))
    
df['COND'] = pd.Series(cond_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 14 modifying adverbs
def modify_adv(text_type): 
    def raw(text_type): 
        return text_type.count(('也', 'adverb'))+    text_type.count(('都', 'adverb'))+text_type.count(('又', 'adverb'))    +text_type.count(('才', 'adverb'))+text_type.count(('就', 'adverb'))    +text_type.count(('就是', 'adverb'))+text_type.count(('倒是', 'adverb'))    +text_type.count(('越来越', 'adverb'))+text_type.count(('一边', 'adverb'))    +text_type.count(('再', 'adverb'))+text_type.count(('甚至', 'adverb'))    +text_type.count(('连', 'particle 连'))+text_type.count(('却', 'adverb'))    +text_type.count(('原本', 'adverb'))+text_type.count(('只', 'adverb'))    +text_type.count(('毕竟', 'adverb'))+text_type.count(('仍然', 'adverb'))    +text_type.count(('反正', 'adverb'))+text_type.count(('等', 'particle 等/等等/云云'))    +text_type.count(('刚', 'adverb'))+text_type.count(('常常', 'adverb'))    +text_type.count(('已经', 'adverb'))+text_type.count(('就要', 'adverb'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



modify_adv_result=[]
for file in tagged_files: 
    modify_adv_result.append(modify_adv(file))
    
df['modify_adv'] = pd.Series(modify_adv_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 11 demonstrative pronoun
def demp(text_type): 
    def raw(text_type): 
        return str(text_type).count('demonstrative pronoun')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 


demp_result=[]
for file in tagged_files: 
    demp_result.append(demp(file))
    
df['DEMP'] = pd.Series(demp_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 7 Be是
def be(text_type):
    def raw(text_type): 
        return text_type.count(('是', 'verb 是'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



be_result=[]
for file in tagged_files: 
    be_result.append(be(file))
    
df['BE'] = pd.Series(be_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 15 EX有
def ex(text_type): 
    def raw(text_type): 
        return str(text_type).count('verb 有')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



ex_result=[]
for file in tagged_files: 
    ex_result.append(ex(file))
    
df['EX'] = pd.Series(ex_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 18 other personal pronouns apart from FPP, SPP, TPP
def other_personal(text_type): 
    def raw(text_type): 
        return str(text_type).count('personal pronoun')-str(text_type).count('我')-str(text_type).count('你')-str(text_type).count('她')-str(text_type).count('他')-str(text_type).count('它')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



other_personal_result=[]
for file in tagged_files: 
    other_personal_result.append(other_personal(file))
    
df['other_personal'] = pd.Series(other_personal_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature interrogative pronouns
def interrogative(text_type): 
    def raw(text_type): 
        return str(text_type).count('interrogative pronoun')-str(text_type).count('predicative interrogative pronoun')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



interrogative_result=[]
for file in tagged_files: 
    interrogative_result.append(interrogative(file))
    
df['interrogative'] = pd.Series(interrogative_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 23 nouns, all other nouns excluding nominalisation
#noun - noun-adjective - noun-verb - pronoun-
#personal pronoun - predicate demonstrative pronoun 
#-demonstrative pronoun - locative demonstrative pronoun 
#- predicate interrogative pronoun
#- interrogative pronoun (什么) - predicate demonstrative pronoun
def noun(text_type):
    def raw(text_type): 
        return str(text_type).count('noun')-str(text_type).count('noun-adjective')-str(text_type).count('noun-verb')-str(text_type).count('pronoun')-str(text_type).count('noun of locality')+str(text_type).count('noun morpheme')+str(text_type).count('proper noun')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)        



noun_result=[]
for file in tagged_files: 
    noun_result.append(noun(file))
    
df['noun'] = pd.Series(noun_result)
df.to_csv(folder + 'linguistic_features.csv')



#please note that nominalization features include also verb plus genitive marker de 的, generated by a separate script
#feature 24 nominalization NOMZ 
#noun-adjective, noun-verb
def nomz(text_type):
    def raw(text_type): 
        return str(text_type).count('noun-adjective')+str(text_type).count('noun-verb')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)        



nomz_result=[]
for file in tagged_files: 
    nomz_result.append(nomz(file))
    
df['NOMZ'] = pd.Series(nomz_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 26 phrasal coordinations PHC 
# same tags before and after 和、以及、与、并、及、暨
#ICTCLAS coordinating conjunction
def phc(text_type):
    def raw(text_type):
        return str(text_type).count('coordinating conjunction')
    def normalized(text_type):                                                                                     
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



phc_result=[]
for file in tagged_files: 
    phc_result.append(phc(file))
    
df['PHC'] = pd.Series(phc_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 28 BPIN disyllabic prepositions
#按照、本着、按着、朝着、趁着、出于、待到、对于、根据、关于、基于、鉴于
#借着、经过、靠着、冒着、面对、面临、凭借、顺着、随着、通过、为了
#围绕、向着、沿着、依据、针对
def bpin(text_type):
    def raw(text_type):
        return text_type.count(('按照', 'preposition'))+text_type.count(('本着', 'preposition'))+text_type.count(('按着', 'preposition'))+text_type.count(('朝着', 'preposition'))+text_type.count(('趁着', 'preposition'))+text_type.count(('出于', 'preposition'))+text_type.count(('待到', 'preposition'))+text_type.count(('对于', 'preposition'))+text_type.count(('根据', 'preposition'))+text_type.count(('关于', 'preposition'))+text_type.count(('基于', 'preposition'))+text_type.count(('鉴于', 'preposition'))+text_type.count(('借着', 'preposition'))+text_type.count(('经过', 'preposition'))+text_type.count(('靠着', 'preposition'))+text_type.count(('冒着', 'preposition'))+text_type.count(('面对', 'preposition'))+text_type.count(('面临', 'preposition'))+text_type.count(('凭借', 'preposition'))+text_type.count(('顺着', 'preposition'))+text_type.count(('随着', 'preposition'))+text_type.count(('通过', 'preposition'))+text_type.count(('为了', 'preposition'))+text_type.count(('围绕', 'preposition'))+text_type.count(('向着', 'preposition'))+text_type.count(('沿着', 'preposition'))+text_type.count(('依据', 'preposition'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)    




bpin_result=[]
for file in tagged_files: 
    bpin_result.append(bpin(file))
    
df['BPIN'] = pd.Series(bpin_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 31 private verbs 
#subset 1
def priv1(text_type):
    def raw(text_type): 
        return text_type.count(('三思', 'verb'))+text_type.count(('三省', 'verb'))+text_type.count(('主张', 'verb'))+text_type.count(('了解', 'verb'))+text_type.count(('亲信', 'verb'))+text_type.count(('以为', 'verb'))+text_type.count(('企图', 'verb'))+text_type.count(('会意', 'verb'))+text_type.count(('伤心', 'verb'))+text_type.count(('估', 'verb'))+text_type.count(('估摸', 'verb'))+text_type.count(('估算', 'verb'))+text_type.count(('估计', 'verb'))+text_type.count(('估量', 'verb'))+text_type.count(('低估', 'verb'))+text_type.count(('体会', 'verb'))+text_type.count(('体味', 'verb'))+text_type.count(('信', 'verb'))+text_type.count(('信任', 'verb'))+text_type.count(('信赖', 'verb'))+                text_type.count(('修省', 'verb'))+text_type.count(('假定', 'verb'))+                text_type.count(('假想', 'verb'))+text_type.count(('允许', 'verb'))+                text_type.count(('关心', 'verb'))+text_type.count(('关怀', 'verb'))+                text_type.count(('内省', 'verb'))+text_type.count(('决定', 'verb'))+                text_type.count(('决心', 'verb'))+text_type.count(('决意', 'verb'))+                text_type.count(('决断', 'verb'))+text_type.count(('决计', 'verb'))+                text_type.count(('准备', 'verb'))+text_type.count(('准许', 'verb'))+                text_type.count(('凝思', 'verb'))+text_type.count(('凝想', 'verb'))+                text_type.count(('凭信', 'verb'))+text_type.count(('分晓', 'verb'))+                text_type.count(('切记', 'verb'))+text_type.count(('划算', 'verb'))+                text_type.count(('判断', 'verb'))+text_type.count(('原谅', 'verb'))+                text_type.count(('参悟', 'verb'))+text_type.count(('反对', 'verb'))+                text_type.count(('反思', 'verb'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)        


p_1 = []
for file in tagged_files: 
    p_1.append(priv1(file)


#subset 2
def priv2(text_type):
    def raw(text_type): 
        return text_type.count(('反省', 'verb'))+text_type.count(('发现', 'verb'))+text_type.count(('发觉', 'verb'))+text_type.count(('吃准', 'verb'))+text_type.count(('合计', 'verb'))+text_type.count(('合谋', 'verb'))+text_type.count(('同情', 'verb'))+text_type.count(('同意', 'verb'))+text_type.count(('否认', 'verb'))+text_type.count(('听信', 'verb'))+text_type.count(('听到', 'verb'))+text_type.count(('听见', 'verb'))+text_type.count(('哭', 'verb'))+text_type.count(('喜欢', 'verb'))+text_type.count(('喜爱', 'verb'))+text_type.count(('回味', 'verb'))+text_type.count(('回忆', 'verb'))+text_type.count(('回念', 'verb'))+text_type.count(('回想', 'verb'))+text_type.count(('回溯', 'verb'))+text_type.count(('回顾', 'verb'))+text_type.count(('图谋', 'verb'))+text_type.count(('图', 'verb'))+text_type.count(('坚信', 'verb'))+text_type.count(('多疑', 'verb'))+text_type.count(('失望', 'verb'))+text_type.count(('失身', 'verb'))+text_type.count(('妄图', 'verb'))+text_type.count(('妄断', 'verb'))+text_type.count(('宠信', 'verb'))+text_type.count(('害怕', 'verb'))+text_type.count(('察觉', 'verb'))+text_type.count(('寻思', 'verb'))+text_type.count(('尊敬', 'verb'))+text_type.count(('尊重', 'verb'))+text_type.count(('小心', 'verb'))+text_type.count(('希望', 'verb'))+text_type.count(('平静', 'verb'))+text_type.count(('幻想', 'verb'))+text_type.count(('当做', 'verb'))+text_type.count(('彻悟', 'verb'))+text_type.count(('得知', 'verb'))+text_type.count(('忆', 'verb'))+text_type.count(('忖度', 'verb'))+text_type.count(('忖量', 'verb'))+text_type.count(('忘', 'verb'))+text_type.count(('忘却', 'verb'))+text_type.count(('忘怀', 'verb'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 


p_2 = []
for file in tagged_files: 
    p_2.append(priv2(file))


#subset 3
def priv3(text_type):
    def raw(text_type): 
        return text_type.count(('忘掉', 'verb'))+text_type.count(('忘记', 'verb'))+text_type.count(('快乐', 'verb'))+text_type.count(('念', 'verb'))+text_type.count(('忽略', 'verb'))+text_type.count(('忽视', 'verb'))+text_type.count(('怀念', 'verb'))+text_type.count(('怀想', 'verb'))+text_type.count(('怀疑', 'verb'))+text_type.count(('怕', 'verb'))+text_type.count(('思忖', 'verb'))+text_type.count(('思想', 'verb'))+text_type.count(('思索', 'verb'))+text_type.count(('思维', 'verb'))+text_type.count(('思考', 'verb'))+text_type.count(('思虑', 'verb'))+text_type.count(('思量', 'verb'))+text_type.count(('恨', 'verb'))+text_type.count(('悟', 'verb'))+text_type.count(('悬想', 'verb'))+text_type.count(('情知', 'verb'))+text_type.count(('惊恐', 'verb'))+text_type.count(('想', 'verb'))+text_type.count(('想像', 'verb'))+text_type.count(('想来', 'verb'))+text_type.count(('想见', 'verb'))+text_type.count(('想象', 'verb'))+text_type.count(('愉快', 'verb'))+text_type.count(('意会', 'verb'))+text_type.count(('意想', 'verb'))+text_type.count(('意料', 'verb'))+text_type.count(('意识', 'verb'))+text_type.count(('感到', 'verb'))+text_type.count(('感动', 'verb'))+text_type.count(('感受', 'verb'))+text_type.count(('感悟', 'verb'))+text_type.count(('感想', 'verb'))+text_type.count(('感激', 'verb'))+text_type.count(('感觉', 'verb'))+text_type.count(('感觉', 'verb'))+text_type.count(('感谢', 'verb'))+text_type.count(('愤怒', 'verb'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 


p_3 = []
for file in tagged_files: 
    p_3.append(priv3(file))



#subset 4
def priv4(text_type):
    def raw(text_type): 
        return text_type.count(('愿意', 'verb'))+text_type.count(('懂', 'verb'))+text_type.count(('懂得', 'verb'))+text_type.count(('打算', 'verb'))+text_type.count(('承想', 'verb'))+text_type.count(('承认', 'verb'))+text_type.count(('担心', 'verb'))+text_type.count(('拥护', 'verb'))+text_type.count(('捉摸', 'verb'))+text_type.count(('掂掇', 'verb'))+text_type.count(('掂量', 'verb'))+text_type.count(('掌握', 'verb'))+text_type.count(('推度', 'verb'))+text_type.count(('推想', 'verb'))+text_type.count(('推敲', 'verb'))+text_type.count(('推断', 'verb'))+text_type.count(('推测', 'verb'))+text_type.count(('推理', 'verb'))+text_type.count(('推算', 'verb'))+text_type.count(('推见', 'verb'))+text_type.count(('措意', 'verb'))+text_type.count(('揆度', 'verb'))+text_type.count(('揣度', 'verb'))+text_type.count(('揣想', 'verb'))+text_type.count(('揣摩', 'verb'))+text_type.count(('揣摸', 'verb'))+text_type.count(('揣测', 'verb'))+text_type.count(('支持', 'verb'))+text_type.count(('放心', 'verb'))+text_type.count(('料想', 'verb'))+text_type.count(('料', 'verb'))+text_type.count(('斟酌', 'verb'))+text_type.count(('断定', 'verb'))+text_type.count(('明了', 'verb'))+text_type.count(('明察', 'verb'))+text_type.count(('明晓', 'verb'))+text_type.count(('明白', 'verb'))+text_type.count(('明知', 'verb'))+text_type.count(('明确', 'verb'))+text_type.count(('晓得', 'verb'))+text_type.count(('权衡', 'verb'))+text_type.count(('梦想', 'verb'))+text_type.count(('欢迎', 'verb'))+text_type.count(('欣赏', 'verb'))+text_type.count(('武断', 'verb'))+text_type.count(('死记', 'verb'))+text_type.count(('沉思', 'verb'))+text_type.count(('注意', 'verb'))+text_type.count(('洞察', 'verb'))+text_type.count(('洞彻', 'verb'))+text_type.count(('洞悉', 'verb'))+text_type.count(('洞晓', 'verb'))+text_type.count(('洞达', 'verb'))+text_type.count(('测度', 'verb'))+text_type.count(('浮想', 'verb'))+text_type.count(('淡忘', 'verb'))+text_type.count(('深信', 'verb'))+text_type.count(('深思', 'verb'))+text_type.count(('深省', 'verb'))+text_type.count(('深醒', 'verb'))+text_type.count(('清楚', 'verb'))+text_type.count(('清楚', 'verb'))+text_type.count(('满意', 'verb'))+text_type.count(('满足', 'verb'))+text_type.count(('激动', 'verb'))+text_type.count(('热爱', 'verb'))+text_type.count(('熟悉', 'verb'))+text_type.count(('熟知', 'verb'))+text_type.count(('熟虑', 'verb'))+text_type.count(('爱', 'verb'))+text_type.count(('爱好', 'verb'))+text_type.count(('牢记', 'verb'))+text_type.count(('犯疑', 'verb'))+text_type.count(('狂想', 'verb'))+text_type.count(('狐疑', 'verb'))+text_type.count(('猛醒', 'verb'))+text_type.count(('猜', 'verb'))+text_type.count(('猜度', 'verb'))+text_type.count(('猜忌', 'verb'))+text_type.count(('猜想', 'verb'))+text_type.count(('猜测', 'verb'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



p_4 = []
for file in tagged_files: 
    p_4.append(priv4(file))



#subset 5 
def priv5(text_type):
    def raw(text_type): 
        return text_type.count(('猜疑', 'verb'))+text_type.count(('玄想', 'verb'))+text_type.count(('理会', 'verb'))+text_type.count(('理解', 'verb'))+text_type.count(('琢磨', 'verb'))+text_type.count(('生气', 'verb'))+text_type.count(('生疑', 'verb'))+text_type.count(('畅想', 'verb'))+text_type.count(('留心', 'verb'))+text_type.count(('留神', 'verb'))+text_type.count(('疏忽', 'verb'))+text_type.count(('疑', 'verb'))+text_type.count(('疑心', 'verb'))+text_type.count(('疑猜', 'verb'))+text_type.count(('疑虑', 'verb'))+text_type.count(('疼', 'verb'))+text_type.count(('盘算', 'verb'))+text_type.count(('相信', 'verb'))+text_type.count(('盼望', 'verb'))+text_type.count(('省察', 'verb'))+text_type.count(('省悟', 'verb'))+text_type.count(('看', 'verb'))+text_type.count(('看到', 'verb'))+text_type.count(('看见', 'verb'))+text_type.count(('看透', 'verb'))+text_type.count(('着想', 'verb'))+text_type.count(('知', 'verb'))+text_type.count(('知悉', 'verb'))+text_type.count(('知晓', 'verb'))+text_type.count(('知道', 'verb'))+text_type.count(('确信', 'verb'))+text_type.count(('确定', 'verb'))+text_type.count(('确认', 'verb'))+text_type.count(('空想', 'verb'))+text_type.count(('立意', 'verb'))+text_type.count(('笃信', 'verb'))+text_type.count(('笑', 'verb'))+text_type.count(('答应', 'verb'))+text_type.count(('策划', 'verb'))+text_type.count(('筹划', 'verb'))+text_type.count(('筹算', 'verb'))+text_type.count(('筹谋', 'verb'))+text_type.count(('算', 'verb'))+text_type.count(('算计', 'verb'))+text_type.count(('粗估', 'verb'))+text_type.count(('约摸', 'verb'))+text_type.count(('置疑', 'verb'))+text_type.count(('考虑', 'verb'))+text_type.count(('考量', 'verb'))+text_type.count(('联想', 'verb'))+text_type.count(('腹诽', 'verb'))+text_type.count(('臆度', 'verb'))+text_type.count(('臆想', 'verb'))+text_type.count(('臆断', 'verb'))+text_type.count(('臆测', 'verb'))+text_type.count(('自信', 'verb'))+text_type.count(('自省', 'verb'))+text_type.count(('蒙', 'verb'))+text_type.count(('蓄念', 'verb'))+text_type.count(('蓄谋', 'verb'))+text_type.count(('衡量', 'verb'))+text_type.count(('裁度', 'verb'))+text_type.count(('要求', 'verb'))+text_type.count(('观察', 'verb'))+text_type.count(('觉察', 'verb'))+text_type.count(('觉得', 'verb'))+text_type.count(('觉悟', 'verb'))+text_type.count(('觉醒', 'verb'))+text_type.count(('警惕', 'verb'))+text_type.count(('警觉', 'verb'))+text_type.count(('计划', 'verb'))+text_type.count(('计算', 'verb'))+text_type.count(('计较', 'verb'))+text_type.count(('认为', 'verb'))+text_type.count(('认可', 'verb'))+text_type.count(('认同', 'verb'))+text_type.count(('认定', 'verb'))+text_type.count(('认得', 'verb'))+text_type.count(('认知', 'verb'))+text_type.count(('认识', 'verb'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 


p_5 = []
for file in tagged_files: 
    p_5.append(priv5(file))




#6
def priv6(text_type):
    def raw(text_type): 
        return text_type.count(('讨厌', 'verb'))+text_type.count(('记', 'verb'))+text_type.count(('记取', 'verb'))+text_type.count(('记得', 'verb'))+text_type.count(('记忆', 'verb'))+text_type.count(('设想', 'verb'))+text_type.count(('识', 'verb'))+text_type.count(('试图', 'verb'))+text_type.count(('试想', 'verb'))+text_type.count(('详悉', 'verb'))+text_type.count(('误会', 'verb'))+text_type.count(('误解', 'verb'))+text_type.count(('谋划', 'verb'))+text_type.count(('谋算', 'verb'))+text_type.count(('谋虑', 'verb'))+text_type.count(('赞同', 'verb'))+text_type.count(('赞成', 'verb'))+text_type.count(('走神儿', 'verb'))+text_type.count(('起疑', 'verb'))+text_type.count(('轻信', 'verb'))+text_type.count(('轻视', 'verb'))+text_type.count(('迷信', 'verb'))+text_type.count(('迷信', 'verb'))+text_type.count(('追忆', 'verb'))+text_type.count(('追怀', 'verb'))+text_type.count(('追思', 'verb'))+text_type.count(('追想', 'verb'))+text_type.count(('通彻', 'verb'))+text_type.count(('通晓', 'verb'))+text_type.count(('通', 'verb'))+text_type.count(('遐想', 'verb'))+text_type.count(('遗忘', 'verb'))+text_type.count(('遥想', 'verb'))+text_type.count(('酌情', 'verb'))+text_type.count(('酌量', 'verb'))+text_type.count(('醒', 'verb'))+text_type.count(('醒悟', 'verb'))+text_type.count(('重视', 'verb'))+text_type.count(('铭记', 'verb'))+text_type.count(('阴谋', 'verb')) +text_type.count(('顾全', 'verb'))+text_type.count(('顾及', 'verb'))+text_type.count(('预卜', 'verb'))+text_type.count(('预想', 'verb'))+text_type.count(('预感', 'verb'))+text_type.count(('预料', 'verb'))+text_type.count(('预期', 'verb'))+text_type.count(('预测', 'verb'))+text_type.count(('预知', 'verb'))+text_type.count(('预见', 'verb'))+text_type.count(('预计', 'verb'))+text_type.count(('预谋', 'verb'))+text_type.count(('领会', 'verb'))+text_type.count(('领悟', 'verb'))+text_type.count(('领略', 'verb'))+text_type.count(('高估', 'verb'))+text_type.count(('高兴', 'verb'))+text_type.count(('默认', 'verb'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 

p_6 = []
for file in tagged_files: 
    p_6.append(priv6(file))


random1=[sum(i) for i in zip(p_1, p_2)]
random2=[sum(i) for i in zip(p_3, p_4)]
random3=[sum(i) for i in zip(p_5, p_6)]
random4=[sum(i) for i in zip(random1, random2)]
final=[sum(i) for i in zip(random3, random4)]
    
df['PRIV'] = pd.Series(final)
df.to_csv(folder + 'linguistic_features.csv')



#feature 32 public verb PUBV
#表示、称、道、说、讲、质疑、认为、坦言、指出、告诉、呼吁、解释、问、建议
def pubv(text_type):
    def raw(text_type): 
        return text_type.count(('表示', 'verb'))+text_type.count(('称', 'verb'))+text_type.count(('道', 'verb'))+text_type.count(('说', 'verb'))+text_type.count(('讲', 'verb'))+text_type.count(('质疑', 'verb'))+text_type.count(('认为', 'verb'))+text_type.count(('坦言', 'verb'))+text_type.count(('指出', 'verb'))+text_type.count(('告诉', 'verb'))+text_type.count(('呼吁', 'verb'))+text_type.count(('解释', 'verb'))+text_type.count(('问', 'verb'))+text_type.count(('建议', 'verb'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)


pubv_result=[]
for file in tagged_files: 
    pubv_result.append(pubv(file))
    
df['PUBV'] = pd.Series(pubv_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 33 RB adverbs 副词
def rb(text_type):
    def raw(text_type): 
        return str(text_type).count('adverb')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)



rb_result=[]
for file in tagged_files: 
    rb_result.append(rb(file))
    
df['RB'] = pd.Series(rb_result)
df.to_csv(folder + 'linguistic_features.csv')


#feature 35 monosyllabic negation 不、别、没
#('别', 'adverb')、('不', 'adverb')
#('没', 'verb')、('没', 'adverb')
def mono_negation(text_type):
    def raw(text_type): 
        return text_type.count(('别', 'adverb'))+text_type.count(('不', 'adverb'))+text_type.count(('没', 'verb'))+text_type.count(('没', 'adverb'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)



mono_negation_result=[]
for file in tagged_files: 
    mono_negation_result.append(mono_negation(file))
    
df['mono_negation'] = pd.Series(mono_negation_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 36 disyllabic negation 没有
#('没有', 'adverb')、('没有', 'verb')
def di_negation(text_type):
    def raw(text_type): 
        return text_type.count(('没有', 'adverb'))+text_type.count(('没有', 'verb'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)

di_negation_result=[]
for file in tagged_files: 
    di_negation_result.append(di_negation(file))
    
df['di_negation'] = pd.Series(di_negation_result)
df.to_csv(folder + 'linguistic_features.csv')


#feature 40 WH 无定代词
#('怎么', 'predicate interrogative pronoun')、('怎么样', 'predicate interrogative pronoun')
#('怎样', 'predicate interrogative pronoun')
#('为什么', 'predicate interrogative pronoun')
#('如何', 'predicate interrogative pronoun')
#('什么样', 'predicate interrogative pronoun')
#search for predicate interrogative pronoun
def wh(text_type):
    def raw(text_type):
        return str(text_type).count('predicate interrogative pronoun')
    def normalized(text_type):                                                                                     
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 

wh_result=[]
for file in tagged_files: 
    wh_result.append(wh(file))
    
df['WH'] = pd.Series(wh_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 41 monosyllabic verbs
#1. convert corpora to dict # no None in values 
#2. select verbs from dict values 
#3. convert corresponding keys to list 
#4. return len(corresponding keys) == 2 or 1 / len (filtered dict)

##dict would remove duplicates, not ideal
dicts=[]
for file in tagged_files: 
    d=dict(file)
    dicts.append(d)


import re
def mono_verbs(text): 
    verbs= {k:v for k,v in text.items() if re.match('.*verb', v)}
    return round ((len([word for word in list(verbs.keys()) if len(word) == 1]) / len(text))*1000, 2)


mono_verbs_result=[]
for d in dicts: 
    mono_verbs_result.append(mono_verbs(d))
    
df['mono_verbs'] = pd.Series(mono_verbs_result)
df.to_csv(folder + 'linguistic_features.csv')


#feature 41 classical function words 文言文功能词
#('所', 'particle 所')、('将', 'adverb')、('将', 'preposition')
#('之', 'particle 之')、('于', 'preposition')、('以', 'preposition')
def classical_func(text_type):
    def raw(text_type): 
        return text_type.count(('所', 'particle 所'))+text_type.count(('将', 'adverb'))+text_type.count(('将', 'preposition'))+text_type.count(('之', 'particle 之'))+text_type.count(('于', 'preposition'))+text_type.count(('以', 'preposition'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)



classical_func_result=[]
for file in tagged_files: 
    classical_func_result.append(classical_func(file))
    
df['classical_func'] = pd.Series(classical_func_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 50 lexical density 
#(noun+verb+adjective+adverb) / total
#note that count.verbs contains count.adverbs
def lexical_density(text): 
    verbs= {k:v for k,v in text.items() if re.match('.*verb', v)}
    nouns= {k:v for k,v in text.items() if re.match('.*noun', v)}
    adjectives= {k:v for k,v in text.items() if re.match('.*adjective', v)}
    #adverbs={k:v for k,v in text.items() if re.match('.*adverb', v)}
    pronouns={k:v for k,v in text.items() if re.match('.*pronoun', v)}
    return round(((len(verbs)+len(nouns)+len(adjectives)-len(pronouns)) / len(text))*1000, 2)



#note that tagged lists are converted to dictionaries here as well 
lexical_density_result=[]
for d in dicts: 
    lexical_density_result.append(lexical_density(d))
    
df['lexical_density'] = pd.Series(lexical_density_result)
df.to_csv(folder + 'linguistic_features.csv')




#feature 61 auxiliary adjectives
def aux_adj(text_type):
    def raw(text_type):
        return str(text_type).count('auxiliary adjective')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)



aux_adj_result=[]
for file in tagged_files: 
    aux_adj_result.append(aux_adj(file))
    
df['aux_adj'] = pd.Series(aux_adj_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 48 classifier 量词
def classifier(text_type):
    def raw(text_type): 
        return str(text_type).count('classifier')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)



classifier_result=[]
for file in tagged_files: 
    classifier_result.append(classifier(file))
    
df['classifier'] = pd.Series(classifier_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 64 modal particles and interjections 
def particle(text_type):
    def raw(text_type): 
        return str(text_type).count('modal particle')+str(text_type).count('interjection')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)



particle_result=[]
for file in tagged_files: 
    particle_result.append(particle(file))
    
df['particle'] = pd.Series(particle_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 49 modifying markers 的、地、得
def modify_marker_di(text_type):
    def raw(text_type):
        return text_type.count(('地', 'particle 地'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)    



modify_marker_di_result=[]
for file in tagged_files: 
    modify_marker_di_result.append(modify_marker_di(file))
    
df['modify_marker_di'] = pd.Series(modify_marker_di_result)
df.to_csv(folder + 'linguistic_features.csv')




#feature 51 modifying marker '得', 'particle 得'
def modify_marker_de(text_type):
    def raw(text_type):
        return text_type.count(('得', 'particle 得'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)    



modify_marker_de_result=[]
for file in tagged_files: 
    modify_marker_de_result.append(modify_marker_de(file))
    
df['modify_marker_de'] = pd.Series(modify_marker_de_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 51 PEAS perfect aspect 
#('了', 'particle 了/喽')、('过', 'particle 过')
def peas(text_type):
    def raw(text_type):
        return text_type.count(('了', 'particle 了/喽'))+text_type.count(('过', 'particle 过'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)    



peas_result=[]
for file in tagged_files: 
    peas_result.append(peas(file))
    
df['PEAS'] = pd.Series(peas_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 52 imperfect aspect 
#('着', 'particle 着')
#('在', 'preposition')、('正在', 'adverb')
#('起来', 'directional verb')、('下去', 'directional verb')
def imperfect(text_type):
    def raw(text_type):
        return text_type.count(('着', 'particle 着'))+text_type.count(('在', 'preposition'))    +text_type.count(('正在', 'adverb'))+text_type.count(('起来', 'directional verb'))+text_type.count(('下去', 'directional verb'))
    def normalized(text_type):                                                                                
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



imperfect_result=[]
for file in tagged_files: 
    imperfect_result.append(imperfect(file))
    
df['imperfect'] = pd.Series(imperfect_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 54 descriptive words 
#ICTCLAS status word
def descriptive(text_type):
    def raw(text_type):
        return str(text_type).count('status word')
    def normalized(text_type):                                                                                     
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 




descriptive_result=[]
for file in tagged_files: 
    descriptive_result.append(descriptive(file))
    
df['descriptive'] = pd.Series(descriptive_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 55 simile
def simile(text_type):
    def raw(text_type):
        return text_type.count(('仿佛', 'adverb'))+text_type.count(('宛若', 'verb'))    +text_type.count(('如', 'verb'))+str(text_type).count(('particle 一样/一般/似的/般'))    +text_type.count(('像', 'verb'))+text_type.count(('像', 'preposition'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)



simile_result=[]
for file in tagged_files: 
    simile_result.append(simile(file))
    
df['simile'] = pd.Series(simile_result)
df.to_csv(folder + 'linguistic_features.csv')


#feature 86 question mark
def question(text_type):
    def raw(text_type):
        return text_type.count(('？', 'question mark'))
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)


question_result=[]
for file in tagged_files: 
    question_result.append(question(file))
    
df['question'] = pd.Series(question_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 87 exclamation mark
def exclamation(text_type):
    def raw(text_type):
        return str(text_type).count('exclamation mark')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)



exclamation_result=[]
for file in tagged_files: 
    exclamation_result.append(exclamation(file))
    
df['exclamation'] = pd.Series(exclamation_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 46 noun + noun combination
#need to plus one for the final result because 
#"if a tag pattern matches at overlapping locations, the leftmost match takes precedence"
#Bird et al.
import nltk
import re 
grammar_noun=r"NN: {<noun.*><noun.*>}"
cp_noun=nltk.RegexpParser(grammar_noun) 



def nn(text): 
    def parse(text): 
        return cp_noun.parse(text)
    nn_list=[]
    for subtree in parse(text).subtrees():
        if subtree.label() == 'NN': 
            nn_list.append(subtree.leaves())
    return round ((len(nn_list)/len(text))*1000, 2)


nn_result=[]
for file in tagged_files: 
    nn_result.append(nn(file))
    
df['consecutive_nouns'] = pd.Series(nn_result)
df.to_csv(folder + 'linguistic_features.csv')


#feature 93 person names, personal name - transcribed personal names
def person(text_type):
    def raw(text_type):
        return str(text_type).count('personal name')+str(text_type).count('Chinese')- str(text_type).count('transcribed personal name')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)



person_result=[]
for file in tagged_files: 
    person_result.append(person(file))
    
df['person'] = pd.Series(person_result)
df.to_csv(folder + 'linguistic_features.csv')

