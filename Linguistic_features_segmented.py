
# coding: utf-8


#input your txt file folder
folder = '//'


#create a csv file to store all linguistic feature data
import csv

with open(folder + 'linguistic_features.csv', 'wb') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)


# In[9]:


#pandas does not allow empty csv to write, so write linguistic features in header
with open(folder + 'linguistic_features.csv', 'w', newline='') as outcsv:
    writer = csv.DictWriter(outcsv, fieldnames = ["Linguistic features"])
    writer.writeheader()


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


#sort text files by their names, if necessary
#if not, check file ids
import fileinput
import fnmatch

files=corpus.fileids()

for f in files: 
    #if fnmatch.fnmatch(f, '7*.txt'):
        print (f)
        #sample_files.append(f)



import pandas as pd  
df = pd.read_csv(folder + 'linguistic_features.csv')


#create corpora
corpora=[]
for file in files: 
    sub_corpora=corpus.raw(file)
    corpora.append(sub_corpora)



#use ICTCLAS to segment
import pynlpir
pynlpir.open()



#segment corpora, here we set pos_tagging to be false
sub_corpora=[]
for corpus in corpora: 
    sub_corpus=pynlpir.segment(corpus, pos_tagging=False)
    sub_corpora.append(sub_corpus)


#close pynlpir to free allocated memory
pynlpir.close()


#feature 3 AMP amplifiers 
#非常、大大、十分、真的、特别、很、最、肯定、挺、顶、极、
#极为、极其、极度、万分、格外、分外、更、更加、更为、尤其、太、过于、
#老、怪、相当、颇、颇为、有点儿、有些、最为、还、越发、越加、愈加、
#稍、稍微、稍稍、略、略略、略微、比较、较、暴、超、恶、怒、巨、粉、奇
amplifier=['非常', '大大', '十分', '真的', '真', '特别', '很', '最', '肯定', '挺', '顶', '极', '极为', '极其', '极度', '万分', '格外', '分外', '更', '更加', '更为', '尤其', '太', '过于', '老', '怪', '相当', '颇', '颇为', '有点儿', '有些', '最为', '还', '越发', '越加', '愈加', '稍', '稍微', '稍稍', '略', '略略', '略微', '比较', '较', '暴', '超', '恶', '怒', '巨', '粉', '奇', '很大', '相当', '完全', '显著', '总是', '根本', '一定']

#define count and standardise function
def amplifiers(text_type):
    def raw(text_type): 
        return len([x for x in text_type if x in amplifier])
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 


amplifiers_result=[]
for corpus in sub_corpora: 
    amplifiers_result.append(amplifiers(corpus))
    
df['AMP'] = pd.Series(amplifiers_result)
df.to_csv(folder + 'linguistic_features.csv')



#strip quotation marks first 
lists=[]
for corpus in sub_corpora: 
    l=list(corpus)
    lists.append(l)



#feature 6 average word length
def awl(text_type): 
    return round ((sum(len(word) for word in text_type) / len(text_type)), 2)


awl_result=[]
for l in lists: 
    awl_result.append(awl(l))
    
df['AWL'] = pd.Series(awl_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 7 mean sentence length
def asl(text_type): 
    sentences = [[]]
    ends = set('。？!……——')
    for word in text_type:
        if word in ends: sentences.append([])
        else: sentences[-1].append(word)
    if sentences[0]:
        if not sentences[-1]: sentences.pop()
        return round (sum(len(s) for s in sentences)/len(sentences), 2)


asl_result=[]
for l in lists: 
    asl_result.append(asl(l))

df['ASL'] = pd.Series(asl_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 8 standard deviation of sentence length 
import statistics

def asl_std(text_type): 
    sentences = [[]]
    ends = set('。？!……——')
    for word in text_type:
        if word in ends: sentences.append([])
        else: sentences[-1].append(word)
    if sentences[0]:
        if not sentences[-1]: sentences.pop()
        return round(statistics.stdev(len(s) for s in sentences), 2)



asl_std_result=[]
for l in lists: 
    for corpus in sub_corpora: 
    acl_std_result.append(acl_std(corpus))
    asl_std_result.append(asl_std(l))

df['ASL_std'] = pd.Series(asl_std_result)
df.to_csv(folder + 'linguistic_features.csv')


#feature 5 mean clause length
def acl(text_type): 
    sentences = [[]]
    ends = set('，：；。？!……——')
    for word in text_type:
        if word in ends: sentences.append([])
        else: sentences[-1].append(word)
    if sentences[0]:
        if not sentences[-1]: sentences.pop()
        return round (sum(len(s) for s in sentences)/len(sentences), 2)


acl_result=[]
for corpus in sub_corpora: 
    acl_result.append(acl(corpus))

df['ACL'] = pd.Series(acl_result)

df.to_csv(folder + 'linguistic_features.csv')



#feature 21 downtoners DWNT
#一点、有点、有点儿、稍、稍微、一些、有些
downtoners=['一点', '一点儿', '有点', '有点儿', '稍', '稍微', '一些', '有些']
def dwnt(text_type):
    def raw(text_type): 
        return len([x for x in text_type if x in downtoners])
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



dwnt_result=[]
for corpus in sub_corpora: 
    dwnt_result.append(dwnt(corpus))
    
df['DWNT'] = pd.Series(dwnt_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 25 FPP1 first person pronoun
#我、我的、我自己、我们、我们自己、我们的
#only 我 and 我们 are needed
def fpp1(text_type):
    def raw(text_type): 
        return text_type.count('我')+text_type.count('我们')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



fpp1_result=[]
for corpus in sub_corpora: 
    fpp1_result.append(fpp1(corpus))
    
df['FPP'] = pd.Series(fpp1_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 47 SPP2 second person pronoun
#你、你们、您、您们
def spp2(text_type):
    def raw(text_type): 
        return text_type.count('你')+text_type.count('你们')+text_type.count('您')+text_type.count('您们')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



spp2_result=[]
for corpus in sub_corpora: 
    spp2_result.append(spp2(corpus))
    
df['SPP'] = pd.Series(spp2_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 26 hedges HDG
#可能、可以、也许、较少、一些、多个、多为、基本、主要、类似、不少
hedges=['可能', '可以', '也许', '较少', '一些', '多个', '多为', '基本', '主要', '类似', '不少']
def hdg(text_type):
    def raw(text_type): 
        return len([x for x in text_type if x in hedges])
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



hdg_result=[]
for corpus in sub_corpora: 
    hdg_result.append(hdg(corpus))
    
df['HDG'] = pd.Series(hdg_result)
df.to_csv(folder + 'linguistic_features.csv')




#feature 31 INPR indefinite pronouns 无定代词
#任何、谁、大家、某、有人、有个、什么
indefinites=['任何', '谁', '大家', '某', '有人', '有个', '什么']
def inpr(text_type):
    def raw(text_type): 
        return len([x for x in text_type if x in indefinites])
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 


inpr_result=[]
for corpus in sub_corpora: 
    inpr_result.append(inpr(corpus))
    
df['INPR'] = pd.Series(inpr_result)
df.to_csv(folder + 'linguistic_features.csv')




#feature 48 SMP seem, appear
#好像、似乎、好象、貌似
def smp(text_type):
    def raw(text_type): 
        return text_type.count('好象')+text_type.count('好像')+text_type.count('似乎')+text_type.count('貌似')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 


smp_result=[]
for corpus in sub_corpora: 
    smp_result.append(smp(corpus))
    
df['SMP'] = pd.Series(smp_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 51 third person pronouns
tpp3s=['她', '他', '它', '她们', '他们', '它们']
def tpp3(text_type):
    def raw(text_type): 
        return len([x for x in text_type if x in tpp3s])
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



tpp3_result=[]
for corpus in sub_corpora: 
    tpp3_result.append(tpp3(corpus))
    
df['TPP'] = pd.Series(tpp3_result)
df.to_csv(folder + 'linguistic_features.csv')


#feature 22 emotion words 
def emotion(text): 
    def raw(text): 
        return text.count('烦恼')+text.count('不幸')+text.count('痛苦')+text.count('苦')+text.count('快乐')+text.count('忍')+text.count('喜')+text.count('乐')+text.count('称心')+text.count('痛快')+text.count('得意')+text.count('欣慰')+text.count('高兴')+text.count('愉悦')+text.count('欣喜')+text.count('欢欣')+text.count('可意')+text.count('乐')+text.count('可心')+text.count('欢畅')+text.count('开心')+text.count('康乐')+text.count('欢快')+text.count('快慰')+text.count('欢')+text.count('舒畅')+text.count('快乐')+text.count('快活')+text.count('欢乐')+text.count('畅快')+text.count('舒心')+text.count('舒坦')+text.count('欢娱')+text.count('如意')+text.count('喜悦')+text.count('顺心')+text.count('欢悦')+text.count('舒服')+text.count('爽心')+text.count('晓畅')+text.count('松快')+text.count('幸福')+text.count('惊喜')+text.count('欢愉')+text.count('称意')+text.count('得志')+text.count('情愿')+text.count('愿意')+text.count('欢喜')+text.count('振奋')+text.count('乐意')+text.count('留神')+text.count('乐于')+text.count('爱')+text.count('关怀')+text.count('偏爱')+text.count('珍爱')+text.count('珍惜')+text.count('神往')+text.count('痴迷')+text.count('喜爱')+text.count('器重')+text.count('娇宠')+text.count('溺爱')+text.count('珍视')+text.count('喜欢')+text.count('动心')+text.count('挂牵')+text.count('赞赏')+text.count('爱好')+text.count('满意')+text.count('羡慕')+text.count('赏识')+text.count('热爱')+text.count('钟爱')+text.count('眷恋')+text.count('关注')+text.count('赞同')+text.count('喜欢')+text.count('想')+text.count('挂心')+text.count('挂念')+text.count('惦念')+text.count('挂虑')+text.count('怀念')+text.count('关切')+text.count('关心')+text.count('惦念')+text.count('牵挂')+text.count('怜悯')+text.count('同情')+text.count('吝惜')+text.count('可惜')+text.count('怜惜')+text.count('感谢')+text.count('感激')+text.count('在乎')+text.count('操心')+text.count('愁')+text.count('闷')+text.count('苦')+text.count('哀怨')+text.count('悲恸')+text.count('悲痛')+text.count('哀伤')+text.count('惨痛')+text.count('沉重')+text.count('感伤')+text.count('悲壮')+text.count('酸辛')+text.count('伤心')+text.count('辛酸')+text.count('悲哀')+text.count('哀痛')+text.count('沉痛')+text.count('痛心')+text.count('悲凉')+text.count('悲凄')+text.count('伤感')+text.count('悲切')+text.count('哀戚')+text.count('悲伤')+text.count('心酸')+text.count('悲怆')+text.count('无奈')+text.count('苍凉')+text.count('不好过')+text.count('抑郁')+text.count('慌')+text.count('吓人')+text.count('畏怯')+text.count('紧张')+text.count('惶恐')+text.count('慌张')+text.count('惊骇')+text.count('恐慌')+text.count('慌乱')+text.count('心虚')+text.count('惊慌')+text.count('惶惑')+text.count('惊惶')+text.count('惊惧')+text.count('惊恐')+text.count('恐惧')+text.count('心慌')+text.count('害怕')+text.count('怕')+text.count('畏惧')+text.count('发慌')+text.count('发憷')+text.count('敬')+text.count('推崇')+text.count('尊敬')+text.count('拥护')+text.count('倚重')+text.count('崇尚')+text.count('尊崇')+text.count('敬仰')+text.count('敬佩')+text.count('尊重')+text.count('敬慕')+text.count('佩服')+text.count('景仰')+text.count('敬重')+text.count('景慕')+text.count('崇敬')+text.count('瞧得起')+text.count('崇奉')+text.count('钦佩')+text.count('崇拜')+text.count('孝敬')+text.count('激动')+text.count('来劲')+text.count('炽烈')+text.count('炽热')+text.count('冲动')+text.count('狂热')+text.count('激昂')+text.count('激动')+text.count('高亢')+text.count('亢奋')+text.count('带劲')+text.count('高涨')+text.count('高昂')+text.count('投入')+text.count('兴奋')+text.count('疯狂')+text.count('狂乱')+text.count('感动')+text.count('羞')+text.count('疚')+text.count('羞涩')+text.count('羞怯')+text.count('羞惭')+text.count('负疚')+text.count('窘')+text.count('窘促')+text.count('不过意')+text.count('惭愧')+text.count('不好意思')+text.count('害羞')+text.count('害臊')+text.count('困窘')+text.count('抱歉')+text.count('抱愧')+text.count('对不起')+text.count('羞愧')+text.count('对不住')+text.count('烦')+text.count('烦躁')+text.count('烦燥')+text.count('烦')+text.count('熬心')+text.count('糟心')+text.count('烦乱')+text.count('烦心')+text.count('烦人')+text.count('烦恼')+text.count('烦杂')+text.count('腻烦')+text.count('厌倦')+text.count('厌烦')+text.count('讨厌')+text.count('头疼')+text.count('急')+text.count('浮躁')+text.count('焦虑')+text.count('焦渴')+text.count('焦急')+text.count('焦躁')+text.count('焦炙')+text.count('心浮')+text.count('心焦')+text.count('揪心')+text.count('心急')+text.count('心切')+text.count('着急')+text.count('不安')+text.count('傲')+text.count('自傲')+text.count('骄横')+text.count('骄慢')+text.count('骄矜')+text.count('骄傲')+text.count('自负')+text.count('自信')+text.count('自豪')+text.count('自满')+text.count('自大')+text.count('狂')+text.count('炫耀')+text.count('吃惊')+text.count('诧异')+text.count('吃惊')+text.count('惊疑')+text.count('愕然')+text.count('惊讶')+text.count('惊奇')+text.count('骇怪')+text.count('骇异')+text.count('惊诧')+text.count('惊愕')+text.count('震惊')+text.count('奇怪')+text.count('怒')+text.count('愤怒')+text.count('忿恨')+text.count('激愤')+text.count('生气')+text.count('愤懑')+text.count('愤慨')+text.count('忿怒')+text.count('悲愤')+text.count('窝火')+text.count('暴怒')+text.count('不平')+text.count('火')+text.count('失望')+text.count('失望')+text.count('绝望')+text.count('灰心')+text.count('丧气')+text.count('低落')+text.count('心寒')+text.count('沮丧')+text.count('消沉')+text.count('颓丧')+text.count('颓唐')+text.count('低沉')+text.count('不满')+text.count('安心')+text.count('安宁')+text.count('闲雅')+text.count('逍遥')+text.count('闲适')+text.count('怡和')+text.count('沉静')+text.count('放松')+text.count('安心')+text.count('宽心')+text.count('自在')+text.count('放心')+text.count('恨')+text.count('恶')+text.count('看不惯')+text.count('痛恨')+text.count('厌恶')+text.count('恼恨')+text.count('反对')+text.count('捣乱')+text.count('怨恨')+text.count('憎恶')+text.count('歧视')+text.count('敌视')+text.count('愤恨')+text.count('嫉')+text.count('妒嫉')+text.count('妒忌')+text.count('嫉妒')+text.count('嫉恨')+text.count('眼红')+text.count('忌恨')+text.count('忌妒')+text.count('蔑视')+text.count('蔑视')+text.count('瞧不起')+text.count('怠慢')+text.count('轻蔑')+text.count('鄙夷')+text.count('鄙薄')+text.count('鄙视')+text.count('悔')+text.count('背悔')+text.count('后悔')+text.count('懊恼')+text.count('懊悔')+text.count('悔恨')+text.count('懊丧')+text.count('委屈')+text.count('委屈')+text.count('冤')+text.count('冤枉')+text.count('无辜')+text.count('谅')+text.count('体谅')+text.count('理解')+text.count('了解')+text.count('体贴')+text.count('信任')+text.count('信赖')+text.count('相信')+text.count('信服')+text.count('疑')+text.count('过敏')+text.count('怀疑')+text.count('疑心')+text.count('疑惑')+text.count('其他')+text.count('缠绵')+text.count('自卑')+text.count('自爱')+text.count('反感')+text.count('感慨')+text.count('动摇')+text.count('消魂')+text.count('痒痒')+text.count('为难')+text.count('解恨')+text.count('迟疑')+text.count('多情')+text.count('充实')+text.count('寂寞')+text.count('遗憾')+text.count('神情')+text.count('慧黠')+text.count('狡黠')+text.count('安详')+text.count('仓皇')+text.count('阴冷')+text.count('阴沉')+text.count('犹豫')+text.count('好')+text.count('坏')+text.count('棒')+text.count('一般')+text.count('差')+text.count('得当')+text.count('标准')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)     



emotion_result=[]
for corpus in sub_corpora: 
    emotion_result.append(emotion(corpus))
    
df['emotion'] = pd.Series(emotion_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 12 classical syntax
def classical(text_type):
    def raw(text_type): 
        return text_type.count('备受')+text_type.count('言必称')+text_type.count('并存')+text_type.count('不得而')+text_type.count('抑且')+text_type.count('不特')+text_type.count('不外乎')+text_type.count('且')+text_type.count('不外乎')+text_type.count('不相')+text_type.count('中不乏')+text_type.count('不啻')+text_type.count('称之为')+text_type.count('称之')+text_type.count('充其量')+text_type.count('出于')+text_type.count('处于')+text_type.count('不次于')+text_type.count('从属于')+text_type.count('从中')+text_type.count('得自于')+text_type.count('得力于')+text_type.count('予以')+text_type.count('给予')+text_type.count('加以')+text_type.count('深具')+text_type.count('之能事')+text_type.count('发轫于')+text_type.count('凡此')+text_type.count('大抵')+text_type.count('凡')+text_type.count('所能及')+text_type.count('所可比')+text_type.count('非但')+text_type.count('庶可')+text_type.count('之故')+text_type.count('工于')+text_type.count('苟')+text_type.count('顾')+text_type.count('广为')+text_type.count('果')+text_type.count('核以')+text_type.count('何其')+text_type.count('或可')+text_type.count('跻身')+text_type.count('跻于')+text_type.count('不日即')+text_type.count('藉')+text_type.count('之大成')+text_type.count('再加')+text_type.count('略加')+text_type.count('详加')+text_type.count('以俱来')+text_type.count('见胜')+text_type.count('见长')+text_type.count('兼')+text_type.count('渐次')+text_type.count('化')+text_type.count('混同于')+text_type.count('归之于')+text_type.count('推广到')+text_type.count('名之为')+text_type.count('引为')+text_type.count('矣')+text_type.count('较')+text_type.count('借以')+text_type.count('尽其')+text_type.count('略陈己见')+text_type.count('而言')+text_type.count('而论')+text_type.count('决定于')+text_type.count('之先河')+text_type.count('苦不能')+text_type.count('莫不是')+text_type.count('乃')+text_type.count('泥于')+text_type.count('偏于')+text_type.count('颇有')+text_type.count('岂不')+text_type.count('岂可')+text_type.count('乎')+text_type.count('哉')+text_type.count('起源于')+text_type.count('何况')+text_type.count('切于')+text_type.count('取信于')+text_type.count('如')+text_type.count('则')+text_type.count('若')+text_type.count('岂')+text_type.count('舍')+text_type.count('甚于')+text_type.count('时年')+text_type.count('时值')+text_type.count('使之')+text_type.count('有别于')+text_type.count('倍加')+text_type.count('所在')+text_type.count('示人以')+text_type.count('随致')+text_type.count('之所以')+text_type.count('所以然')+text_type.count('所verb者')+text_type.count('无所')+text_type.count('有所')+text_type.count('皆指')+text_type.count('所引致')+text_type.count('罕为')+text_type.count('鲜为')+text_type.count('多为')+text_type.count('唯')+text_type.count('尚未')+text_type.count('无一不')+text_type.count('无不能')+text_type.count('无从')+text_type.count('可见')+text_type.count('毋宁')+text_type.count('无宁')+text_type.count('务')+text_type.count('系于')+text_type.count('仅限于')+text_type.count('方能')+text_type.count('需')+text_type.count('须')+text_type.count('许之为')+text_type.count('一改')+text_type.count('一变')+text_type.count('与否')+text_type.count('业已')+text_type.count('不以为然')+text_type.count('为能')+text_type.count('为多')+text_type.count('为最')+text_type.count('以期')+text_type.count('不宜')+text_type.count('宜于')+text_type.count('异于')+text_type.count('益见')+text_type.count('抑或')+text_type.count('故')+text_type.count('之便')+text_type.count('应推')+text_type.count('着手')+text_type.count('着眼')+text_type.count('可证')+text_type.count('可知')+text_type.count('可见')+text_type.count('而成')+text_type.count('有不')+text_type.count('有所')+text_type.count('有待于')+text_type.count('有赖于')+text_type.count('有助于')+text_type.count('有进于')+text_type.count('之分')+text_type.count('之别')+text_type.count('多有')+text_type.count('囿于')+text_type.count('与之')+text_type.count('同/共')+text_type.count('同为')+text_type.count('欲')+text_type.count('必')+text_type.count('喻之')+text_type.count('曰')+text_type.count('之际')+text_type.count('已然')+text_type.count('在于')+text_type.count('则')+text_type.count('者')+text_type.count('即是')+text_type.count('皆是')+text_type.count('云者')+text_type.count('者有之')+text_type.count('首属')+text_type.count('首推')+text_type.count('莫过于')+text_type.count('之')+text_type.count('之于')+text_type.count('置身于')+text_type.count('转而')+text_type.count('自')+text_type.count('自况')+text_type.count('自命')+text_type.count('自诩')+text_type.count('自认')+text_type.count('自居')+text_type.count('自许')+text_type.count('以降')+text_type.count('足以')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 


classical_result=[]
for corpus in sub_corpora: 
    classical_result.append(classical(corpus))
    
df['classical_syntax'] = pd.Series(classical_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 18 disyllabic words 
def disyllabic(text_type):
    def raw(text_type): 
        return text_type.count('购买')+text_type.count('具有')+text_type.count('在于')+text_type.count('寻找')+text_type.count('获得')+text_type.count('询问')+text_type.count('进入')+text_type.count('等候')+text_type.count('安定')+text_type.count('安装')+text_type.count('办理')+text_type.count('保持')+text_type.count('保留')+text_type.count('保卫')+text_type.count('保障')+text_type.count('报道')+text_type.count('暴露')+text_type.count('爆发')+text_type.count('被迫')+text_type.count('必然')+text_type.count('必修')+text_type.count('必要')+text_type.count('避免')+text_type.count('编制')+text_type.count('变动')+text_type.count('变革')+text_type.count('辩论')+text_type.count('表达')+text_type.count('表示')+text_type.count('表演')+text_type.count('并肩')+text_type.count('补习')+text_type.count('不断')+text_type.count('不时')+text_type.count('不住')+text_type.count('布置')+text_type.count('采取')+text_type.count('采用')+text_type.count('参考')+text_type.count('测量')+text_type.count('测试')+text_type.count('测验')+text_type.count('颤动')+text_type.count('抄写')+text_type.count('陈列')+text_type.count('成立')+text_type.count('成为')+text_type.count('承担')+text_type.count('承认')+text_type.count('持枪')+text_type.count('充分')+text_type.count('充满')+text_type.count('充实')+text_type.count('仇恨')+text_type.count('出版')+text_type.count('处于')+text_type.count('处处')+text_type.count('传播')+text_type.count('传达')+text_type.count('创立')+text_type.count('次要')+text_type.count('匆忙')+text_type.count('从容')+text_type.count('从事')+text_type.count('促进')+text_type.count('摧毁')+text_type.count('达成')+text_type.count('达到')+text_type.count('打扫')+text_type.count('大力')+text_type.count('大有')+text_type.count('担任')+text_type.count('导致')+text_type.count('到达')+text_type.count('等待')+text_type.count('等候')+text_type.count('奠定')+text_type.count('雕刻')+text_type.count('调查')+text_type.count('动员')+text_type.count('独自')+text_type.count('端正')+text_type.count('锻炼')+text_type.count('夺取')+text_type.count('发表')+text_type.count('发动')+text_type.count('发挥')+text_type.count('发射')+text_type.count('发生')+text_type.count('发行')+text_type.count('发扬')+text_type.count('发展')+text_type.count('反抗')+text_type.count('防守')+text_type.count('防御')+text_type.count('防止')+text_type.count('防治')+text_type.count('非法')+text_type.count('废除')+text_type.count('粉碎')+text_type.count('丰富')+text_type.count('封锁')+text_type.count('符合')+text_type.count('负担')+text_type.count('负责')+text_type.count('复述')+text_type.count('复习')+text_type.count('复印')+text_type.count('复杂')+text_type.count('复制')+text_type.count('富有')+text_type.count('改编')+text_type.count('改革')+text_type.count('改进')+text_type.count('改良')+text_type.count('改善')+text_type.count('改正')+text_type.count('干涉')+text_type.count('敢于')+text_type.count('高大')+text_type.count('高度')+text_type.count('高速')+text_type.count('格外')+text_type.count('给以')+text_type.count('更加')+text_type.count('公开')+text_type.count('公然')+text_type.count('巩固')+text_type.count('贡献')+text_type.count('共同')+text_type.count('构成')+text_type.count('购买')+text_type.count('观测')+text_type.count('观察')+text_type.count('观看')+text_type.count('贯彻')+text_type.count('灌溉')+text_type.count('光临')+text_type.count('规划')+text_type.count('合成')+text_type.count('合法')+text_type.count('宏伟')+text_type.count('缓和')+text_type.count('缓缓')+text_type.count('回答')+text_type.count('汇报')+text_type.count('混淆')+text_type.count('活跃')+text_type.count('获得')+text_type.count('基本')+text_type.count('集合')+text_type.count('集中')+text_type.count('极为')+text_type.count('即将')+text_type.count('计划')+text_type.count('记载')+text_type.count('继承')+text_type.count('加工')+text_type.count('加紧')+text_type.count('加速')+text_type.count('加以')+text_type.count('驾驶')+text_type.count('歼灭')+text_type.count('坚定')+text_type.count('减轻')+text_type.count('检验')+text_type.count('简直')+text_type.count('建立')+text_type.count('建造')+text_type.count('建筑')+text_type.count('交换')+text_type.count('交流')+text_type.count('结束')+text_type.count('竭力')+text_type.count('解决')+text_type.count('解释')+text_type.count('紧急')+text_type.count('紧密')+text_type.count('谨慎')+text_type.count('进军')+text_type.count('进攻')+text_type.count('进入')+text_type.count('进行')+text_type.count('尽力')+text_type.count('禁止')+text_type.count('精彩')+text_type.count('进过')+text_type.count('经历')+text_type.count('经受')+text_type.count('经营')+text_type.count('竞争')+text_type.count('竟然')+text_type.count('纠正')+text_type.count('举办')+text_type.count('举行')+text_type.count('具备')+text_type.count('具体')+text_type.count('具有')+text_type.count('开办')+text_type.count('开动')+text_type.count('开发')+text_type.count('开明')+text_type.count('开辟')+text_type.count('开枪')+text_type.count('开设')+text_type.count('开展')+text_type.count('抗议')+text_type.count('克服')+text_type.count('刻苦')+text_type.count('空前')+text_type.count('扩大')+text_type.count('来自')+text_type.count('滥用')+text_type.count('朗读')+text_type.count('力求')+text_type.count('力争')+text_type.count('连接')+text_type.count('列举')+text_type.count('流传')+text_type.count('垄断')+text_type.count('笼罩')+text_type.count('轮流')+text_type.count('掠夺')+text_type.count('满腔')+text_type.count('盲目')+text_type.count('猛烈')+text_type.count('猛然')+text_type.count('梦想')+text_type.count('勉强')+text_type.count('面临')+text_type.count('明明')+text_type.count('明确')+text_type.count('难以')+text_type.count('扭转')+text_type.count('拍摄')+text_type.count('排列')+text_type.count('攀登')+text_type.count('炮打')+text_type.count('赔偿')+text_type.count('评价')+text_type.count('评论')+text_type.count('赔偿')+text_type.count('评价')+text_type.count('评论')+text_type.count('破坏')+text_type.count('普遍')+text_type.count('普及')+text_type.count('起源')+text_type.count('签订')+text_type.count('强调')+text_type.count('抢夺')+text_type.count('切实')+text_type.count('侵略')+text_type.count('侵入')+text_type.count('轻易')+text_type.count('取得')+text_type.count('全部')+text_type.count('全面')+text_type.count('燃烧')+text_type.count('热爱')+text_type.count('忍受')+text_type.count('仍旧')+text_type.count('日益')+text_type.count('如同')+text_type.count('散布')+text_type.count('丧失')+text_type.count('设法')+text_type.count('设立')+text_type.count('实施')+text_type.count('实现')+text_type.count('实行')+text_type.count('实验')+text_type.count('适合')+text_type.count('试验')+text_type.count('收集')+text_type.count('收缩')+text_type.count('树立')+text_type.count('束缚')+text_type.count('思考')+text_type.count('思念')+text_type.count('思索')+text_type.count('丝毫')+text_type.count('四处')+text_type.count('饲养')+text_type.count('损害')+text_type.count('损坏')+text_type.count('损失')+text_type.count('缩短')+text_type.count('缩小')+text_type.count('贪图')+text_type.count('谈论')+text_type.count('探索')+text_type.count('逃避')+text_type.count('提倡')+text_type.count('提供')+text_type.count('提前')+text_type.count('体现')+text_type.count('调节')+text_type.count('调整')+text_type.count('停止')+text_type.count('统一')+text_type.count('突破')+text_type.count('推迟')+text_type.count('推动')+text_type.count('推进')+text_type.count('脱离')+text_type.count('歪曲')+text_type.count('完善')+text_type.count('万分')+text_type.count('万万')+text_type.count('危害')+text_type.count('违背')+text_type.count('违反')+text_type.count('维持')+text_type.count('维护')+text_type.count('围绕')+text_type.count('伟大')+text_type.count('位于')+text_type.count('污染')+text_type.count('无比')+text_type.count('无法')+text_type.count('无穷')+text_type.count('无限')+text_type.count('武装')+text_type.count('吸取')+text_type.count('袭击')+text_type.count('喜爱')+text_type.count('显示')+text_type.count('限制')+text_type.count('陷入')+text_type.count('相互')+text_type.count('详细')+text_type.count('响应')+text_type.count('享受')+text_type.count('象征')+text_type.count('消除')+text_type.count('消耗')+text_type.count('小心')+text_type.count('写作')+text_type.count('辛勤')+text_type.count('修改')+text_type.count('修正')+text_type.count('修筑')+text_type.count('选择')+text_type.count('严格')+text_type.count('严禁')+text_type.count('严厉')+text_type.count('严密')+text_type.count('严肃')+text_type.count('研制')+text_type.count('延长')+text_type.count('掩盖')+text_type.count('养成')+text_type.count('一经')+text_type.count('依法')+text_type.count('依旧')+text_type.count('依然')+text_type.count('抑制')+text_type.count('应用')+text_type.count('永远')+text_type.count('踊跃')+text_type.count('游览')+text_type.count('予以')+text_type.count('遇到')+text_type.count('预防')+text_type.count('预习')+text_type.count('阅读')+text_type.count('运用')+text_type.count('再三')+text_type.count('遭到')+text_type.count('遭受')+text_type.count('遭遇')+text_type.count('增加')+text_type.count('增进')+text_type.count('增强')+text_type.count('占领')+text_type.count('占有')+text_type.count('战胜')+text_type.count('掌握')+text_type.count('照例')+text_type.count('镇压')+text_type.count('征服')+text_type.count('征求')+text_type.count('争夺')+text_type.count('争论')+text_type.count('整顿')+text_type.count('证明')+text_type.count('直到')+text_type.count('执行')+text_type.count('制定')+text_type.count('制订')+text_type.count('制造')+text_type.count('治疗')+text_type.count('中断')+text_type.count('重大')+text_type.count('专心')+text_type.count('转入')+text_type.count('转移')+text_type.count('装备')+text_type.count('装饰')+text_type.count('追求')+text_type.count('自学')+text_type.count('综合')+text_type.count('总结')+text_type.count('阻止')+text_type.count('钻研')+text_type.count('遵守')+text_type.count('左右')
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2)      



disyllabic_result=[]
for corpus in sub_corpora: 
    disyllabic_result.append(disyllabic(corpus))
    
df['disyllabic_words'] = pd.Series(disyllabic_result)
df.to_csv(folder + 'linguistic_features.csv')



#feature 28 HSK core vocabulary level 1 150 words 

HSK1=['爱', '八', '爸爸', '杯子', '北京', '本', '不', '不客气', '菜', '茶', '吃', '出租车', '打电话', '大', '的', '点', '电脑', '电视', '电影', '东西', '都', '读', '对不起', '多', '多少', '儿子', '二', '饭店', '飞机', '分钟', '高兴', '个', '工作', '狗', '汉语', '好', '号', '喝', '和', '很', '后面', '回', '会', '几', '家', '叫', '今天', '九', '开', '看', '看见', '块', '来', '老师', '了', '冷', '里', '六', '妈妈', '吗', '买', '猫', '没关系', '没有', '米饭', '名字', '明天', '哪', '哪儿', '那', '呢', '能', '你', '年', '女儿', '朋友', '漂亮', '苹果', '七', '前面', '钱', '请', '去', '热', '人', '认识', '三', '商店', '上', '上午', '少', '谁', '什么', '十', '时候', '是', '书', '水', '水果', '睡觉', '说', '四', '岁', '他', '她', '太', '天气', '听', '同学', '喂', '我', '我们', '五', '喜欢', '下', '下午', '下雨', '先生', '现在', '想', '小', '小姐', '些', '写', '谢谢', '星期', '学生', '学习', '学校', '一', '一点儿', '衣服', '医生', '医院', '椅子', '有', '月', '再见', '在', '怎么', '怎么样', '这', '中国', '中午', '住', '桌子', '字', '昨天', '坐', '做']

def hsk1(text_type):
    def raw(text_type): 
        return len([x for x in text_type if x in HSK1])
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



hsk1_result=[]
for corpus in sub_corpora: 
    hsk1_result.append(hsk1(corpus))
    
df['HSK_1'] = pd.Series(hsk1_result)
df.to_csv(folder + 'linguistic_features.csv')



HSK3=['阿姨', '啊', '矮', '爱', '爱好', '安静', '把', '吧', '白', '百', '班', '搬', '办法', '办公室', '半', '帮忙', '帮助', '包', '饱', '报纸', '北方', '被', '鼻子', '比', '比较', '比赛', '笔记本', '必须', '变化', '别', '别人', '宾馆', '冰箱', '不但', '而且', '菜单', '参加', '草', '层', '差', '长', '唱歌', '超市', '衬衫', '成绩', '城市', '迟到', '出', '除了', '穿', '船', '春', '词典', '次', '聪明', '从', '错', '打篮球', '打扫', '打算', '大家', '带', '担心', '蛋糕', '当然', '到', '地', '得', '灯', '等', '地方', '地铁', '地图', '弟弟', '第一', '电梯', '电子邮件', '东', '冬', '懂', '动物', '短', '段', '锻炼', '对', '多么', '饿', '耳朵', '发', '发烧', '发现', '方便', '房间', '放', '放心', '非常', '分', '服务员', '附近', '复习', '干净', '感冒', '感兴趣', '刚才', '高', '告诉', '哥哥', '个子', '给', '根据', '跟', '更', '公共汽车', '公斤', '公司', '公园', '故事', '刮风', '关', '关系', '关心', '关于', '贵', '国家', '过', '过去', '还', '还是', '孩子', '害怕', '好吃', '黑', '黑板', '红', '后来', '护照', '花', '画', '坏', '欢迎', '环境', '换', '黄河', '回答', '会议', '火车站', '或者', '几乎', '机场', '机会', '鸡蛋', '极', '记得', '季节', '检查', '简单', '见面', '件', '健康', '讲', '教', '角', '脚', '教室', '接', '街道', '节目', '节日', '结婚', '结束', '姐姐', '解决', '介绍', '借', '进', '近', '经常', '经过', '经理', '久', '旧', '就', '句子', '决定', '觉得', '咖啡', '开始', '考试', '可爱', '可能', '可以', '渴', '刻', '客人', '课', '空调', '口', '哭', '裤子', '快', '快乐', '筷子', '蓝', '老', '累', '离', '离开', '礼物', '历史', '脸', '练习', '两', '辆', '聊天', '了解', '邻居', '零', '留学', '楼', '路', '旅游', '绿', '马', '马上', '卖', '满意', '慢', '忙', '帽子', '每', '妹妹', '门', '米', '面包', '面条', '明白', '拿', '奶奶', '男', '南', '难', '难过', '年级', '年轻', '鸟', '您', '牛奶', '努力', '女', '爬山', '盘子', '旁边', '胖', '跑步', '皮鞋', '啤酒', '便宜', '票', '瓶子', '妻子', '其实', '其他', '奇怪', '骑', '起床', '起飞', '起来', '千', '铅笔', '清楚', '晴', '请假', '秋', '去年', '裙子', '然后', '让', '热情', '认为', '认真', '日', '容易', '如果', '伞', '上班', '上网', '谁', '身体', '生病', '生气', '生日', '声音', '时间', '世界', '事情', '试', '手表', '手机', '瘦', '叔叔', '舒服', '树', '数学', '刷牙', '双', '水平', '说话', '司机', '送', '虽然', '但是', '它', '她', '太阳', '特别', '疼', '踢足球', '提高', '题', '体育', '甜', '条', '跳舞', '同事', '同意', '头发', '突然', '图书馆', '腿', '外', '完', '完成', '玩', '晚上', '碗', '万', '往', '忘记', '为', '为了', '为什么', '位', '文化', '问', '问题', '西', '西瓜', '希望', '习惯', '洗', '洗手间', '洗澡', '夏', '先', '相信', '香蕉', '向', '像', '小时', '小心', '校长', '笑', '新', '新闻', '新鲜', '信用卡', '行李箱', '姓', '熊猫', '休息', '需要', '选择', '雪', '颜色', '眼睛', '羊肉', '要求', '药', '要', '爷爷', '也', '一般', '一边', '一定', '一共', '一会儿', '一起', '一下', '一样', '一直', '已经', '以前', '意思', '因为', '所以', '阴', '音乐', '银行', '饮料', '应该', '影响', '用', '游戏', '游泳', '有名', '又', '右边', '鱼', '遇到', '元', '远', '愿意', '月亮', '越', '运动', '再', '早上', '站', '张', '丈夫', '着急', '找', '照顾', '照片', '照相机', '着', '真', '正在', '只', '只有', '才', '中间', '中文', '终于', '种', '重要', '周末', '主要', '注意', '准备', '自己', '自行车', '总是', '走', '嘴', '最', '最后', '最近', '左边', '作业']


#feature 29 HSK core vocabulary level 3 (150-600), 450 words 
def hsk3(text_type):
    def raw(text_type): 
        return len([x for x in text_type if x in HSK3])
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



hsk3_result=[]
for corpus in sub_corpora: 
    hsk3_result.append(hsk3(corpus))
    
df['HSK_3'] = pd.Series(hsk3_result)
df.to_csv(folder + 'linguistic_features.csv')



# feature 27 honorifics


honorifics=['千金', '相公', '姑姥爷', '伯伯', '伯父', '伯母', '大伯', '大哥', '大姐', '大妈', '大爷', '大嫂', '嫂夫人', '大婶儿', '大叔', '大姨', '哥', '姐', '大娘', '妈妈', '奶 奶', '爷爷', '姨', '老伯', '老兄', '老爹', '老大爷', '老爷爷', '老太太', '老奶奶', '老大娘', '老板', '老公', '老婆婆', '老前辈', '老人家', '老师', '老师傅', '老寿星', '老太爷', '老翁', '老爷子', '老丈', '老总', '大驾', '夫人', '高徒', '高足', '官人', '贵客', '贵人', '嘉宾', '列位', '男士', '女士', '女主 人', '前辈', '台驾', '太太', '先生', '贤契', '贤人', '贤士', '先哲', '小姐', '学长', '爷', '诸位', '足下', '师傅', '师母', '师娘', '人士', '长老', '禅师', '船老大', '大师', '大师傅', '大王', '恩师', '法师', '法王', '佛爷', '夫子', '父母官', '国父', '麾下', '教授', '武师', '千 岁', '孺人', '圣母', '圣人', '师父', '王尊', '至尊', '座', '少奶奶', '少爷', '金枝玉叶', '工程师', '高级工程师', '经济师', '讲师', '教授', '副教授', '教师', '老师', '国家主席', '国家总理', '部长', '厅长', '市长', '局长', '科长', '校长', '烈士', '先烈', '先哲', '荣誉军人', '陛下', '殿下', '阁下', '阿公', '阿婆', '大人', '公', '公公', '娘子', '婆婆', '丈人', '师长', '义士', '勇士', '志士', '壮士', '学生', '兄弟', '小弟', '弟', '妹', '儿子', '女儿']



def honor(text_type):
    def raw(text_type): 
        return len([x for x in text_type if x in honorifics])
    def normalized(text_type): 
        return raw(text_type) / len(text_type)
    return round(normalized (text_type) * 1000, 2) 



honor_result=[]
for corpus in sub_corpora: 
    honor_result.append(honor(corpus))
    
df['honorifics'] = pd.Series(honor_result)
df.to_csv(folder + 'linguistic_features.csv')


#feature 54 unique words ratio 
def unique(text): 
    return round((len([x for x in text if text.count(x)==1]) / len(text))*1000, 2)


unique_result=[]
for corpus in sub_corpora: 
    unique_result.append(unique(corpus))



df['unique'] = pd.Series(unique_result)
df.to_csv(folder + 'linguistic_features.csv')

