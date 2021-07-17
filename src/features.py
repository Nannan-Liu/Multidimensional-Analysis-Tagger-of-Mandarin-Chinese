#!/usr/bin/python3
# coding=UTF-8 #note capitalisation

#dependencies
import os
import csv
import pynlpir #use ICTCLAS to segment
import nltk 
from nltk.corpus import CategorizedPlaintextCorpusReader 
import pandas as pd
import statistics #for deviation
import re

#input your txt file folder
user_input = input("Enter the path of your txt file(s): ")
     
assert os.path.exists(user_input), "MulDi Chinese did not find the files at, "+str(user_input)

print("Found your path")

folder=str(user_input)

#use txt file names as dataframe index
files=[]
for file in os.listdir(folder):
    if file.endswith(".txt"):
        files.append(file)
        file_names = sorted([f.replace('.txt', '') for f in files])

#create stats file 
df = pd.DataFrame(file_names, columns=["text"])

print("Feature stats file created")

#convert your text into a corpus
corpus = CategorizedPlaintextCorpusReader(
    folder,
    r'(?!\.).*\.txt',
    cat_pattern=os.path.join(r'(neg|pos)', '.*',),
    encoding='utf-8')

#see example words
print("Your file starts with the following words: ", corpus.words())

files=corpus.fileids()

#create corpora
corpora=[]
for file in files: 
    sub_corpora=corpus.raw(file)
    corpora.append(sub_corpora)

#segment words
pynlpir.open()

sub_corpora=[]
for corpus in corpora: 
    sub_corpus=pynlpir.segment(corpus, pos_tagging=False)
    sub_corpora.append(sub_corpus)

#close pynlpir to free allocated memory
pynlpir.close()


#-----------------------------segmented Features-------------------------------------- 


#Feature 1 amplifiers AMP
amplifier=['非常', '大大', '十分', '真的', '真', '特别', '很', '最', '肯定', '挺', '顶', '极', '极为', '极其', '极度', '万分', '格外', '分外', '更', '更加', '更为', '尤其', '太', '过于', '老', '怪', '相当', '颇', '颇为', '有点儿', '有些', '最为', '还', '越发', '越加', '愈加', '稍', '稍微', '稍稍', '略', '略略', '略微', '比较', '较', '暴', '超', '恶', '怒', '巨', '粉', '奇', '很大', '相当', '完全', '显著', '总是', '根本', '一定']

#define count and standardise function
def amplifiers(text):
    def raw(text): 
        return len([x for x in text if x in amplifier])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


amplifiers_result=[]
for corpus in sub_corpora: 
    amplifiers_result.append(amplifiers(corpus))
    
df['AMP'] = pd.Series(amplifiers_result)

print("Standardised frequencies of amplifiers (AMP) written.")


#Feature 2 average word length
#strip quotation marks first 
lists=[]
for corpus in sub_corpora: 
    l=list(corpus)
    lists.append(l)


def awl(text): 
    return round ((sum(len(word) for word in text) / len(text)), 2)

awl_result=[]
for l in lists: 
    awl_result.append(awl(l))
    
df['AWL'] = pd.Series(awl_result)

print("Standardised average word length (AWL) written.")


#Feature 3 mean sentence length
def asl(text): 
    sentences = [[]]
    ends = set('。？!……——')
    for word in text:
        if word in ends: sentences.append([])
        else: sentences[-1].append(word)
    if sentences[0]:
        if not sentences[-1]: sentences.pop()
        return round (sum(len(s) for s in sentences)/len(sentences), 2)


asl_result=[]
for l in lists: 
    asl_result.append(asl(l))

df['ASL'] = pd.Series(asl_result)

print("Standardised average sentence length (ASL) written.")


#Feature 4 standard deviation of sentence length 

def asl_std(text): 
    sentences = [[]]
    ends = set('。？!……——')
    for word in text:
        if word in ends: sentences.append([])
        else: sentences[-1].append(word)
    if sentences[0]:
        if not sentences[-1]: sentences.pop()
        return round(statistics.stdev(len(s) for s in sentences), 2)


asl_std_result=[]
for l in lists: 
    for corpus in sub_corpora: 
        asl_std_result.append(asl_std(l))

df['ASL_std'] = pd.Series(asl_std_result)

print("Standard deviation of average sentence length (ASL_STD) written.")

#Feature 5 average clause length
def acl(text): 
    sentences = [[]]
    ends = set('，：；。？!……——')
    for word in text:
        if word in ends: sentences.append([])
        else: sentences[-1].append(word)
    if sentences[0]:
        if not sentences[-1]: sentences.pop()
        return round (sum(len(s) for s in sentences)/len(sentences), 2)


acl_result=[]
for corpus in sub_corpora: 
    acl_result.append(acl(corpus))

df['ACL'] = pd.Series(acl_result)

print("Standardised average clause length written.")


#Feature 6 downtoners DWNT
downtoners=['一点', '一点儿', '有点', '有点儿', '稍', '稍微', '一些', '有些']
def dwnt(text):
    def raw(text): 
        return len([x for x in text if x in downtoners])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


dwnt_result=[]
for corpus in sub_corpora: 
    dwnt_result.append(dwnt(corpus))
    
df['DWNT'] = pd.Series(dwnt_result)

print("Standardised frequnecies of downtoner written.")


#Feature 7 FPP1 first person pronoun
def fpp1(text):
    def raw(text): 
        return text.count('我')+text.count('我们')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


fpp1_result=[]
for corpus in sub_corpora: 
    fpp1_result.append(fpp1(corpus))
    
df['FPP'] = pd.Series(fpp1_result)

print("Standardised frequencies of first person pronouns (FPPs) written.")

#Feature 8 SPP2 second person pronoun
def spp2(text):
    def raw(text): 
        return sum(map(str(text).count, ['你', '你们', '您','您们']))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


spp2_result=[]
for corpus in sub_corpora: 
    spp2_result.append(spp2(corpus))
    
df['SPP'] = pd.Series(spp2_result)

print("Standardised frequencies of second person pronouns (SPPs) written.")

#Feature 9 hedges HDG
hedges=['可能', '可以', '也许', '较少', '一些', '多个', '多为', '基本', '主要', '类似', '不少']
def hdg(text):
    def raw(text): 
        return len([x for x in text if x in hedges])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


hdg_result=[]
for corpus in sub_corpora: 
    hdg_result.append(hdg(corpus))
    
df['HDG'] = pd.Series(hdg_result)

print("Standardised frequencies of hedges (HDGs) written.")

#Feature 10 INPR indefinite pronouns 无定代词
indefinites=['任何', '谁', '大家', '某', '有人', '有个', '什么']
def inpr(text):
    def raw(text): 
        return len([x for x in text if x in indefinites])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 

inpr_result=[]
for corpus in sub_corpora: 
    inpr_result.append(inpr(corpus))
    
df['INPR'] = pd.Series(inpr_result)

print("Standardised frequencies of indefinite pronouns (INPRs) written.")


#Feature 11 SMP (seem, appear)
smps=['好像', '好象', '似乎', '貌似']
def smp(text):
    def raw(text): 
        return len([x for x in text if x in smps])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 

smp_result=[]
for corpus in sub_corpora: 
    smp_result.append(smp(corpus))
    
df['SMP'] = pd.Series(smp_result)

print("Standardised frequencies of seem/appear words (SMPs) written.")


#Feature 12 third person pronouns
tpp3s=['她', '他', '它', '她们', '他们', '它们']
def tpp3(text):
    def raw(text): 
        return len([x for x in text if x in tpp3s])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


tpp3_result=[]
for corpus in sub_corpora: 
    tpp3_result.append(tpp3(corpus))
    
df['TPP'] = pd.Series(tpp3_result)

print("Standardised frequencies of third person pronouns (TPPs) written.")

#Feature 13 emotion words
emotion_words=['烦恼', '不幸', '痛苦', '苦', '快乐', '忍', '喜', '乐', '称心', '痛快', \
'得意', '欣慰', '高兴', '愉悦', '欣喜', '欢欣', '可意', '乐', '可心', '欢畅', \
'开心', '康乐', '欢快', '快慰', '欢', '舒畅', '快乐', '快活', '欢乐', '畅快', \
'舒心', '舒坦', '欢娱', '如意', '喜悦', '顺心', '欢悦', '舒服', '爽心', '晓畅', \
'松快', '幸福', '惊喜', '欢愉', '称意', '得志', '情愿', '愿意', '欢喜', '振奋', \
'乐意', '留神', '乐于', '爱', '关怀', '偏爱', '珍爱', '珍惜', '神往', '痴迷', '喜爱',\
'器重', '娇宠', '溺爱', '珍视', '喜欢', '动心', '挂牵', '赞赏', '爱好', '满意', '羡慕',\
'赏识', '热爱', '钟爱', '眷恋', '关注', '赞同', '喜欢', '想', '挂心', '挂念', '惦念', \
'挂虑', '怀念', '关切', '关心', '惦念', '牵挂', '怜悯', '同情', '吝惜', '可惜', '怜惜', \
'感谢', '感激', '在乎', '操心', '愁', '闷', '苦', '哀怨', '悲恸', '悲痛', '哀伤', '惨痛', \
'沉重', '感伤', '悲壮', '酸辛', '伤心', '辛酸', '悲哀', '哀痛', '沉痛', '痛心', '悲凉', \
'悲凄', '伤感', '悲切', '哀戚', '悲伤', '心酸', '悲怆', '无奈', '苍凉', '不好过', '抑郁', \
'慌', '吓人', '畏怯', '紧张', '惶恐', '慌张', '惊骇', '恐慌', '慌乱', '心虚', '惊慌', \
'惶惑', '惊惶', '惊惧', '惊恐', '恐惧', '心慌', '害怕', '怕', '畏惧', '发慌', '发憷', \
'敬', '推崇', '尊敬', '拥护', '倚重', '崇尚', '尊崇', '敬仰', '敬佩', '尊重', '敬慕', \
'佩服', '景仰', '敬重', '景慕', '崇敬', '瞧得起', '崇奉', '钦佩', '崇拜', '孝敬', '激动', \
'来劲', '炽烈', '炽热', '冲动', '狂热', '激昂', '激动', '高亢', '亢奋', '带劲', '高涨', \
'高昂', '投入', '兴奋', '疯狂', '狂乱', '感动', '羞', '疚', '羞涩', '羞怯', '羞惭', '负疚', \
'窘', '窘促', '不过意', '惭愧', '不好意思', '害羞', '害臊', '困窘', '抱歉', '抱愧', '对不起', \
'羞愧', '对不住', '烦', '烦躁', '烦燥', '烦', '熬心', '糟心', '烦乱', '烦心', '烦人', '烦恼', \
'烦杂', '腻烦', '厌倦', '厌烦', '讨厌', '头疼', '急', '浮躁', '焦虑', '焦渴', '焦急', '焦躁', \
'焦炙', '心浮', '心焦', '揪心', '心急', '心切', '着急', '不安', '傲', '自傲', '骄横', '骄慢', \
'骄矜', '骄傲', '自负', '自信', '自豪', '自满', '自大', '狂', '炫耀', '吃惊', '诧异', '吃惊', \
'惊疑', '愕然', '惊讶', '惊奇', '骇怪', '骇异', '惊诧', '惊愕', '震惊', '奇怪', '怒', '愤怒', \
'忿恨', '激愤', '生气', '愤懑', '愤慨', '忿怒', '悲愤', '窝火', '暴怒', '不平', '火', '失望', \
'失望', '绝望', '灰心', '丧气', '低落', '心寒', '沮丧', '消沉', '颓丧', '颓唐', '低沉', '不满', \
'安心', '安宁', '闲雅', '逍遥', '闲适', '怡和', '沉静', '放松', '安心', '宽心', '自在', '放心', \
'恨', '恶', '看不惯', '痛恨', '厌恶', '恼恨', '反对', '捣乱', '怨恨', '憎恶', '歧视', '敌视', \
'愤恨', '嫉', '妒嫉', '妒忌', '嫉妒', '嫉恨', '眼红', '忌恨', '忌妒', '蔑视', '蔑视', '瞧不起', \
'怠慢', '轻蔑', '鄙夷', '鄙薄', '鄙视', '悔', '背悔', '后悔', '懊恼', '懊悔', '悔恨', '懊丧', \
'委屈', '委屈', '冤', '冤枉', '无辜', '谅', '体谅', '理解', '了解', '体贴', '信任', '信赖', \
'相信', '信服', '疑', '过敏', '怀疑', '疑心', '疑惑', '其他', '缠绵', '自卑', '自爱', '反感', \
'感慨', '动摇', '消魂', '痒痒', '为难', '解恨', '迟疑', '多情', '充实', '寂寞', '遗憾', '神情', \
'慧黠', '狡黠', '安详', '仓皇', '阴冷', '阴沉', '犹豫', '好', '坏', '棒', '一般', '差', \
'得当', '标准']

def emotion(text): 
    def raw(text): 
        return len([x for x in text if x in emotion_words])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)     


emotion_result=[]
for corpus in sub_corpora: 
    emotion_result.append(emotion(corpus))
    
df['emotion'] = pd.Series(emotion_result)

print("Standardised frequencies of emotion words written.")

#Feature 14 classical syntax markers
classical_syntax_markers=['备受', '言必称', '并存', '不得而', '抑且', '不特', '不外乎', \
'且', '不外乎', '不相', '中不乏', '不啻', '称之为', '称之', '充其量', '出于', '处于', \
'不次于', '从属于', '从中', '得自于', '得力于', '予以', '给予', '加以', '深具', '之能事', \
'发轫于', '凡此', '大抵', '凡', '所能及', '所可比', '非但', '庶可', '之故', '工于', '苟', \
'顾', '广为', '果', '核以', '何其', '或可', '跻身', '跻于', '不日即', '藉', '之大成', '再加', \
'略加', '详加', '以俱来', '见胜', '见长', '兼', '渐次', '化', '混同于', '归之于', '推广到', \
'名之为', '引为', '矣', '较', '借以', '尽其', '略陈己见', '而言', '而论', '决定于', '之先河', \
'苦不能', '莫不是', '乃', '泥于', '偏于', '颇有', '岂不', '岂可', '乎', '哉', '起源于', \
'何况', '切于', '取信于', '如', '则', '若', '岂', '舍', '甚于', '时年', '时值', '使之', \
'有别于', '倍加', '所在', '示人以', '随致', '之所以', '所以然', '无所', '有所', \
'皆指', '所引致', '罕为', '鲜为', '多为', '唯', '尚未', '无一不', '无不能', '无从', '可见', \
'毋宁', '无宁', '务', '系于', '仅限于', '方能', '需', '须', '许之为', '一改', '一变', '与否', \
'业已', '不以为然', '为能', '为多', '为最', '以期', '不宜', '宜于', '异于', '益见', '抑或', \
'故', '之便', '应推', '着手', '着眼', '可证', '可知', '可见', '而成', '有不', '有所', '有待于', \
'有赖于', '有助于', '有进于', '之分', '之别', '多有', '囿于', '与之', '同/共', '同为', '欲', \
'必', '喻之', '曰', '之际', '已然', '在于', '则', '者', '即是', '皆是', '云者', '者有之', \
'首属', '首推', '莫过于', '之', '之于', '置身于', '转而', '自', '自况', '自命', '自诩', \
'自认', '自居', '自许', '以降', '足以']

def classical_syntax(text):
    def raw(text): 
        return len([x for x in text if x in classical_syntax_markers])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized(text) * 1000, 2) 


classical_syntax_result=[]
for corpus in sub_corpora: 
    classical_syntax_result.append(classical_syntax(corpus))
    
df['classical_syntax'] = pd.Series(classical_syntax_result)

print("Standardised frequencies of classical syntax written.")


#Feature 15 disyllabic words 
disyllabic_words=['购买', '具有', '在于', '寻找', '获得', '询问', '进入', '等候', '安定', \
'安装', '办理', '保持', '保留', '保卫', '保障', '报道', '暴露', '爆发', '被迫', '必然', \
'必修', '必要', '避免', '编制', '变动', '变革', '辩论', '表达', '表示', '表演', '并肩', \
'补习', '不断', '不时', '不住', '布置', '采取', '采用', '参考', '测量', '测试', '测验', \
'颤动', '抄写', '陈列', '成立', '成为', '承担', '承认', '持枪', '充分', '充满', '充实', \
'仇恨', '出版', '处于', '处处', '传播', '传达', '创立', '次要', '匆忙', '从容', '从事', \
'促进', '摧毁', '达成', '达到', '打扫', '大力', '大有', '担任', '导致', '到达', '等待', \
'等候', '奠定', '雕刻', '调查', '动员', '独自', '端正', '锻炼', '夺取', '发表', '发动', \
'发挥', '发射', '发生', '发行', '发扬', '发展', '反抗', '防守', '防御', '防止', '防治', \
'非法', '废除', '粉碎', '丰富', '封锁', '符合', '负担', '负责', '复述', '复习', '复印', \
'复杂', '复制', '富有', '改编', '改革', '改进', '改良', '改善', '改正', '干涉', '敢于', \
'高大', '高度', '高速', '格外', '给以', '更加', '公开', '公然', '巩固', '贡献', '共同', \
'构成', '购买', '观测', '观察', '观看', '贯彻', '灌溉', '光临', '规划', '合成', '合法', \
'宏伟', '缓和', '缓缓', '回答', '汇报', '混淆', '活跃', '获得', '基本', '集合', '集中', \
'极为', '即将', '计划', '记载', '继承', '加工', '加紧', '加速', '加以', '驾驶', '歼灭', \
'坚定', '减轻', '检验', '简直', '建立', '建造', '建筑', '交换', '交流', '结束', '竭力', \
'解决', '解释', '紧急', '紧密', '谨慎', '进军', '进攻', '进入', '进行', '尽力', '禁止', \
'精彩', '进过', '经历', '经受', '经营', '竞争', '竟然', '纠正', '举办', '举行', '具备', \
'具体', '具有', '开办', '开动', '开发', '开明', '开辟', '开枪', '开设', '开展', '抗议', \
'克服', '刻苦', '空前', '扩大', '来自', '滥用', '朗读', '力求', '力争', '连接', '列举', \
'流传', '垄断', '笼罩', '轮流', '掠夺', '满腔', '盲目', '猛烈', '猛然', '梦想', '勉强', \
'面临', '明明', '明确', '难以', '扭转', '拍摄', '排列', '攀登', '炮打', '赔偿', '评价', \
'评论', '赔偿', '评价', '评论', '破坏', '普遍', '普及', '起源', '签订', '强调', '抢夺', \
'切实', '侵略', '侵入', '轻易', '取得', '全部', '全面', '燃烧', '热爱', '忍受', '仍旧', \
'日益', '如同', '散布', '丧失', '设法', '设立', '实施', '实现', '实行', '实验', '适合', \
'试验', '收集', '收缩', '树立', '束缚', '思考', '思念', '思索', '丝毫', '四处', '饲养', \
'损害', '损坏', '损失', '缩短', '缩小', '贪图', '谈论', '探索', '逃避', '提倡', '提供', \
'提前', '体现', '调节', '调整', '停止', '统一', '突破', '推迟', '推动', '推进', '脱离', \
'歪曲', '完善', '万分', '万万', '危害', '违背', '违反', '维持', '维护', '围绕', '伟大', \
'位于', '污染', '无比', '无法', '无穷', '无限', '武装', '吸取', '袭击', '喜爱', '显示', \
'限制', '陷入', '相互', '详细', '响应', '享受', '象征', '消除', '消耗', '小心', '写作', \
'辛勤', '修改', '修正', '修筑', '选择', '严格', '严禁', '严厉', '严密', '严肃', '研制', \
'延长', '掩盖', '养成', '一经', '依法', '依旧', '依然', '抑制', '应用', '永远', '踊跃', \
'游览', '予以', '遇到', '预防', '预习', '阅读', '运用', '再三', '遭到', '遭受', '遭遇', \
'增加', '增进', '增强', '占领', '占有', '战胜', '掌握', '照例', '镇压', '征服', '征求', \
'争夺', '争论', '整顿', '证明', '直到', '执行', '制定', '制订', '制造', '治疗', '中断', \
'重大', '专心', '转入', '转移', '装备', '装饰', '追求', '自学', '综合', '总结', '阻止', \
'钻研', '遵守', '左右']

def disyllabic(text):
    def raw(text): 
        return len([x for x in text if x in disyllabic_words])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)      


disyllabic_words_result=[]
for corpus in sub_corpora: 
    disyllabic_words_result.append(disyllabic(corpus))
    
df['disyllabic_words'] = pd.Series(disyllabic_words_result)

print("Standardised frequencies of disyllabic words written.")

#Feature 16 HSK core vocabulary level 1 150 words 
HSK1=['爱', '八', '爸爸', '杯子', '北京', '本', '不', '不客气', '菜', '茶', '吃', '出租车', '打电话', '大', '的', '点', '电脑', '电视', '电影', '东西', '都', '读', '对不起', '多', '多少', '儿子', '二', '饭店', '飞机', '分钟', '高兴', '个', '工作', '狗', '汉语', '好', '号', '喝', '和', '很', '后面', '回', '会', '几', '家', '叫', '今天', '九', '开', '看', '看见', '块', '来', '老师', '了', '冷', '里', '六', '妈妈', '吗', '买', '猫', '没关系', '没有', '米饭', '名字', '明天', '哪', '哪儿', '那', '呢', '能', '你', '年', '女儿', '朋友', '漂亮', '苹果', '七', '前面', '钱', '请', '去', '热', '人', '认识', '三', '商店', '上', '上午', '少', '谁', '什么', '十', '时候', '是', '书', '水', '水果', '睡觉', '说', '四', '岁', '他', '她', '太', '天气', '听', '同学', '喂', '我', '我们', '五', '喜欢', '下', '下午', '下雨', '先生', '现在', '想', '小', '小姐', '些', '写', '谢谢', '星期', '学生', '学习', '学校', '一', '一点儿', '衣服', '医生', '医院', '椅子', '有', '月', '再见', '在', '怎么', '怎么样', '这', '中国', '中午', '住', '桌子', '字', '昨天', '坐', '做']

def hsk1(text):
    def raw(text): 
        return len([x for x in text if x in HSK1])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


hsk1_result=[]
for corpus in sub_corpora: 
    hsk1_result.append(hsk1(corpus))
    
df['HSK_1'] = pd.Series(hsk1_result)

print("Standardised frequencies of HSK1 vocabulary written.")


#Feature 17 HSK core vocabulary level 3 (150-600), 450 words 
HSK3=['阿姨', '啊', '矮', '爱', '爱好', '安静', '把', '吧', '白', '百', '班', '搬', '办法', '办公室', '半', '帮忙', '帮助', '包', '饱', '报纸', '北方', '被', '鼻子', '比', '比较', '比赛', '笔记本', '必须', '变化', '别', '别人', '宾馆', '冰箱', '不但', '而且', '菜单', '参加', '草', '层', '差', '长', '唱歌', '超市', '衬衫', '成绩', '城市', '迟到', '出', '除了', '穿', '船', '春', '词典', '次', '聪明', '从', '错', '打篮球', '打扫', '打算', '大家', '带', '担心', '蛋糕', '当然', '到', '地', '得', '灯', '等', '地方', '地铁', '地图', '弟弟', '第一', '电梯', '电子邮件', '东', '冬', '懂', '动物', '短', '段', '锻炼', '对', '多么', '饿', '耳朵', '发', '发烧', '发现', '方便', '房间', '放', '放心', '非常', '分', '服务员', '附近', '复习', '干净', '感冒', '感兴趣', '刚才', '高', '告诉', '哥哥', '个子', '给', '根据', '跟', '更', '公共汽车', '公斤', '公司', '公园', '故事', '刮风', '关', '关系', '关心', '关于', '贵', '国家', '过', '过去', '还', '还是', '孩子', '害怕', '好吃', '黑', '黑板', '红', '后来', '护照', '花', '画', '坏', '欢迎', '环境', '换', '黄河', '回答', '会议', '火车站', '或者', '几乎', '机场', '机会', '鸡蛋', '极', '记得', '季节', '检查', '简单', '见面', '件', '健康', '讲', '教', '角', '脚', '教室', '接', '街道', '节目', '节日', '结婚', '结束', '姐姐', '解决', '介绍', '借', '进', '近', '经常', '经过', '经理', '久', '旧', '就', '句子', '决定', '觉得', '咖啡', '开始', '考试', '可爱', '可能', '可以', '渴', '刻', '客人', '课', '空调', '口', '哭', '裤子', '快', '快乐', '筷子', '蓝', '老', '累', '离', '离开', '礼物', '历史', '脸', '练习', '两', '辆', '聊天', '了解', '邻居', '零', '留学', '楼', '路', '旅游', '绿', '马', '马上', '卖', '满意', '慢', '忙', '帽子', '每', '妹妹', '门', '米', '面包', '面条', '明白', '拿', '奶奶', '男', '南', '难', '难过', '年级', '年轻', '鸟', '您', '牛奶', '努力', '女', '爬山', '盘子', '旁边', '胖', '跑步', '皮鞋', '啤酒', '便宜', '票', '瓶子', '妻子', '其实', '其他', '奇怪', '骑', '起床', '起飞', '起来', '千', '铅笔', '清楚', '晴', '请假', '秋', '去年', '裙子', '然后', '让', '热情', '认为', '认真', '日', '容易', '如果', '伞', '上班', '上网', '谁', '身体', '生病', '生气', '生日', '声音', '时间', '世界', '事情', '试', '手表', '手机', '瘦', '叔叔', '舒服', '树', '数学', '刷牙', '双', '水平', '说话', '司机', '送', '虽然', '但是', '它', '她', '太阳', '特别', '疼', '踢足球', '提高', '题', '体育', '甜', '条', '跳舞', '同事', '同意', '头发', '突然', '图书馆', '腿', '外', '完', '完成', '玩', '晚上', '碗', '万', '往', '忘记', '为', '为了', '为什么', '位', '文化', '问', '问题', '西', '西瓜', '希望', '习惯', '洗', '洗手间', '洗澡', '夏', '先', '相信', '香蕉', '向', '像', '小时', '小心', '校长', '笑', '新', '新闻', '新鲜', '信用卡', '行李箱', '姓', '熊猫', '休息', '需要', '选择', '雪', '颜色', '眼睛', '羊肉', '要求', '药', '要', '爷爷', '也', '一般', '一边', '一定', '一共', '一会儿', '一起', '一下', '一样', '一直', '已经', '以前', '意思', '因为', '所以', '阴', '音乐', '银行', '饮料', '应该', '影响', '用', '游戏', '游泳', '有名', '又', '右边', '鱼', '遇到', '元', '远', '愿意', '月亮', '越', '运动', '再', '早上', '站', '张', '丈夫', '着急', '找', '照顾', '照片', '照相机', '着', '真', '正在', '只', '只有', '才', '中间', '中文', '终于', '种', '重要', '周末', '主要', '注意', '准备', '自己', '自行车', '总是', '走', '嘴', '最', '最后', '最近', '左边', '作业']

def hsk3(text):
    def raw(text): 
        return len([x for x in text if x in HSK3])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


hsk3_result=[]
for corpus in sub_corpora: 
    hsk3_result.append(hsk3(corpus))
    
df['HSK_3'] = pd.Series(hsk3_result)

print("Standardised frequencies of HSK3 vocabulary written.")


#Feature 18 honourifics
honourifics=['千金', '相公', '姑姥爷', '伯伯', '伯父', '伯母', '大伯', '大哥', '大姐', '大妈', '大爷', '大嫂', '嫂夫人', '大婶儿', '大叔', '大姨', '哥', '姐', '大娘', '妈妈', '奶 奶', '爷爷', '姨', '老伯', '老兄', '老爹', '老大爷', '老爷爷', '老太太', '老奶奶', '老大娘', '老板', '老公', '老婆婆', '老前辈', '老人家', '老师', '老师傅', '老寿星', '老太爷', '老翁', '老爷子', '老丈', '老总', '大驾', '夫人', '高徒', '高足', '官人', '贵客', '贵人', '嘉宾', '列位', '男士', '女士', '女主 人', '前辈', '台驾', '太太', '先生', '贤契', '贤人', '贤士', '先哲', '小姐', '学长', '爷', '诸位', '足下', '师傅', '师母', '师娘', '人士', '长老', '禅师', '船老大', '大师', '大师傅', '大王', '恩师', '法师', '法王', '佛爷', '夫子', '父母官', '国父', '麾下', '教授', '武师', '千 岁', '孺人', '圣母', '圣人', '师父', '王尊', '至尊', '座', '少奶奶', '少爷', '金枝玉叶', '工程师', '高级工程师', '经济师', '讲师', '教授', '副教授', '教师', '老师', '国家主席', '国家总理', '部长', '厅长', '市长', '局长', '科长', '校长', '烈士', '先烈', '先哲', '荣誉军人', '陛下', '殿下', '阁下', '阿公', '阿婆', '大人', '公', '公公', '娘子', '婆婆', '丈人', '师长', '义士', '勇士', '志士', '壮士', '学生', '兄弟', '小弟', '弟', '妹', '儿子', '女儿']

def honor(text):
    def raw(text): 
        return len([x for x in text if x in honourifics])
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


honor_result=[]
for corpus in sub_corpora: 
    honor_result.append(honor(corpus))
    
df['honourifics'] = pd.Series(honor_result)

print("Standardised frequencies of honourifics written.")


#convert text to lists for unique words
text_lists=[]
for corpus in sub_corpora: 
    text_list=list(corpus)
    text_lists.append(text_list)

#Feature 19 unique items 
def unique(text): 
    return round((len([x for x in text if text.count(x)==1]) / len(text))*1000, 2)

unique_result=[]
for corpus in sub_corpora: 
    unique_result.append(unique(corpus))

df['unique'] = pd.Series(unique_result)

print("Standardised unique words ratio written.")


#-------------------------------------part of speech Features------------------------

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


#print (none_list)

#the number in (range) is your number of files 
#the following are words I found that ICTCLAS does not know
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
#print (none_list)

#close pynlpir to free allocated memory
pynlpir.close()


#Feature 20 COND 条件连词、副词
#note the pos of some conjuncts needs to be specified
def cond(text):
    def raw(text): 
        return str(text).count('如果') + str(text).count('只有') + str(text).count('假如')+str(text).count('除非')+str(text).count('要是')+str(text).count('要不是')+str(text).count('只要')+str(text).count('假如')+str(text).count('倘若')+str(text).count('倘或')+str(text).count('设使')+str(text).count('设若')+str(text).count('如若')+str(text).count('若')+text.count(('的话', 'particle 的话'))+str(text).count("('的', 'particle 的/底'), ('时候', 'noun')")
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


cond_result=[]
for file in tagged_files: 
    cond_result.append(cond(file))
    
df['COND'] = pd.Series(cond_result)

print("Standardised frequencies of conditional conjuncts written.")


#Feature 21 modifying adverbs
def modify_adv(text): 
    def raw(text): 
        return text.count(('也', 'adverb'))+text.count(('都', 'adverb'))+text.count(('又', 'adverb'))+text.count(('才', 'adverb'))+text.count(('就', 'adverb'))+text.count(('就是', 'adverb'))+text.count(('倒是', 'adverb'))+text.count(('越来越', 'adverb'))+text.count(('一边', 'adverb'))+text.count(('再', 'adverb'))+text.count(('甚至', 'adverb'))+text.count(('连', 'particle 连'))+text.count(('却', 'adverb'))+text.count(('原本', 'adverb'))+text.count(('只', 'adverb'))+text.count(('毕竟', 'adverb'))+text.count(('仍然', 'adverb'))    +text.count(('反正', 'adverb'))+text.count(('等', 'particle 等/等等/云云'))+text.count(('刚', 'adverb'))+text.count(('常常', 'adverb'))+text.count(('已经', 'adverb'))+text.count(('就要', 'adverb'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


modify_adv_result=[]
for file in tagged_files: 
    modify_adv_result.append(modify_adv(file))
    
df['modify_adv'] = pd.Series(modify_adv_result)

print("Standardised frequencies of modifying adverbs written.")


#Feature 22 demonstrative pronoun
def demp(text): 
    def raw(text): 
        return str(text).count('demonstrative pronoun')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


demp_result=[]
for file in tagged_files: 
    demp_result.append(demp(file))
    
df['DEMP'] = pd.Series(demp_result)

print("Standardised frequencies of demonstrative pronouns (DEMP) written.")


#Feature 23 Be 是
def be(text):
    def raw(text): 
        return text.count(('是', 'verb 是'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


be_result=[]
for file in tagged_files: 
    be_result.append(be(file))
    
df['BE'] = pd.Series(be_result)

print("Standardised frequencies of be 是 (BE) written.")


#Feature 24 EX有
def ex(text): 
    def raw(text): 
        return str(text).count('verb 有')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


ex_result=[]
for file in tagged_files: 
    ex_result.append(ex(file))
    
df['EX'] = pd.Series(ex_result)

print("Standardised frequencies of existential 有 (EX) written.")


#Feature 25 other personal pronouns apart from FPP, SPP, TPP
def other_personal(text): 
    def raw(text): 
        return str(text).count('personal pronoun')-sum(map(str(text).count, ['我', '你', '您','她', '他', '它']))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


other_personal_result=[]
for file in tagged_files: 
    other_personal_result.append(other_personal(file))
    
df['other_personal'] = pd.Series(other_personal_result)

print("Standardised frequencies of other personal pronouns written.")


#Feature 26 interrogative pronouns 
#excluding predicative interrogative noun (tagged as WH words)
def interrogative(text): 
    def raw(text): 
        return str(text).count('interrogative pronoun')-str(text).count('predicative interrogative pronoun')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


interrogative_result=[]
for file in tagged_files: 
    interrogative_result.append(interrogative(file))
    
df['interrogative'] = pd.Series(interrogative_result)

print("Standardised frequencies of interrogative pronouns written.")


#Feature 27 noun nouns, all other nouns
#formula: noun - noun-adjective - noun-verb - pronoun-
#personal pronoun - predicate demonstrative pronoun 
#-demonstrative pronoun - locative demonstrative pronoun 
#- predicate interrogative pronoun
#- interrogative pronoun (WH words) - predicate demonstrative pronoun
def noun(text):
    def raw(text): 
        return str(text).count('noun')-sum(map(str(text).count, ['noun-adjective', 'noun-verb', 'pronoun','noun of locality', 'noun morpheme', 'proper noun']))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)        


noun_result=[]
for file in tagged_files: 
    noun_result.append(noun(file))
    
df['noun'] = pd.Series(noun_result)


print("Standardised frequencies of nouns written.")


#Feature 28 phrasal coordinations PHC 
#和、以及、与、并、及、暨
###to add:  same tags before and after ###
#ICTCLAS coordinating conjunction
def phc(text):
    def raw(text):
        return str(text).count('coordinating conjunction')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


phc_result=[]
for file in tagged_files: 
    phc_result.append(phc(file))
    
df['PHC'] = pd.Series(phc_result)


print("Standardised frequencies of phrasal coordinators (PHC) written.")

#Feature 29 BPIN disyllabic prepositions
def bpin(text):
    def raw(text):
        return text.count(('按照', 'preposition'))+text.count(('本着', 'preposition'))+text.count(('按着', 'preposition'))+text.count(('朝着', 'preposition'))+text.count(('趁着', 'preposition'))+text.count(('出于', 'preposition'))+text.count(('待到', 'preposition'))+text.count(('对于', 'preposition'))+text.count(('根据', 'preposition'))+text.count(('关于', 'preposition'))+text.count(('基于', 'preposition'))+text.count(('鉴于', 'preposition'))+text.count(('借着', 'preposition'))+text.count(('经过', 'preposition'))+text.count(('靠着', 'preposition'))+text.count(('冒着', 'preposition'))+text.count(('面对', 'preposition'))+text.count(('面临', 'preposition'))+text.count(('凭借', 'preposition'))+text.count(('顺着', 'preposition'))+text.count(('随着', 'preposition'))+text.count(('通过', 'preposition'))+text.count(('为了', 'preposition'))+text.count(('围绕', 'preposition'))+text.count(('向着', 'preposition'))+text.count(('沿着', 'preposition'))+text.count(('依据', 'preposition'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)    


bpin_result=[]
for file in tagged_files: 
    bpin_result.append(bpin(file))
    
df['BPIN'] = pd.Series(bpin_result)


print("Standardised frequencies of disyllabic prepositions (BPIN) written.")

#Feature 30 private verbs 
#subset 1
def priv1(text):
    def raw(text): 
        return text.count(('三思', 'verb'))+text.count(('三省', 'verb'))+ text.count(('主张', 'verb'))+text.count(('了解', 'verb'))+ text.count(('亲信', 'verb'))+text.count(('以为', 'verb'))+ text.count(('企图', 'verb'))+text.count(('会意', 'verb'))+  text.count(('伤心', 'verb'))+text.count(('估', 'verb'))+  text.count(('估摸', 'verb'))+text.count(('估算', 'verb'))+  text.count(('估计', 'verb'))+text.count(('估量', 'verb'))+  text.count(('低估', 'verb'))+text.count(('体会', 'verb'))+  text.count(('体味', 'verb'))+text.count(('信', 'verb'))+  text.count(('信任', 'verb'))+text.count(('信赖', 'verb'))+  text.count(('修省', 'verb'))+text.count(('假定', 'verb'))+  text.count(('假想', 'verb'))+text.count(('允许', 'verb'))+  text.count(('关心', 'verb'))+text.count(('关怀', 'verb'))+text.count(('内省', 'verb'))+text.count(('决定', 'verb'))+  text.count(('决心', 'verb'))+text.count(('决意', 'verb'))+text.count(('决断', 'verb'))+text.count(('决计', 'verb'))+text.count(('准备', 'verb'))+text.count(('准许', 'verb'))+  text.count(('凝思', 'verb'))+text.count(('凝想', 'verb'))+text.count(('凭信', 'verb'))+text.count(('分晓', 'verb'))+  text.count(('切记', 'verb'))+text.count(('划算', 'verb'))+  text.count(('判断', 'verb'))+text.count(('原谅', 'verb'))+  text.count(('参悟', 'verb'))+text.count(('反对', 'verb'))+  text.count(('反思', 'verb'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)        


p_1 = []
for file in tagged_files: 
    p_1.append(priv1(file))


#2
def priv2(text):
    def raw(text): 
        return text.count(('反省', 'verb'))+text.count(('发现', 'verb'))+text.count(('发觉', 'verb'))+text.count(('吃准', 'verb'))+text.count(('合计', 'verb'))+text.count(('合谋', 'verb'))+text.count(('同情', 'verb'))+text.count(('同意', 'verb'))+text.count(('否认', 'verb'))+text.count(('听信', 'verb'))+text.count(('听到', 'verb'))+text.count(('听见', 'verb'))+text.count(('哭', 'verb'))+text.count(('喜欢', 'verb'))+text.count(('喜爱', 'verb'))+text.count(('回味', 'verb'))+text.count(('回忆', 'verb'))+text.count(('回念', 'verb'))+text.count(('回想', 'verb'))+text.count(('回溯', 'verb'))+text.count(('回顾', 'verb'))+text.count(('图谋', 'verb'))+text.count(('图', 'verb'))+text.count(('坚信', 'verb'))+text.count(('多疑', 'verb'))+text.count(('失望', 'verb'))+text.count(('失身', 'verb'))+text.count(('妄图', 'verb'))+text.count(('妄断', 'verb'))+text.count(('宠信', 'verb'))+text.count(('害怕', 'verb'))+text.count(('察觉', 'verb'))+text.count(('寻思', 'verb'))+text.count(('尊敬', 'verb'))+text.count(('尊重', 'verb'))+text.count(('小心', 'verb'))+text.count(('希望', 'verb'))+text.count(('平静', 'verb'))+text.count(('幻想', 'verb'))+text.count(('当做', 'verb'))+text.count(('彻悟', 'verb'))+text.count(('得知', 'verb'))+text.count(('忆', 'verb'))+text.count(('忖度', 'verb'))+text.count(('忖量', 'verb'))+text.count(('忘', 'verb'))+text.count(('忘却', 'verb'))+text.count(('忘怀', 'verb'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


p_2 = []
for file in tagged_files: 
    p_2.append(priv2(file))


#3
def priv3(text):
    def raw(text): 
        return text.count(('忘掉', 'verb'))+text.count(('忘记', 'verb'))+text.count(('快乐', 'verb'))+text.count(('念', 'verb'))+text.count(('忽略', 'verb'))+text.count(('忽视', 'verb'))+text.count(('怀念', 'verb'))+text.count(('怀想', 'verb'))+text.count(('怀疑', 'verb'))+text.count(('怕', 'verb'))+text.count(('思忖', 'verb'))+text.count(('思想', 'verb'))+text.count(('思索', 'verb'))+text.count(('思维', 'verb'))+text.count(('思考', 'verb'))+text.count(('思虑', 'verb'))+text.count(('思量', 'verb'))+text.count(('恨', 'verb'))+text.count(('悟', 'verb'))+text.count(('悬想', 'verb'))+text.count(('情知', 'verb'))+text.count(('惊恐', 'verb'))+text.count(('想', 'verb'))+text.count(('想像', 'verb'))+text.count(('想来', 'verb'))+text.count(('想见', 'verb'))+text.count(('想象', 'verb'))+text.count(('愉快', 'verb'))+text.count(('意会', 'verb'))+text.count(('意想', 'verb'))+text.count(('意料', 'verb'))+text.count(('意识', 'verb'))+text.count(('感到', 'verb'))+text.count(('感动', 'verb'))+text.count(('感受', 'verb'))+text.count(('感悟', 'verb'))+text.count(('感想', 'verb'))+text.count(('感激', 'verb'))+text.count(('感觉', 'verb'))+text.count(('感觉', 'verb'))+text.count(('感谢', 'verb'))+text.count(('愤怒', 'verb'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 



p_3 = []
for file in tagged_files: 
    p_3.append(priv3(file))



#4
def priv4(text):
    def raw(text): 
        return text.count(('愿意', 'verb'))+text.count(('懂', 'verb'))+text.count(('懂得', 'verb'))+text.count(('打算', 'verb'))+text.count(('承想', 'verb'))+text.count(('承认', 'verb'))+text.count(('担心', 'verb'))+text.count(('拥护', 'verb'))+text.count(('捉摸', 'verb'))+text.count(('掂掇', 'verb'))+text.count(('掂量', 'verb'))+text.count(('掌握', 'verb'))+text.count(('推度', 'verb'))+text.count(('推想', 'verb'))+text.count(('推敲', 'verb'))+text.count(('推断', 'verb'))+text.count(('推测', 'verb'))+text.count(('推理', 'verb'))+text.count(('推算', 'verb'))+text.count(('推见', 'verb'))+text.count(('措意', 'verb'))+text.count(('揆度', 'verb'))+text.count(('揣度', 'verb'))+text.count(('揣想', 'verb'))+text.count(('揣摩', 'verb'))+text.count(('揣摸', 'verb'))+text.count(('揣测', 'verb'))+text.count(('支持', 'verb'))+text.count(('放心', 'verb'))+text.count(('料想', 'verb'))+text.count(('料', 'verb'))+text.count(('斟酌', 'verb'))+text.count(('断定', 'verb'))+text.count(('明了', 'verb'))+text.count(('明察', 'verb'))+text.count(('明晓', 'verb'))+text.count(('明白', 'verb'))+text.count(('明知', 'verb'))+text.count(('明确', 'verb'))+text.count(('晓得', 'verb'))+text.count(('权衡', 'verb'))+text.count(('梦想', 'verb'))+text.count(('欢迎', 'verb'))+text.count(('欣赏', 'verb'))+text.count(('武断', 'verb'))+text.count(('死记', 'verb'))+text.count(('沉思', 'verb'))+text.count(('注意', 'verb'))+text.count(('洞察', 'verb'))+text.count(('洞彻', 'verb'))+text.count(('洞悉', 'verb'))+text.count(('洞晓', 'verb'))+text.count(('洞达', 'verb'))+text.count(('测度', 'verb'))+text.count(('浮想', 'verb'))+text.count(('淡忘', 'verb'))+text.count(('深信', 'verb'))+text.count(('深思', 'verb'))+text.count(('深省', 'verb'))+text.count(('深醒', 'verb'))+text.count(('清楚', 'verb'))+text.count(('清楚', 'verb'))+text.count(('满意', 'verb'))+text.count(('满足', 'verb'))+text.count(('激动', 'verb'))+text.count(('热爱', 'verb'))+text.count(('熟悉', 'verb'))+text.count(('熟知', 'verb'))+text.count(('熟虑', 'verb'))+text.count(('爱', 'verb'))+text.count(('爱好', 'verb'))+text.count(('牢记', 'verb'))+text.count(('犯疑', 'verb'))+text.count(('狂想', 'verb'))+text.count(('狐疑', 'verb'))+text.count(('猛醒', 'verb'))+text.count(('猜', 'verb'))+text.count(('猜度', 'verb'))+text.count(('猜忌', 'verb'))+text.count(('猜想', 'verb'))+text.count(('猜测', 'verb'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


p_4 = []
for file in tagged_files: 
    p_4.append(priv4(file))


#5 
def priv5(text):
    def raw(text): 
        return text.count(('猜疑', 'verb'))+text.count(('玄想', 'verb'))+text.count(('理会', 'verb'))+text.count(('理解', 'verb'))+text.count(('琢磨', 'verb'))+text.count(('生气', 'verb'))+text.count(('生疑', 'verb'))+text.count(('畅想', 'verb'))+text.count(('留心', 'verb'))+text.count(('留神', 'verb'))+text.count(('疏忽', 'verb'))+text.count(('疑', 'verb'))+text.count(('疑心', 'verb'))+text.count(('疑猜', 'verb'))+text.count(('疑虑', 'verb'))+text.count(('疼', 'verb'))+text.count(('盘算', 'verb'))+text.count(('相信', 'verb'))+text.count(('盼望', 'verb'))+text.count(('省察', 'verb'))+text.count(('省悟', 'verb'))+text.count(('看', 'verb'))+text.count(('看到', 'verb'))+text.count(('看见', 'verb'))+text.count(('看透', 'verb'))+text.count(('着想', 'verb'))+text.count(('知', 'verb'))+text.count(('知悉', 'verb'))+text.count(('知晓', 'verb'))+text.count(('知道', 'verb'))+text.count(('确信', 'verb'))+text.count(('确定', 'verb'))+text.count(('确认', 'verb'))+text.count(('空想', 'verb'))+text.count(('立意', 'verb'))+text.count(('笃信', 'verb'))+text.count(('笑', 'verb'))+text.count(('答应', 'verb'))+text.count(('策划', 'verb'))+text.count(('筹划', 'verb'))+text.count(('筹算', 'verb'))+text.count(('筹谋', 'verb'))+text.count(('算', 'verb'))+text.count(('算计', 'verb'))+text.count(('粗估', 'verb'))+text.count(('约摸', 'verb'))+text.count(('置疑', 'verb'))+text.count(('考虑', 'verb'))+text.count(('考量', 'verb'))+text.count(('联想', 'verb'))+text.count(('腹诽', 'verb'))+text.count(('臆度', 'verb'))+text.count(('臆想', 'verb'))+text.count(('臆断', 'verb'))+text.count(('臆测', 'verb'))+text.count(('自信', 'verb'))+text.count(('自省', 'verb'))+text.count(('蒙', 'verb'))+text.count(('蓄念', 'verb'))+text.count(('蓄谋', 'verb'))+text.count(('衡量', 'verb'))+text.count(('裁度', 'verb'))+text.count(('要求', 'verb'))+text.count(('观察', 'verb'))+text.count(('觉察', 'verb'))+text.count(('觉得', 'verb'))+text.count(('觉悟', 'verb'))+text.count(('觉醒', 'verb'))+text.count(('警惕', 'verb'))+text.count(('警觉', 'verb'))+text.count(('计划', 'verb'))+text.count(('计算', 'verb'))+text.count(('计较', 'verb'))+text.count(('认为', 'verb'))+text.count(('认可', 'verb'))+text.count(('认同', 'verb'))+text.count(('认定', 'verb'))+text.count(('认得', 'verb'))+text.count(('认知', 'verb'))+text.count(('认识', 'verb'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


p_5 = []
for file in tagged_files: 
    p_5.append(priv5(file))


#6
def priv6(text):
    def raw(text): 
        return text.count(('讨厌', 'verb'))+text.count(('记', 'verb'))+text.count(('记取', 'verb'))+text.count(('记得', 'verb'))+text.count(('记忆', 'verb'))+text.count(('设想', 'verb'))+text.count(('识', 'verb'))+text.count(('试图', 'verb'))+text.count(('试想', 'verb'))+text.count(('详悉', 'verb'))+text.count(('误会', 'verb'))+text.count(('误解', 'verb'))+text.count(('谋划', 'verb'))+text.count(('谋算', 'verb'))+text.count(('谋虑', 'verb'))+text.count(('赞同', 'verb'))+text.count(('赞成', 'verb'))+text.count(('走神儿', 'verb'))+text.count(('起疑', 'verb'))+text.count(('轻信', 'verb'))+text.count(('轻视', 'verb'))+text.count(('迷信', 'verb'))+text.count(('迷信', 'verb'))+text.count(('追忆', 'verb'))+text.count(('追怀', 'verb'))+text.count(('追思', 'verb'))+text.count(('追想', 'verb'))+text.count(('通彻', 'verb'))+text.count(('通晓', 'verb'))+text.count(('通', 'verb'))+text.count(('遐想', 'verb'))+text.count(('遗忘', 'verb'))+text.count(('遥想', 'verb'))+text.count(('酌情', 'verb'))+text.count(('酌量', 'verb'))+text.count(('醒', 'verb'))+text.count(('醒悟', 'verb'))+text.count(('重视', 'verb'))+text.count(('铭记', 'verb'))+text.count(('阴谋', 'verb')) +text.count(('顾全', 'verb'))+text.count(('顾及', 'verb'))+text.count(('预卜', 'verb'))+text.count(('预想', 'verb'))+text.count(('预感', 'verb'))+text.count(('预料', 'verb'))+text.count(('预期', 'verb'))+text.count(('预测', 'verb'))+text.count(('预知', 'verb'))+text.count(('预见', 'verb'))+text.count(('预计', 'verb'))+text.count(('预谋', 'verb'))+text.count(('领会', 'verb'))+text.count(('领悟', 'verb'))+text.count(('领略', 'verb'))+text.count(('高估', 'verb'))+text.count(('高兴', 'verb'))+text.count(('默认', 'verb'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


p_6 = []
for file in tagged_files: 
    p_6.append(priv6(file))


random1=[sum(i) for i in zip(p_1, p_2)]
random2=[sum(i) for i in zip(p_3, p_4)]
random3=[sum(i) for i in zip(p_5, p_6)]
random4=[sum(i) for i in zip(random1, random2)]
final=[sum(i) for i in zip(random3, random4)]
    
df['PRIV'] = pd.Series(final)



print("Standardised frequencies of private verbs (PRIV) written.")

#Feature 31 public verb PUBV
def pubv(text):
    def raw(text): 
        return text.count(('表示', 'verb'))+text.count(('称', 'verb'))+text.count(('道', 'verb'))+text.count(('说', 'verb'))+text.count(('讲', 'verb'))+text.count(('质疑', 'verb'))+text.count(('认为', 'verb'))+text.count(('坦言', 'verb'))+text.count(('指出', 'verb'))+text.count(('告诉', 'verb'))+text.count(('呼吁', 'verb'))+text.count(('解释', 'verb'))+text.count(('问', 'verb'))+text.count(('建议', 'verb'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)


pubv_result=[]
for file in tagged_files: 
    pubv_result.append(pubv(file))
    
df['PUBV'] = pd.Series(pubv_result)


print("Standardised frequencies of public verbs (PUBV) written.")


#Feature 32 RB adverbs 副词
def rb(text):
    def raw(text): 
        return str(text).count('adverb')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)


rb_result=[]
for file in tagged_files: 
    rb_result.append(rb(file))
    
df['RB'] = pd.Series(rb_result)


print("Standardised frequencies of adverbs (RB) written.")


#Feature 33 mononegation 不、别、没
def mono_negation(text):
    def raw(text): 
        return text.count(('别', 'adverb'))+text.count(('不', 'adverb'))+text.count(('没', 'verb'))+text.count(('没', 'adverb'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)


mono_negation_result=[]
for file in tagged_files: 
    mono_negation_result.append(mono_negation(file))
    
df['mono_negation'] = pd.Series(mono_negation_result)


print("Standardised frequencies of monosyllabic negation written.")

#Feature 34 disyllabic negation 没有
def di_negation(text):
    def raw(text): 
        return text.count(('没有', 'adverb'))+text.count(('没有', 'verb'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)


di_negation_result=[]
for file in tagged_files: 
    di_negation_result.append(di_negation(file))
    
df['di_negation'] = pd.Series(di_negation_result)


print("Standardised frequencies of disyllabic negation written.")


#Feature 35 WH 无定代词
def wh(text):
    def raw(text):
        return str(text).count('predicate interrogative pronoun')
    def normalized(text):                                                                                     
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


wh_result=[]
for file in tagged_files: 
    wh_result.append(wh(file))
    
df['WH'] = pd.Series(wh_result)


print("Standardised frequencies of wh-words (WH) written.")


#Feature 36 monosyllabic verbs
#1. convert corpora to dict # no None in values 
#2. select verbs from dict values 
#3. convert corresponding keys to list 
#4. return len(corresponding keys) == 2 or 1 / len (filtered dict)

##dict would remove duplicates, not ideal
dicts=[]
for file in tagged_files: 
    d=dict(file)
    dicts.append(d)

def mono_verbs(text): 
    verbs= {k:v for k,v in text.items() if re.match('.*verb', v)}
    return round ((len([word for word in list(verbs.keys()) if len(word) == 1]) / len(text))*1000, 2)


mono_verbs_result=[]
for d in dicts: 
    mono_verbs_result.append(mono_verbs(d))
    
df['mono_verbs'] = pd.Series(mono_verbs_result)

print("Standardised frequencies of monosyllabic verbs written.")


#Feature 37 classical grammatical words 文言文功能词
def classical_gram(text):
    def raw(text): 
        return text.count(('所', 'particle 所'))+text.count(('将', 'adverb'))+text.count(('将', 'preposition'))+text.count(('之', 'particle 之'))+text.count(('于', 'preposition'))+text.count(('以', 'preposition'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)


classical_gram_result=[]
for file in tagged_files: 
    classical_gram_result.append(classical_gram(file))
    
df['classical_gram'] = pd.Series(classical_gram_result)

print("Standardised frequencies of classical grammatical words written.")


#Feature 38 lexical density 
#(noun+verb+adjective+numeral) / total
def lexical_density(text): 
    verbs= {k:v for k,v in text.items() if re.match('.*verb', v)}
    nouns= {k:v for k,v in text.items() if re.match('.*noun', v)}
    adjectives= {k:v for k,v in text.items() if re.match('.*adjective', v)}
    numerals= {k:v for k,v in text.items() if re.match('.*numeral', v)}
    adverbs={k:v for k,v in text.items() if re.match('.*adverb', v)}
    pronouns={k:v for k,v in text.items() if re.match('.*pronoun', v)}
    return round(((len(verbs)+len(nouns)+len(adjectives)+len(numerals)-len(adverbs)-len(pronouns)) / len(text))*1000, 2)


#note that tagged lists are converted to dictionaries here as well 
lexical_density_result=[]
for d in dicts: 
    lexical_density_result.append(lexical_density(d))
    
df['lexical_density'] = pd.Series(lexical_density_result)

print("Lexical density written.")


#Feature 39 auxiliary adjectives
def aux_adj(text):
    def raw(text):
        return str(text).count('auxiliary adjective')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)


aux_adj_result=[]
for file in tagged_files: 
    aux_adj_result.append(aux_adj(file))
    
df['aux_adj'] = pd.Series(aux_adj_result)

print("Standardised frequencies of auxiliary adjectives written.")


#Feature 40 classifier 量词
def classifier(text):
    def raw(text): 
        return str(text).count('classifier')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)


classifier_result=[]
for file in tagged_files: 
    classifier_result.append(classifier(file))
    
df['classifier'] = pd.Series(classifier_result)

print("Standardised frequencies of classifiers written.")


#Feature 41 modal particles and interjections 
def particle(text):
    def raw(text): 
        return str(text).count('modal particle')+str(text).count('interjection')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)


particle_result=[]
for file in tagged_files: 
    particle_result.append(particle(file))
    
df['particle'] = pd.Series(particle_result)

print("Standardised frequencies of modal particles and interjections written.")


#Feature 42 adverbial marker 地
def adverbial_marker_di(text):
    def raw(text):
        return text.count(('地', 'particle 地'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)    


adverbial_marker_di_result=[]
for file in tagged_files: 
    adverbial_marker_di_result.append(adverbial_marker_di(file))
    
df['adverbial_marker_di'] = pd.Series(adverbial_marker_di_result)

print("Standardised frequencies of adverbial marker di written.")


#Feature 43 complement marker '得', 'particle 得'
def complement_marker_de(text):
    def raw(text):
        return text.count(('得', 'particle 得'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)    

complement_marker_de_result=[]
for file in tagged_files: 
    complement_marker_de_result.append(complement_marker_de(file))
    
df['complement_marker_de'] = pd.Series(complement_marker_de_result)

print("Standardised frequencies of complement marker de written.")

#Feature 44 perfect aspect (PEAS)
def peas(text):
    def raw(text):
        return text.count(('了', 'particle 了/喽'))+text.count(('过', 'particle 过'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)    

peas_result=[]
for file in tagged_files: 
    peas_result.append(peas(file))
    
df['PEAS'] = pd.Series(peas_result)

print("Standardised frequencies of perfect aspect markers (PEAS) written.")


#Feature 45 imperfect aspect 
def imperfect(text):
    def raw(text):
        return text.count(('着', 'particle 着'))+text.count(('在', 'preposition'))+text.count(('正在', 'adverb'))+text.count(('起来', 'directional verb'))+text.count(('下去', 'directional verb'))
    def normalized(text):                                                                                
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 

imperfect_result=[]
for file in tagged_files: 
    imperfect_result.append(imperfect(file))
    
df['imperfect'] = pd.Series(imperfect_result)

print("Standardised frequencies of imperfect aspect markers written.")

#Feature 46 descriptive words 
#ICTCLAS status word
def descriptive(text):
    def raw(text):
        return str(text).count('status word')
    def normalized(text):                                                                                     
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2) 


descriptive_result=[]
for file in tagged_files: 
    descriptive_result.append(descriptive(file))
    
df['descriptive'] = pd.Series(descriptive_result)

print("Standardised frequencies of descriptive words written.")


#Feature 47 similie
def simile(text):
    def raw(text):
        return text.count(('仿佛', 'adverb'))+text.count(('宛若', 'verb'))+text.count(('如', 'verb'))+str(text).count(('particle 一样/一般/似的/般'))+text.count(('像', 'verb'))+text.count(('像', 'preposition'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)

simile_result=[]
for file in tagged_files: 
    simile_result.append(simile(file))
    
df['simile'] = pd.Series(simile_result)

print("Standardised frequencies of simile written.")


#Feature 48 questions
def question(text):
    def raw(text):
        return text.count(('？', 'question mark'))
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)

question_result=[]
for file in tagged_files: 
    question_result.append(question(file))

df['question'] = pd.Series(question_result)

print("Standardised frequencies of questions written.")

#Feature 49 exclamation mark
def exclamation(text):
    def raw(text):
        return str(text).count('exclamation mark')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)

exclamation_result=[]
for file in tagged_files: 
    exclamation_result.append(exclamation(file))
    
df['exclamation'] = pd.Series(exclamation_result)

print("Standardised frequencies of exclamative sentences written.")

#Feature 50 Chinese person names
#personal name + Chinese - transcribed personal names
def person(text):
    def raw(text):
        return str(text).count('personal name')+str(text).count('Chinese')- str(text).count('transcribed personal name')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)

person_result=[]
for file in tagged_files: 
    person_result.append(person(file))
    
df['Chinese_person'] = pd.Series(person_result)

print("Standardised frequencies of Chinese person names written.")

#feature 51 intransitive verbs
def vi(text):
    def raw(text):
        return str(text).count('intransitive verb')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)

vi_result=[]
for file in tagged_files: 
    vi_result.append(vi(file))

df['intransitive'] = pd.Series(vi_result)

print("Standardised frequencies of intransitive verbs written.")


#feature 52
#onomatopoeia 拟声词
def ono(text):
    def raw(text):
        return str(text).count('onomatopoeia')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)


ono_result=[]
for file in tagged_files: 
    ono_result.append(ono(file))
    
df['onomatopoeia'] = pd.Series(ono_result)


#feature 53 disyllabic verbs
def di_verbs(text): 
    no_adverbs={k:v for k,v in dict(text).items() if re.match('^((?!adverb).)*$', v)}
    verbs= {k:v for k,v in no_adverbs.items() if re.match('.*verb', v)}
    return round((len([word for word in list(verbs.keys()) if len(word) == 2]) / len(text))*1000, 2)


di_verb_result=[]
for file in tagged_files: 
    di_verb_result.append(di_verbs(file))

df['di_verbs'] = pd.Series(di_verb_result)


#Feature 54 nominalisation NOMZ 
#note that apart from noun-adjective, noun-verb, nominalisation 
#includes also verb plus genitive marker de 的

def nomz(text):
    def raw(text): 
        return str(text).count('noun-adjective')+str(text).count('noun-verb')
    def normalized(text): 
        return raw(text) / len(text)
    return round(normalized (text) * 1000, 2)        


nomz_result=[]
for file in tagged_files: 
    nomz_result.append(nomz(file))
    
#add verb plus genitive de to nominalisation count
def verb_de(text): 
    verb_list=list(map(list, zip([item for item in text if re.match(r'\bverb\b', item[1])], text[1:])))
    flat_list = [item for sublist in verb_list for item in sublist]
    return round ((flat_list.count(('的', 'particle'))/len(text))*1000, 2)

verb_de_result=[]
for file in tagged_files: 
    verb_de_result.append(verb_de(file))
    
df['NOMZ'] = pd.Series([x + y for x, y in zip(nomz_result, verb_de_result)])


print("Standardised frequencies of nominalisation written.")

df.to_csv(folder + 'linguistic_features.csv', index=False)

print("Completed. Standarised frequencies of all 54 features written.")