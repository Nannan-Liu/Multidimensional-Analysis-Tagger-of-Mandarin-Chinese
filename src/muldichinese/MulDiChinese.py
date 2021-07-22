#!/usr/bin/python3
# coding=UTF-8 #note capitalisation

#dependencies
import os
import pynlpir #use ICTCLAS to segment
import pandas as pd
from nltk.corpus import CategorizedPlaintextCorpusReader 
import statistics
import re
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv


class MulDiChinese:

    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise ValueError(
                f"MulDi Chinese did not find the files at: {file_path}")
        self.file_path = file_path

    def print_filepath(self):
        print(self.file_path)

    def first_five_files(self): 
        folder=self.file_path
        files=sorted([x for x in os.listdir(folder) if x.endswith(".txt")])
        file_names = [f.replace('.txt', '') for f in files]
        global df
        df = pd.DataFrame(file_names, columns=["text"])

        #convert your text into a corpus
        corpus = CategorizedPlaintextCorpusReader(
            self.file_path,
            r'(?!\.).*\.txt',
            cat_pattern=os.path.join(r'(neg|pos)', '.*',),
            encoding='utf-8')
        files=corpus.fileids()

        #create corpora
        global corpora
        raw_corpora=[corpus.raw(file) for file in files]
        corpora=[corpus.replace('\r', '').replace(' ', '').replace('\u3000', '') for corpus in raw_corpora]
        return (df.head())

    def pos(self):         
        #pos_tagging corpora
        pynlpir.open()
        global tag_corpora
        tag_corpora=[pynlpir.segment(file, pos_names='child') for file in corpora]
        pynlpir.close()
        #ICTCLAS has problems with some proper nouns and new nouns
        #they are not tagged, so need to be manually tagged
        #the following are words I found that ICTCLAS does not know
        unknown_NPs=['新华社', '新华网', '中新网', '人民网', '中国青年网', \
                              '中评社', '中国日报网', '南华早报', '国际在线',\
                              '南方日报', '法新社', '美联社', '路透社', '环球时报']
        unknown_nouns=['网民', '屌丝', '富帅', '飞机', '贴吧']
        unknown_nomzs=['派', '解构']
        unknown_numerals=['甲', '乙', '丙', '丁', '辰', '癸', '戊', '巳']
        unknown_adjs=['身份卑微']

        unknown_NP_tags=[(word, tag) for word, tag in dict(zip(unknown_NPs, [None]*len(unknown_NPs))).items()]
        unknown_noun_tags=[(word, tag) for word, tag in dict(zip(unknown_nouns, [None]*len(unknown_nouns))).items()]
        unknown_nomz_tags=[(word, tag) for word, tag in dict(zip(unknown_nomzs, [None]*len(unknown_nomzs))).items()]
        unknown_numeral_tags=[(word, tag) for word, tag in dict(zip(unknown_numerals, [None]*len(unknown_numerals))).items()]
        unknown_adj_tags=[(word, tag) for word, tag in dict(zip(unknown_adjs, [None]*len(unknown_adjs))).items()]
        for corpus in tag_corpora: 
            for index, item in enumerate(corpus):
                if item in unknown_NP_tags:
                    none_list = list(item)
                    none_list[1]= 'proper-noun'
                    corpus[index] = tuple(none_list)
                    tag_corpora.append(corpus)
                if item in unknown_noun_tags:
                    none_list = list(item)
                    none_list[1]= 'noun'
                    corpus[index] = tuple(none_list)
                    tag_corpora.append(corpus)
                if item in unknown_nomz_tags:
                    none_list = list(item)
                    none_list[1]= 'noun-verb'
                    corpus[index] = tuple(none_list)
                    tag_corpora.append(corpus)
                if item in unknown_numeral_tags:
                    none_list = list(item)
                    none_list[1]= 'numeral'
                    corpus[index] = tuple(none_list)
                    tag_corpora.append(corpus)
                if item in unknown_adj_tags:
                    none_list = list(item)
                    none_list[1]= 'adjective'
                    corpus[index] = tuple(none_list)
                    tag_corpora.append(corpus)

        print("POS tagging completed.")

    def features(self): 

        words=[[t[0] for t in text] for text in tag_corpora]
        tags=[[t[1] for t in text] for text in tag_corpora]
        tag_length=[len(tag) for tag in tags]

        #Feature 1 adverbial marker 地
        adverbial_marker_di_results=[round((tag.count('particle 地')/len(tag))*1000, 3) for tag in tags]
        df['adverbial_marker_di'] = pd.Series(adverbial_marker_di_results)

        #Feature 2 adverbs (RB)
        rb_results=[round((tag.count('adverb')/len(tag))*1000, 3) for tag in tags]
        df['RB'] = pd.Series(rb_results)

        #Feature 3 amplifiers
        amplifier_list=['非常', '大大', '十分', '真的', '真', '特别', '很', '最', '肯定', '挺', '顶', '极', '极为', '极其', '极度', '万分', '格外', '分外', '更', '更加', '更为', '尤其', '太', '过于', '老', '怪', '相当', '颇', '颇为', '有点儿', '有些', '最为', '还', '越发', '越加', '愈加', '稍', '稍微', '稍稍', '略', '略略', '略微', '比较', '较', '暴', '超', '恶', '怒', '巨', '粉', '奇', '很大', '相当', '完全', '显著', '总是', '根本', '一定']
        amplifiers_results=[round((len([x for x in word_list if x in amplifier_list])/len(word_list))*1000, 3) for word_list in words]
        df['AMP'] = pd.Series(amplifiers_results)

        #Feature 4 auxiliary adjectives
        aux_adj_results=[round((tag.count('auxiliary adjective')/len(tag))*1000, 3) for tag in tags]
        df['aux_adj'] = pd.Series(aux_adj_results)

        #Feature 5 average clause length
        clause_ends = set('，：；。？!……——')
        non_clause_ends = set('＂＃＄％＆＇（）＊＋－／＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿‘’‛“”„‟‧﹏.')
        words_non_clause_ends_removed=[[w for w in word_list if w not in non_clause_ends] for word_list in words]
        acl_results = [round((len(word_list) - sum(1 for item in word_list if item in clause_ends))  / float(sum(1 for item in word_list if item in clause_ends)),3) for word_list in words_non_clause_ends_removed]
        df['ACL'] = pd.Series(acl_results)

        #Feature 6 average word length
        awl_results=[round((sum(len(word) for word in word_list) / len(word_list)), 3) for word_list in words]
        df['AWL'] = pd.Series(awl_results)

        #Feature 7 average sentence length
        ends = set('。？!……——')
        nonends = set('＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿‘’‛“”„‟‧﹏.')
        words_nonends_removed=[[w for w in word_list if w not in nonends] for word_list in words]
        asl_results = [round((len(word_list) - sum(1 for item in word_list if item in ends))  / float(sum(1 for item in word_list if item in ends)),3) for word_list in words_nonends_removed]
        df['ASL'] = pd.Series(asl_results)

        #Feature 8 Chinese person names
        #personal name + Chinese - transcribed personal names
        Chinese_person_results=[round(sum(map(tag.count, ['personal name', 'Chinese given name', 'Chinese surname']))/len(tag)*1000, 3) for tag in tags]
        df['Chinese_person'] = pd.Series(Chinese_person_results)

        #Feature 9 classifiers 量词
        classifiers_results=[round((len([tag for tag in tagged_sent if re.match('.*classifier', tag)])/len(tagged_sent))*1000, 3) for tagged_sent in tags]
        df['classifier'] = pd.Series(classifiers_results)

        #Feature 10 classical grammatical words 文言文功能词
        classical_gram_list=['所', '将', '之', '于', '以']
        classical_grams_results=[round((len([x for x in word_list if x in classical_gram_list])/len(word_list))*1000, 3) for word_list in words]
        df['classical_gram'] = pd.Series(classical_grams_results)

        #Feature 11 classical syntax markers
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
        classical_syntax_results=[round((len([x for x in word_list if x in classical_syntax_markers])/len(word_list))*1000, 3) for word_list in words]
        df['classical_syntax'] = pd.Series(classical_syntax_results)

        #Feature 12 complement marker '得', 'particle 得'
        complement_marker_de_results=[round((tag.count('particle 得')/len(tag))*1000, 3) for tag in tags]
        df['complement_marker_de'] = pd.Series(complement_marker_de_results)

        #Feature 13 COND 条件连词、副词
        conditional_conjunct_list=['如果', '只有', '假如', '除非', '要是', '要不是', '只要', '假如', '倘若', '倘或', '设使', '设若', '如若', '若', '的话']
        de_shihou=['的', '时候']
        conditional_conjuncts_results=[round((len([x for x in word_list if x in conditional_conjunct_list])/len(word_list))*1000, 3) for word_list in words]
        de_shihou_results=[round(((''.join(word_list).count('的时候'))/len(word_list))*1000,3) for word_list in words]
        conditional_conjuncts_final_results=[sum(x) for x in zip(conditional_conjuncts_results, de_shihou_results)]
        df['COND'] = pd.Series(conditional_conjuncts_final_results)

        #Feature 14 demonstrative pronoun
        demp_results=[round((len([tag for tag in tagged_sent if re.match('.*demonstrative pronoun', tag)])/len(tagged_sent))*1000, 3) for tagged_sent in tags]
        df['DEMP'] = pd.Series(demp_results)

        #Feature 15 descriptive words #ICTCLAS status word
        status_words_results=[round((tag.count('status word')/len(tag))*1000, 3) for tag in tags]
        df['descriptive'] = pd.Series(status_words_results)

        #Feature 16 disyllabic negation 没有
        di_negation_results=[round((word_list.count('没有')/len(word_list))*1000, 3) for word_list in words]
        df['di_negation'] = pd.Series(di_negation_results)

        #Feature 17 disyllabic words 
        disyllabic_word_list=['购买', '具有', '在于', '寻找', '获得', '询问', '进入', '等候', '安定', \
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
        disyllabic_words_results=[round((len([x for x in word_list if x in disyllabic_word_list])/len(word_list))*1000, 3) for word_list in words]
        df['disyllabic_words'] = pd.Series(disyllabic_words_results)

        #Feature 18 disyllabic prepositions (BPIN)
        bpin_list=['按照', '本着', '按着', '朝着', '趁着', '出于', '待到', '对于', '根据', '关于', '基于', '鉴于', '借着', '经过', '靠着', '冒着', '面对', '面临', '凭借', '顺着', '随着', '通过', '为了', '围绕', '向着', '沿着', '依据']
        bpin_tags=[(word, tag) for word, tag in dict(zip(bpin_list, ['preposition']*len(bpin_list))).items()]
        bpin_results=[round((len([x for x in tuple_list if x in bpin_tags])/len(tuple_list))*1000, 3) for tuple_list in tag_corpora]
        df['BPIN'] = pd.Series(bpin_results)

        #feature 19 disyllabic verbs
        verb_indices=[[index for index,tag in enumerate(tagged_sent) if tag !='adverb' and re.match('.*verb.*', tag)] for tagged_sent in tags]
        all_verbs=[[words[num][i] for i in verb_indices[num]] for num in range(len(tag_corpora))]
        diverbs=[[verb for verb in verb_list if len(verb)==2] for verb_list in all_verbs]
        di_verbs_results=[round((x/y)*1000, 3) for x,y in zip([len(i) for i in diverbs],tag_length)]
        df['di_verbs'] = pd.Series(di_verbs_results)

        #Feature 20 downtoners DWNT
        downtoner_list=['一点', '一点儿', '有点', '有点儿', '稍', '稍微', '一些', '有些']
        downtoners_results=[round((len([x for x in word_list if x in downtoner_list])/len(word_list))*1000, 3) for word_list in words]
        df['DWNT'] = pd.Series(downtoners_results)

        #Feature 21 emotion words
        emotion_word_list=['烦恼', '不幸', '痛苦', '苦', '快乐', '忍', '喜', '乐', '称心', '痛快', \
        '得意', '欣慰', '高兴', '愉悦', '欣喜', '欢欣', '可意', '乐', '可心', '欢畅', '开心', '康乐', \
        '欢快', '快慰', '欢', '舒畅', '快乐', '快活', '欢乐', '畅快', '舒心', '舒坦', '欢娱', '如意', \
        '喜悦', '顺心', '欢悦', '舒服', '爽心', '晓畅', '松快', '幸福', '惊喜', '欢愉', '称意', '得志', \
        '情愿', '愿意', '欢喜', '振奋', '乐意', '留神', '乐于', '爱', '关怀', '偏爱', '珍爱', '珍惜', \
        '神往', '痴迷', '喜爱', '器重', '娇宠', '溺爱', '珍视', '喜欢', '动心', '挂牵', '赞赏', '爱好', \
        '满意', '羡慕', '赏识', '热爱', '钟爱', '眷恋', '关注', '赞同', '喜欢', '想', '挂心', '挂念', \
        '惦念', '挂虑', '怀念', '关切', '关心', '惦念', '牵挂', '怜悯', '同情', '吝惜', '可惜', '怜惜', \
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
        '慧黠', '狡黠', '安详', '仓皇', '阴冷', '阴沉', '犹豫', '好', '坏', '棒', '一般', '差', '得当', '标准']
        emotion_results=[round((len([x for x in word_list if x in emotion_word_list])/len(word_list))*1000, 3) for word_list in words]
        df['emotion'] = pd.Series(emotion_results)

        #Feature 22 exclamation mark
        exclamation_results=[round((word_list.count('！')/len(word_list))*1000, 3) for word_list in words]
        df['exclamation'] = pd.Series(exclamation_results)

        #Feature 23 EX有
        ex_results=[round((tag.count('verb 有')/len(tag))*1000, 3) for tag in tags]
        df['EX'] = pd.Series(ex_results)

        #Feature 24 FPP first person pronoun
        fpp_results=[round(((word_list.count('我')+word_list.count('我们'))/len(word_list))*1000, 3) for word_list in words]
        df['FPP'] = pd.Series(fpp_results)

        #Feature 25 hedges HDG
        hedge_list=['可能', '可以', '也许', '较少', '一些', '多个', '多为', '基本', '主要', '类似', '不少']
        hedges_results=[round((len([x for x in word_list if x in hedge_list])/len(word_list))*1000, 3) for word_list in words]
        df['HDG'] = pd.Series(hedges_results)

        #Feature 26 honourifics
        honourifics_list=['千金', '相公', '姑姥爷', '伯伯', '伯父', '伯母', '大伯', '大哥', '大姐', '大妈', '大爷', '大嫂', '嫂夫人', '大婶儿', '大叔', '大姨', '哥', '姐', '大娘', '妈妈', '奶 奶', '爷爷', '姨', '老伯', '老兄', '老爹', '老大爷', '老爷爷', '老太太', '老奶奶', '老大娘', '老板', '老公', '老婆婆', '老前辈', '老人家', '老师', '老师傅', '老寿星', '老太爷', '老翁', '老爷子', '老丈', '老总', '大驾', '夫人', '高徒', '高足', '官人', '贵客', '贵人', '嘉宾', '列位', '男士', '女士', '女主 人', '前辈', '台驾', '太太', '先生', '贤契', '贤人', '贤士', '先哲', '小姐', '学长', '爷', '诸位', '足下', '师傅', '师母', '师娘', '人士', '长老', '禅师', '船老大', '大师', '大师傅', '大王', '恩师', '法师', '法王', '佛爷', '夫子', '父母官', '国父', '麾下', '教授', '武师', '千 岁', '孺人', '圣母', '圣人', '师父', '王尊', '至尊', '座', '少奶奶', '少爷', '金枝玉叶', '工程师', '高级工程师', '经济师', '讲师', '教授', '副教授', '教师', '老师', '国家主席', '国家总理', '部长', '厅长', '市长', '局长', '科长', '校长', '烈士', '先烈', '先哲', '荣誉军人', '陛下', '殿下', '阁下', '阿公', '阿婆', '大人', '公', '公公', '娘子', '婆婆', '丈人', '师长', '义士', '勇士', '志士', '壮士', '学生', '兄弟', '小弟', '弟', '妹', '儿子', '女儿']
        honourifics_results=[round((len([x for x in word_list if x in honourifics_list])/len(word_list))*1000, 3) for word_list in words]
        df['honourifics'] = pd.Series(honourifics_results)

        #Feature 27 HSK core vocabulary level 1, 150 words 
        HSK1_list=['爱', '八', '爸爸', '杯子', '北京', '本', '不', '不客气', '菜', '茶', '吃', '出租车', '打电话', '大', '的', '点', '电脑', '电视', '电影', '东西', '都', '读', '对不起', '多', '多少', '儿子', '二', '饭店', '飞机', '分钟', '高兴', '个', '工作', '狗', '汉语', '好', '号', '喝', '和', '很', '后面', '回', '会', '几', '家', '叫', '今天', '九', '开', '看', '看见', '块', '来', '老师', '了', '冷', '里', '六', '妈妈', '吗', '买', '猫', '没关系', '没有', '米饭', '名字', '明天', '哪', '哪儿', '那', '呢', '能', '你', '年', '女儿', '朋友', '漂亮', '苹果', '七', '前面', '钱', '请', '去', '热', '人', '认识', '三', '商店', '上', '上午', '少', '谁', '什么', '十', '时候', '是', '书', '水', '水果', '睡觉', '说', '四', '岁', '他', '她', '太', '天气', '听', '同学', '喂', '我', '我们', '五', '喜欢', '下', '下午', '下雨', '先生', '现在', '想', '小', '小姐', '些', '写', '谢谢', '星期', '学生', '学习', '学校', '一', '一点儿', '衣服', '医生', '医院', '椅子', '有', '月', '再见', '在', '怎么', '怎么样', '这', '中国', '中午', '住', '桌子', '字', '昨天', '坐', '做']
        HSK1_results=[round((len([x for x in word_list if x in HSK1_list])/len(word_list))*1000, 3) for word_list in words]
        df['HSK_1'] = pd.Series(HSK1_results)

        #Feature 28 HSK core vocabulary level 3 (150-600), 450 words
        HSK3_list=['阿姨', '啊', '矮', '爱', '爱好', '安静', '把', '吧', '白', '百', '班', '搬', '办法', '办公室', '半', '帮忙', '帮助', '包', '饱', '报纸', '北方', '被', '鼻子', '比', '比较', '比赛', '笔记本', '必须', '变化', '别', '别人', '宾馆', '冰箱', '不但', '而且', '菜单', '参加', '草', '层', '差', '长', '唱歌', '超市', '衬衫', '成绩', '城市', '迟到', '出', '除了', '穿', '船', '春', '词典', '次', '聪明', '从', '错', '打篮球', '打扫', '打算', '大家', '带', '担心', '蛋糕', '当然', '到', '地', '得', '灯', '等', '地方', '地铁', '地图', '弟弟', '第一', '电梯', '电子邮件', '东', '冬', '懂', '动物', '短', '段', '锻炼', '对', '多么', '饿', '耳朵', '发', '发烧', '发现', '方便', '房间', '放', '放心', '非常', '分', '服务员', '附近', '复习', '干净', '感冒', '感兴趣', '刚才', '高', '告诉', '哥哥', '个子', '给', '根据', '跟', '更', '公共汽车', '公斤', '公司', '公园', '故事', '刮风', '关', '关系', '关心', '关于', '贵', '国家', '过', '过去', '还', '还是', '孩子', '害怕', '好吃', '黑', '黑板', '红', '后来', '护照', '花', '画', '坏', '欢迎', '环境', '换', '黄河', '回答', '会议', '火车站', '或者', '几乎', '机场', '机会', '鸡蛋', '极', '记得', '季节', '检查', '简单', '见面', '件', '健康', '讲', '教', '角', '脚', '教室', '接', '街道', '节目', '节日', '结婚', '结束', '姐姐', '解决', '介绍', '借', '进', '近', '经常', '经过', '经理', '久', '旧', '就', '句子', '决定', '觉得', '咖啡', '开始', '考试', '可爱', '可能', '可以', '渴', '刻', '客人', '课', '空调', '口', '哭', '裤子', '快', '快乐', '筷子', '蓝', '老', '累', '离', '离开', '礼物', '历史', '脸', '练习', '两', '辆', '聊天', '了解', '邻居', '零', '留学', '楼', '路', '旅游', '绿', '马', '马上', '卖', '满意', '慢', '忙', '帽子', '每', '妹妹', '门', '米', '面包', '面条', '明白', '拿', '奶奶', '男', '南', '难', '难过', '年级', '年轻', '鸟', '您', '牛奶', '努力', '女', '爬山', '盘子', '旁边', '胖', '跑步', '皮鞋', '啤酒', '便宜', '票', '瓶子', '妻子', '其实', '其他', '奇怪', '骑', '起床', '起飞', '起来', '千', '铅笔', '清楚', '晴', '请假', '秋', '去年', '裙子', '然后', '让', '热情', '认为', '认真', '日', '容易', '如果', '伞', '上班', '上网', '谁', '身体', '生病', '生气', '生日', '声音', '时间', '世界', '事情', '试', '手表', '手机', '瘦', '叔叔', '舒服', '树', '数学', '刷牙', '双', '水平', '说话', '司机', '送', '虽然', '但是', '它', '她', '太阳', '特别', '疼', '踢足球', '提高', '题', '体育', '甜', '条', '跳舞', '同事', '同意', '头发', '突然', '图书馆', '腿', '外', '完', '完成', '玩', '晚上', '碗', '万', '往', '忘记', '为', '为了', '为什么', '位', '文化', '问', '问题', '西', '西瓜', '希望', '习惯', '洗', '洗手间', '洗澡', '夏', '先', '相信', '香蕉', '向', '像', '小时', '小心', '校长', '笑', '新', '新闻', '新鲜', '信用卡', '行李箱', '姓', '熊猫', '休息', '需要', '选择', '雪', '颜色', '眼睛', '羊肉', '要求', '药', '要', '爷爷', '也', '一般', '一边', '一定', '一共', '一会儿', '一起', '一下', '一样', '一直', '已经', '以前', '意思', '因为', '所以', '阴', '音乐', '银行', '饮料', '应该', '影响', '用', '游戏', '游泳', '有名', '又', '右边', '鱼', '遇到', '元', '远', '愿意', '月亮', '越', '运动', '再', '早上', '站', '张', '丈夫', '着急', '找', '照顾', '照片', '照相机', '着', '真', '正在', '只', '只有', '才', '中间', '中文', '终于', '种', '重要', '周末', '主要', '注意', '准备', '自己', '自行车', '总是', '走', '嘴', '最', '最后', '最近', '左边', '作业']
        HSK3_results=[round((len([x for x in word_list if x in HSK3_list])/len(word_list))*1000, 3) for word_list in words]
        df['HSK_3'] = pd.Series(HSK3_results)

        #Feature 29 imperfect aspect 
        imperfect_marker_list=['着', '在', '正在', '起来', '下去']
        imperfect_results=[round((len([x for x in word_list if x in imperfect_marker_list])/len(word_list))*1000, 3) for word_list in words]
        df['imperfect'] = pd.Series(imperfect_results)

        #Feature 30 INPR indefinite pronouns 无定代词
        indefinite_list=['任何', '谁', '大家', '某', '有人', '有个', '什么']
        indefinites_results=[round((len([x for x in word_list if x in indefinite_list])/len(word_list))*1000, 3) for word_list in words]
        df['INPR'] = pd.Series(indefinites_results)

        #Feature 31 intransitive verbs
        vi_results=[round((tag.count('intransitive verb')/len(tag))*1000, 3) for tag in tags]
        df['intransitive'] = pd.Series(vi_results)

        #Feature 32 lexical density 
        #(noun+verb+adjective+adverb) / total
        verb_count=[len([tag for tag in tagged_sent if tag != 'adverb' and re.match('.*verb.*', tag)]) for tagged_sent in tags]
        noun_tag_list=['organization/group name', 'noun', 'noun morpheme', 'noun phrase', 'other proper noun', 'proper noun', 'noun of locality']
        noun_count=[len([tag for tag in tagged_sent if tag in noun_tag_list]) for tagged_sent in tags]
        adj_count=[len([tag for tag in tagged_sent if re.match('.*adjective.*', tag)]) for tagged_sent in tags]
        adv_count=[len([tag for tag in tagged_sent if re.match('.*adverb', tag)]) for tagged_sent in tags]
        lexical_counts=[a+b+c+d for a,b,c,d in zip(noun_count, verb_count, adj_count, adv_count)]
        lexical_density_results=[round((x/y)*1000, 3) for x,y in zip(lexical_counts,tag_length)]
        df['lexical_density'] = pd.Series(lexical_density_results)

        #Feature 33 modal particles and interjections 
        particles_results=[round(sum(map(tag.count, ['modal particle', 'interjection']))/len(tag)*1000, 3) for tag in tags]
        df['particle'] = pd.Series(particles_results)

        #Feature 34 modifying adverbs
        adverb_list=['也', '都', '又', '才', '就', '就是', '倒是', '越来越', '一边', '再', '甚至', '却', '原本', '只', '毕竟', '仍然', '反正', '刚', '常常', '已经', '就要']
        adverb_tags=[(word, tag) for word, tag in dict(zip(adverb_list, ['adverb']*len(adverb_list))).items()]+[('连', 'particle 连'), ('等', 'particle 等/等等/云云')]
        adverbs_results=[round((len([x for x in tuple_list if x in adverb_tags])/len(tuple_list))*1000, 3) for tuple_list in tag_corpora]
        df['modify_adv'] = pd.Series(adverbs_results)

        #Feature 35 monosyllabic negation 不、别、没
        mono_negation_list=['别', '不', '没']
        mono_negation_results=[round((len([x for x in word_list if x in mono_negation_list])/len(word_list))*1000, 3) for word_list in words]
        df['mono_negation'] = pd.Series(mono_negation_results)

        #Feature 36 monosyllabic verbs
        monoverbs=[[verb for verb in verb_list if len(verb)==1] for verb_list in all_verbs]
        mono_verbs_results=[round((x/y)*1000, 3) for x,y in zip([len(i) for i in monoverbs],tag_length)]
        df['mono_verbs'] = pd.Series(mono_verbs_results)

        #Feature 37 nominalisation NOMZ 
        #note that apart from noun-adjective, noun-verb
        #nominalisation includes also verb plus genitive marker de 的

        noun_adj_verb_results=[sum(map(tag.count, ['noun-adjective', 'noun-verb'])) for tag in tags]
        next_word_indices=[[(i+1) for i in verb_index] for verb_index in verb_indices]
        next_words=[[tags[num][i] for i in next_word_indices[num]] for num in range(len(tags))]
        verb_de_results=[len([i for i in word if i=='particle 的/底']) for word in next_words]
        nomz_results=[round(((x+y)/z)*1000, 3) for x,y,z in zip(noun_adj_verb_results,verb_de_results, tag_length)]
        df['NOMZ'] = pd.Series(nomz_results)

        #Feature 38 onomatopoeia 拟声词
        ono_results=[round((tag.count('onomatopoeia')/len(tag))*1000, 3) for tag in tags]
        df['onomatopoeia'] = pd.Series(ono_results)

        #Feature 39 other personal pronouns apart from FPP, SPP, TPP
        all_personal_pronouns_results=[round((tag.count('personal pronoun')/len(tag))*1000,3) for tag in tags]
        other_personal_results=[round(((a-b-c-d)/e)*1000, 3) for a,b,c,d,e in zip(all_personal_pronouns_results,fpp_results,spp_results,tpp_list,tag_length)]
        df['other_personal'] = pd.Series(other_personal_results)

        #Feature 40 perfect aspect (PEAS)
        peas_results=[round((sum(map(tag.count, ['particle 了/喽', '过', 'particle 过']))/len(tag))*1000, 3) for tag in tags]
        df['PEAS'] = pd.Series(peas_results)

        #Feature 41 private vervs
        private_verb_list=['三思', '三省', '主张', '了解', '亲信', '以为', '企图', '会意',\
                   '伤心', '估','估摸', '估算','估计', '估量','低估', '体会','体味', \
                   '信','信任', '信赖','修省', '假定','假想', '允许','关心', '关怀', \
                   '内省', '决定','决心', '决意', '决断', '决计', '准备', '准许','凝思', \
                   '凝想', '凭信', '分晓','切记', '划算','判断', '原谅','参悟', '反对',\
                   '反思', '反省', '发现', '发觉', '吃准', '合计', '合谋', '同情', \
                   '同意', '否认', '听信', '听到', '听见', '哭', '喜欢', '喜爱', \
                   '回味', '回忆', '回念', '回想', '回溯', '回顾', '图谋', '图', \
                   '坚信', '多疑', '失望', '失身', '妄图', '妄断', '宠信', '害怕', \
                   '察觉', '寻思', '尊敬', '尊重', '小心', '希望', '平静', '幻想', \
                   '当做', '彻悟', '得知', '忆', '忖度', '忖量', '忘', '忘却', '忘怀', \
                   '忘掉', '忘记', '快乐', '念', '忽略', '忽视', '怀念', '怀想', \
                   '怀疑', '怕', '思忖', '思想', '思索', '思维', '思考', '思虑', \
                   '思量', '恨', '悟', '悬想', '情知', '惊恐', '想', '想像', '想来', \
                   '想见', '想象', '愉快', '意会', '意想', '意料', '意识', '感到', \
                   '感动', '感受', '感悟', '感想', '感激', '感觉', '感觉', '感谢', \
                   '愤怒', '愿意', '懂', '懂得', '打算', '承想', '承认', '担心', \
                   '拥护', '捉摸', '掂掇', '掂量', '掌握', '推度', '推想', '推敲', \
                   '推断', '推测', '推理', '推算', '推见', '措意', '揆度', '揣度', \
                   '揣想', '揣摩', '揣摸', '揣测', '支持', '放心', '料想', '料', \
                   '斟酌', '断定', '明了', '明察', '明晓', '明白', '明知', '明确', \
                   '晓得', '权衡', '梦想', '欢迎', '欣赏', '武断', '死记', '沉思', \
                   '注意', '洞察', '洞彻', '洞悉', '洞晓', '洞达', '测度', '浮想', \
                   '淡忘', '深信', '深思', '深省', '深醒', '清楚', '清楚', '满意', \
                   '满足', '激动', '热爱', '熟悉', '熟知', '熟虑', '爱', '爱好', \
                   '牢记', '犯疑', '狂想', '狐疑', '猛醒', '猜', '猜度', '猜忌', \
                   '猜想', '猜测', '猜疑', '玄想', '理会', '理解', '琢磨', '生气', \
                   '生疑', '畅想', '留心', '留神', '疏忽', '疑', '疑心', '疑猜', \
                   '疑虑', '疼', '盘算', '相信', '盼望', '省察', '省悟', '看', \
                   '看到', '看见', '看透', '着想', '知', '知悉', '知晓', '知道', \
                   '确信', '确定', '确认', '空想', '立意', '笃信', '笑', '答应', \
                   '策划', '筹划', '筹算', '筹谋', '算', '算计', '粗估', '约摸', \
                   '置疑', '考虑', '考量', '联想', '腹诽', '臆度', '臆想', '臆断', \
                   '臆测', '自信', '自省', '蒙', '蓄念', '蓄谋', '衡量', '裁度', \
                   '要求', '观察', '觉察', '觉得', '觉悟', '觉醒', '警惕', '警觉', \
                   '计划', '计算', '计较', '认为', '认可', '认同', '认定', '认得', \
                   '认知', '认识', '讨厌', '记', '记取', '记得', '记忆', '设想', \
                   '识', '试图', '试想', '详悉', '误会', '误解', '谋划', '谋算', \
                   '谋虑', '赞同', '赞成', '走神', '起疑', '轻信', '轻视', '迷信', \
                   '迷信', '追忆', '追怀', '追思', '追想', '通彻', '通晓', '通', \
                   '遐想', '遗忘', '遥想', '酌情', '酌量', '醒', '醒悟', '重视', \
                   '铭记', '阴谋', '顾全', '顾及', '预卜', '预想', '预感', '预料', \
                   '预期', '预测', '预知', '预见', '预计', '预谋', '领会', '领悟', \
                   '领略', '高估', '高兴', '默认']
        private_verbs_results=[round((len([x for x in word_list if x in private_verb_list])/len(word_list))*1000, 3) for word_list in words]
        df['PRIV'] = pd.Series(private_verbs_results)

        #Feature 42 phrasal coordinations PHC 
        #same tags before and after phc
        phc_indices=[[index for index in (i for i,value in enumerate(tag) if value == 'coordinating conjunction')] for tag in tags]
        previous_tag_indices=[[(i-1) for i in phc_index] for phc_index in phc_indices]
        next_tag_indices=[[(i+1) for i in phc_index] for phc_index in phc_indices]
        previous_tags=[[tags[num][i] for i in previous_tag_indices[num]] for num in range(len(tags))]
        next_tags=[[tags[num][i] for i in next_tag_indices[num]] for num in range(len(tags))]
        same_tag_counts=[len(set(a) & set(b)) for a,b in zip(previous_tags, next_tags)]
        phc_results=[round((x/y)*1000, 3) for x,y in zip(same_tag_counts,tag_length)]
        df['PHC'] = pd.Series(phc_results)

        #Feature 43 public verbs 
        public_verb_list=['表示', '称', '道', '说', '讲', '质疑', '认为', '坦言', '指出', '告诉', '呼吁', '解释', '问', '建议']
        public_verbs_results=[round((len([x for x in word_list if x in public_verb_list])/len(word_list))*1000, 3) for word_list in words]
        df['PUBV'] = pd.Series(public_verbs_results)

        #Feature 44 questions
        question_results=[round((word_list.count('？')/len(word_list))*1000, 3) for word_list in words]
        df['question'] = pd.Series(question_results)

        #Feature 45 SPP second person pronoun
        spp_results=[round((sum(map(word_list.count, ['你', '你们', '您','您们']))/len(word_list))*1000, 3) for word_list in words]    
        df['SPP'] = pd.Series(spp_results)

        #Feature 46 SMP (seem, appear)
        smp_list=['好像', '好象', '似乎', '貌似']
        smps_results=[round((len([x for x in word_list if x in smp_list])/len(word_list))*1000, 3) for word_list in words]
        df['SMP'] = pd.Series(smps_results)

        #Feature 47 Be 是
        be_results=[round((tag.count('verb 是')/len(tag))*1000, 3) for tag in tags]
        df['BE'] = pd.Series(be_results)

        #feature 48 similie
        simile_1_counts=[tag.count('particle 一样/一般/似的/般') for tag in tags]
        simile_word_list=['仿佛', '宛若', '如', '像']
        simile_2_counts=[len([x for x in word_list if x in simile_word_list]) for word_list in words]
        simile_results=[round(((x+y)/z)*1000, 3) for x,y,z in zip(simile_1_counts,simile_2_counts,tag_length)]
        df['simile'] = pd.Series(simile_results)

        #Features 49 standard deviation of sentence length
        all_sentences=[]
        for word_list in words_nonends_removed: 
            sentences = [[]]
            for word in word_list:
                if word in ends: sentences.append([])
                else: sentences[-1].append(word)
            
            if sentences[0]:
                if not sentences[-1]: sentences.pop() 
                all_sentences.append(sentences)
        
        sl_std_results=[round(statistics.stdev([len(s) for s in corpus_sent]), 3) for corpus_sent in all_sentences]
        df['SL_std'] = pd.Series(sl_std_results)

        #Feature 50 third person pronouns
        tpp_list=['她', '他', '它', '她们', '他们', '它们']
        tpps_results=[round((len([x for x in word_list if x in tpp_list])/len(word_list))*1000, 3) for word_list in words]
        df['TPP'] = pd.Series(tpps_results)

        #Feature 51 nouns
        nouns_results=[round((x/y)*1000, 3) for x,y in zip(noun_count,tag_length)]
        df['noun'] = pd.Series(nouns_results)

        #Feature 52 unique items 
        unique_words_results=[round((len([x for x in word_list if word_list.count(x)==1]) / len(word_list))*1000, 3) for word_list in words]
        df['unique'] = pd.Series(unique_words_results)

        #Feature 53 wh words 
        wh_results=[round((len([tag for tag in tagged_sent if re.match('.*interrogative pronoun', tag)])/len(tagged_sent))*1000, 3) for tagged_sent in tags]
        df['WH'] = pd.Series(wh_results)

        df.to_csv(self.file_path + 'linguistic_features.csv', index=False)
        print("Standarised frequencies of all 53 features written.")

    #create dataframe from standardised feature frequencies
    def dimensions(self): 
        df=pd.read_csv(self.file_path + 'linguistic_features.csv', header=0, index_col=0, quoting=csv.QUOTE_NONE)
        stdsc = StandardScaler()
        stats_std = stdsc.fit_transform(df)
        stats = pd.DataFrame(data=np.array(stats_std), index=df.index, columns=df.columns)
        text_list=list(stats.index.values)

        dimension1_scores=[(stats.loc[text, 'question']+stats.loc[text, 'particle']+stats.loc[text, 'exclamation']+stats.loc[text, 'SPP']+stats.loc[text, 'WH']+stats.loc[text, 'PUBV']+stats.loc[text, 'mono_negation']+stats.loc[text, 'Chinese_person']+stats.loc[text, 'honourifics']+stats.loc[text, 'FPP']+stats.loc[text, 'INPR']+stats.loc[text, 'emotion']-stats.loc[text, 'AWL']) for text in text_list]
        dimension1_scores=[round(x, 3) for x in dimension1_scores]

        dimension2_scores=[(stats.loc[text, 'descriptive']+stats.loc[text, 'imperfect']+stats.loc[text, 'adverbial_marker_di']+stats.loc[text, 'simile']+stats.loc[text, 'PEAS']+stats.loc[text, 'onomatopoeia']+stats.loc[text, 'TPP']+stats.loc[text, 'classifier']+stats.loc[text, 'mono_verbs']+stats.loc[text, 'SMP']+stats.loc[text, 'complement_marker_de']) for text in text_list]
        dimension2_scores=[round(x, 3) for x in dimension2_scores]

        dimension3_scores=[(stats.loc[text, 'BE']+stats.loc[text, 'COND']+stats.loc[text, 'modify_adv']+stats.loc[text, 'AMP']+stats.loc[text, 'EX']+stats.loc[text, 'HDG']+stats.loc[text, 'DWNT']+stats.loc[text, 'RB']+stats.loc[text, 'di_negation']+stats.loc[text, 'HSK_3']+stats.loc[text, 'DEMP']+stats.loc[text, 'PRIV']+stats.loc[text, 'HSK_1']+stats.loc[text, 'other_personal']-stats.loc[text, 'noun']) for text in text_list]
        dimension3_scores=[round(x, 3) for x in dimension3_scores]

        dimension4_scores=[(stats.loc[text, 'ACL']+stats.loc[text, 'ASL']+stats.loc[text, 'SL_std']+stats.loc[text, 'PHC']+stats.loc[text, 'BPIN']-stats.loc[text, 'unique']-stats.loc[text, 'intransitive']-stats.loc[text, 'di_verbs']) for text in text_list]
        dimension4_scores=[round(x, 3) for x in dimension4_scores]

        dimension5_scores=[(stats.loc[text, 'classical_gram']+stats.loc[text, 'classical_syntax']-stats.loc[text, 'aux_adj']-stats.loc[text, 'lexical_density']-stats.loc[text, 'NOMZ']-stats.loc[text, 'disyllabic_words']) for text in text_list]
        dimension5_scores=[round(x, 3) for x in dimension5_scores]

        #create dataframe for dimension scores
        d = {'text': text_list, 'dimension1':dimension1_scores, 'dimension2':dimension2_scores,'dimension3':dimension3_scores,'dimension4':dimension4_scores,'dimension5':dimension5_scores}
        dimension_scores = pd.DataFrame(data=d)

        #write dimension scores
        dimension_scores.to_csv(self.file_path + 'dimension_scores.csv', index=False)

        print("Dimension scores written.")