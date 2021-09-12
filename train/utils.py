from nltk.tokenize import sent_tokenize
import re
import treetaggerwrapper
from collections import Counter
from collections import OrderedDict
import numpy as np


class Surface:
    def __init__(self, text):
        self.text = text
        self.sentences = sent_tokenize(self.text.lower())
        # これを基に前処理を行う
        self.sen_length = len(self.sentences)
        # remove symbol
        self.rm_symbol_sentences = [re.sub("[!-/:-@[-`{-~]", "", sentence) for sentence in self.sentences]
        self.prop_sentences = [str(re.sub(r"([+-]?[0-9]+\.?[0-9]*)", "NUM", sentence)) for sentence in
                               self.rm_symbol_sentences]
        self.prop_words = ' '.join(self.prop_sentences).split()
        self.total_words = len(self.prop_words)
        self.word_types = set(self.prop_words)

        self.a1 = "../dat/a1.word"
        self.a2 = "../dat/a2.word"
        self.b1 = "../dat/b1.word"
        self.fun = "../dat/func.word"

        self.a1_words = []
        self.a2_words = []
        self.b1_words = []
        self.fun_words = []
        self.diff_words = []

        with open(self.a1) as fa1, open(self.a2) as fa2, open(self.b1) as fb1, open(self.fun) as ffn:
            for a1w in fa1:
                self.a1_words.append(a1w.lower().split()[0])
                self.diff_words.append(a1w.lower().split()[0])
            for a2w in fa2:
                self.a2_words.append(a2w.lower().split()[0])
                self.diff_words.append(a2w.lower().split()[0])
            for b1w in fb1:
                self.b1_words.append(b1w.lower().split()[0])
                self.diff_words.append(b1w.lower().split()[0])
            for funw in ffn:
                self.fun_words.append(funw.lower().split()[0])
                self.diff_words.append(funw.lower().split()[0])

    def stats(self):
        return [self.sen_length, self.total_words, float(len(self.word_types) / float(self.total_words))]

    def ngram(self):
        all_ngram = []
        xx = 1
        # for num in [1, 2, 3]:
        for num in [1, 2]:
            _ngrams = [list(zip(*(sentence.split()[i:] for i in range(num)))) for sentence in self.prop_sentences]
            ngrams = [flat for inner in _ngrams for flat in inner]
            all_ngram.extend(set(ngrams))
            '''
            for k,v in sorted(Counter(ngrams).items(), key=lambda x: -x[1]):
                if v < 5:
                    pass
                else:
                    print(xx, k, v, sep='\t')
                    xx += 1
            '''
        return Counter(all_ngram)

    def word_difficulty(self):
        '''
        !!!要修正!!!
        '''
        a1_ratio = len(self.word_types & set(self.a1_words)) / float(self.total_words)
        a2_ratio = len(self.word_types & set(self.a2_words)) / float(self.total_words)
        b1_ratio = len(self.word_types & set(self.b1_words)) / float(self.total_words)
        fun_ratio = len(self.word_types & set(self.fun_words)) / float(self.total_words)

        return [a1_ratio, a2_ratio, b1_ratio, fun_ratio]

    def features(self):
        ngrams = self.ngram()
        stats = self.stats()
        diff = self.word_difficulty()

        return ngrams, stats, diff


class GrmItem:
    def __init__(self, text):
        tagger = treetaggerwrapper.TreeTagger(TAGLANG='en', TAGDIR='/home/lr/hayashi/ra_web_app')
        self.text = text
        # 小文字にすると拾えない
        self.sentences = sent_tokenize(self.text)
        self.tagged = [tagger.TagText(sentence) for sentence in self.sentences]
        self.parsed = [' '.join(sentence).replace('\t', '_') for sentence in self.tagged]

        # 文法項目の読み込み
        self.grmlist = []
        self.num_grm_dic = {}
        self.num_list_dic = {}
        with open('../dat/grmitem.txt', 'r') as f:
            for num, i in enumerate(f, 1):
                self.grmlist.append(i.rstrip().split('\t')[1])
                self.num_grm_dic[num] = i.rstrip().split('\t')[1]
                self.num_list_dic[num] = i.rstrip().split('\t')[0]

    def detect(self, grmlist, itemslist):
        grm_dic = {}
        use_item = []
        for num, grm in enumerate(grmlist, 1):
            try:
                _grm_freq = [re.findall(grm, sentence) for sentence in self.parsed]
                grm_freq = [flat for inner in _grm_freq for flat in inner]
                if len(grm_freq) != 0:
                    grm_dic[num] = len(grm_freq)
                    use_item.append(itemslist[num])
            except:
                pass

        return grm_dic, use_item

    def pos_ngram(self):
        _tmp = []
        pos_list = []
        for sentence in self.tagged:
            # print(sentence)
            try:
                for word in sentence:
                    _tmp.append(str(word.split('\t')[1]))
                pos_list.append(' '.join(_tmp))
            except:
                pass
            _tmp = []

        all_pos_ngrams = []
        xx = 1
        # for num in [1, 2, 3]:
        for num in [1, 2]:
            _pos_ngrams = [list(zip(*(sentence.split()[i:] for i in range(num)))) for sentence in pos_list]
            pos_ngrams = [flat for inner in _pos_ngrams for flat in inner]
            all_pos_ngrams.extend(pos_ngrams)
            '''

            for k,v in sorted(Counter(pos_ngrams).items(), key=lambda x: -x[1]):
                if v < 5:
                    pass
                else:
                    print(xx, k, v, sep='\t')
                    xx += 1
            '''

        return Counter(all_pos_ngrams)

    def features(self):
        grmitem, use_list = self.detect(self.grmlist, self.num_list_dic)
        pos_ngram = self.pos_ngram()
        for k, v in grmitem.items():
            if v == 0:
                del (grmitem[k])

        return grmitem, pos_ngram, use_list


class Feature:
    def __init__(self, ngram={}, pos_ngram={}, grmitem={}, word_difficulty={}, stats={}):
        self.ngram = ngram
        self.pos_ngram = pos_ngram
        self.grmitem = grmitem
        self.word_difficulty = word_difficulty
        self.stats = stats
        self.word_dic = {}
        self.pos_dic = {}
        for line in open("../dat/word_essay.dat", "r"):
            self.word_dic[line.split('\t')[1]] = line.split('\t')[0]
        for line in open("../dat/pos_essay.dat", "r"):
            self.pos_dic[line.split('\t')[1]] = line.split('\t')[0]

    def ngram2vec(self):
        fdic = OrderedDict()
        # word ngram
        for feature in self.ngram:
            if str(feature) in self.word_dic:
                fdic[int(self.word_dic[str(feature)]) - 1] = self.ngram[feature] / float(self.stats[1])
            else:
                pass

        # pos ngram
        for feature in self.pos_ngram:
            if str(feature) in self.pos_dic:
                fdic[int(self.pos_dic[str(feature)]) - 1 + len(self.word_dic)] = self.pos_ngram[feature] / float(
                    self.stats[1])
            else:
                pass

        # grm item
        for key, value in self.grmitem.items():
            fdic[int(key) - 1 + len(self.pos_dic) + len(self.word_dic)] = value / float(self.stats[1])

        # word diff
        for number, feature in enumerate(self.word_difficulty, 0):
            # 501 is length of grm item
            fdic[number + int(501) + len(self.pos_dic) + len(self.word_dic)] = feature

        return fdic

    def concat(self):
        ngrams = self.ngram2vec()
        vec_size = 4 + int(501) + len(self.pos_dic) + len(self.word_dic)
        inputs = np.zeros([1, vec_size])

        for k, v in ngrams.items():
            inputs[0, k] = v

        return inputs[0]


def output(grade, stats, word_diff, grmitem):
    grade_class = {1: 'A1', 2: 'A2', 3: 'B1', 4: 'B2', 5: 'C1'}
    output_dic = {}
    output_dic['grade'] = grade_class[grade[0]]
    output_dic['stats'] = stats
    output_dic['word_diff'] = word_diff
    output_dic['grmitem'] = grmitem

    return output_dic
