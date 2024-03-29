import re
from collections import Counter, OrderedDict

import numpy as np
import regex
import treetaggerwrapper
from nltk.tokenize import sent_tokenize

# 何回も呼ぶと遅いので
tagger = treetaggerwrapper.TreeTagger(
    TAGLANG="en", TAGDIR="/home/lr/hayashi/ra_web_app"
)


# 表層情報
class Surface:
    def __init__(self, text):
        self.text = text
        self.sentences = sent_tokenize(self.text.lower())
        # これを基に前処理を行う
        self.sen_length = len(self.sentences)
        # remove symbol
        self.rm_symbol_sentences = [
            re.sub("[!-/:-@[-`{-~]", "", sentence) for sentence in self.sentences
        ]
        self.prop_sentences = [
            str(re.sub(r"([+-]?[0-9]+\.?[0-9]*)", "NUM", sentence))
            for sentence in self.rm_symbol_sentences
        ]
        self.prop_words = " ".join(self.prop_sentences).split()
        self.total_words = len(self.prop_words)
        self.word_types = set(self.prop_words)

        # 単語難易度読み込み
        self.a1 = "../dat/a1.word"
        self.a2 = "../dat/a2.word"
        self.b1 = "../dat/b1.word"
        self.fun = "../dat/func.word"

        self.a1_words = []
        self.a2_words = []
        self.b1_words = []
        self.fun_words = []
        self.diff_words = []

        with open(self.a1) as fa1, open(self.a2) as fa2, open(self.b1) as fb1, open(
            self.fun
        ) as ffn:
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
        return [
            self.sen_length,
            self.total_words,
            float(len(self.word_types) / float(self.total_words)),
        ]

    def ngram(self):
        all_ngram = []
        for num in [1, 2]:
            _ngrams = [
                list(zip(*(sentence.split()[i:] for i in range(num))))
                for sentence in self.prop_sentences
            ]
            ngrams = [flat for inner in _ngrams for flat in inner]
            all_ngram.extend(set(ngrams))
        return Counter(all_ngram)

    def word_difficulty(self):
        """!!!要修正!!!"""
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
        self.text = text
        # 小文字にすると拾えない
        self.sentences = sent_tokenize(self.text)
        self.tagged = [tagger.TagText(sentence) for sentence in self.sentences]
        self.parsed = [
            " ".join(sentence).replace("\t", "_") for sentence in self.tagged
        ]

        # 文法項目の読み込み
        self.grmlist = []
        self.num_grm_dic = {}
        self.num_list_dic = {}
        with open("../dat/grmitem.txt", "r") as f:
            for num, i in enumerate(f, 1):
                self.grmlist.append(i.rstrip().split("\t")[1])
                self.num_grm_dic[num] = i.rstrip().split("\t")[1]
                self.num_list_dic[num] = i.rstrip().split("\t")[0]

    def detect(self, grmlist, itemslist):
        grm_dic = {}
        use_item = []
        for num, grm in enumerate(grmlist, 1):
            try:
                _grm_freq = [regex.findall(grm, sentence) for sentence in self.parsed]
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
            try:
                for word in sentence:
                    _tmp.append(str(word.split("\t")[1]))
                pos_list.append(" ".join(_tmp))
            except:
                pass
            _tmp = []

        all_pos_ngrams = []
        for num in [1, 2]:
            _pos_ngrams = [
                list(zip(*(sentence.split()[i:] for i in range(num))))
                for sentence in pos_list
            ]
            pos_ngrams = [flat for inner in _pos_ngrams for flat in inner]
            all_pos_ngrams.extend(pos_ngrams)

        return Counter(all_pos_ngrams)

    def features(self):
        grmitem, use_list = self.detect(self.grmlist, self.num_list_dic)
        pos_ngram = self.pos_ngram()
        for k, v in grmitem.items():
            if v == 0:
                del grmitem[k]

        return grmitem, pos_ngram, use_list


# GrmItem継承させる
# gec前後で文に対し抽出後差分を一行ずつ見る
class GrmItem_gec(GrmItem):
    # オリジナル，GEC後，DPマッチ
    def __init__(self, text, gec, dp):
        # 小文字にすると拾えない
        self.text = text.capitalize()
        self.gec = gec.capitalize()
        self.dp = dp
        self.tagged = tagger.TagText(self.text)
        self.tagged_gec = tagger.TagText(self.gec)
        self.parsed = [
            " ".join(self.tagged)
            .replace("\t", "_")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        ]
        self.parsed_gec = [
            " ".join(self.tagged_gec)
            .replace("\t", "_")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        ]
        # 文法項目の読み込み
        self.grmlist = []
        self.num_grm_dic = {}
        self.num_list_dic = {}
        with open("../dat/grmitem.txt", "r") as f:
            for num, i in enumerate(f, 1):
                self.grmlist.append(i.rstrip().split("\t")[1])
                self.num_grm_dic[num] = i.rstrip().split("\t")[1]
                self.num_list_dic[num] = i.rstrip().split("\t")[0]

    def compare(self):
        # grmlist(correct, error) * 2
        # 501種類に対して実行する
        correct_grm = []
        error_grm = []
        for num, grm in enumerate(self.grmlist, 1):
            try:
                # (正規表現を信じて) findallする
                grm_freq = regex.findall(grm, self.parsed[0])
                grm_freq_gec = regex.findall(grm, self.parsed_gec[0])

                if (len(grm_freq) != 0) and (len(grm_freq_gec) != 0):
                    correct_grm.append(num)
                # 修正前あり，修正前なし -> oms or msf
                elif (len(grm_freq) != 0) and (len(grm_freq_gec) == 0):
                    error_grm.append(num)
                # elif 修正前あり，修正後なし→誤用していた（文法項目+oms of msf)
                elif (len(grm_freq) == 0) and (len(grm_freq_gec) != 0):
                    error_grm.append(num)

            except:
                pass

        error_grm = [x + 501 for x in error_grm]
        output_grm = correct_grm + error_grm

        return output_grm

    def features(self):
        pos_ngram = self.pos_ngram()

        return pos_ngram


# 素性作成用
class Feature:
    def __init__(
        self, ngram={}, pos_ngram={}, grmitem={}, word_difficulty={}, stats={}
    ):
        self.ngram = ngram
        self.pos_ngram = pos_ngram
        self.grmitem = grmitem
        self.word_difficulty = word_difficulty
        self.stats = stats
        self.word_dic = {}
        self.pos_dic = {}
        for line in open("../dat/word_essay.dat", "r"):
            self.word_dic[line.split("\t")[1]] = line.split("\t")[0]
        for line in open("../dat/pos_essay.dat", "r"):
            self.pos_dic[line.split("\t")[1]] = line.split("\t")[0]

    def ngram2vec(self):
        # order: [ngram(26591), pos_ngram(7933), grm_item(501), word_difficulty(4)] -> 35029
        fdic = OrderedDict()
        # word ngram
        for feature in self.ngram:
            if str(feature) in self.word_dic:
                fdic[int(self.word_dic[str(feature)]) - 1] = self.ngram[
                    feature
                ] / float(self.stats[1])
            else:
                pass

        # pos ngram
        for feature in self.pos_ngram:
            if str(feature) in self.pos_dic:
                fdic[
                    int(self.pos_dic[str(feature)]) - 1 + len(self.word_dic)
                ] = self.pos_ngram[feature] / float(self.stats[1])
            else:
                pass

        # grm item
        for key, value in self.grmitem.items():
            fdic[int(key) - 1 + len(self.pos_dic) + len(self.word_dic)] = value / float(
                self.stats[1]
            )

        # word diff
        for number, feature in enumerate(self.word_difficulty, 0):
            # 501 is length of grm item
            fdic[number + 501 + len(self.pos_dic) + len(self.word_dic)] = feature

        return fdic

    def concat(self):
        ngrams = self.ngram2vec()
        vec_size = 4 + 501 + len(self.pos_dic) + len(self.word_dic)
        inputs = np.zeros([1, vec_size])

        for k, v in ngrams.items():
            inputs[0, k] = v

        return inputs[0]


class Feature_gec(Feature):
    def __init__(
        self,
        ngram={},
        pos_ngram={},
        grmitem={},
        word_difficulty={},
        stats={},
        operations={},
    ):
        self.ngram = ngram
        self.pos_ngram = pos_ngram
        self.grmitem = grmitem
        self.word_difficulty = word_difficulty
        self.stats = stats
        self.operations = operations
        self.word_dic = {}
        self.pos_dic = {}
        self.operation_dic = {}
        with open("../dat/word_essay.dat", "r") as f:
            for line in f:
                self.word_dic[line.split("\t")[1]] = line.split("\t")[0]
        with open("../dat/pos_essay.dat", "r") as f:
            for line in f:
                self.pos_dic[line.split("\t")[1]] = line.split("\t")[0]
        with open("../dat/treetagger_feature.dat", "r") as f:
            for num, line in enumerate(f, 1):
                self.operation_dic[num] = line.rstrip() + "(余剰)"
            for num, line in enumerate(f, 245):
                self.operation_dic[num] = line.rstrip() + "(脱落)"
            for num, line in enumerate(f, 489):
                self.operation_dic[num] = line.rstirip() + "(置換)"

    # grmitemが誤り対応，操作
    def ngram2vec(self):
        fdic = OrderedDict()
        # order: [ngram(26591), pos_ngram(7933), grm_item(501*2), operations(244*3), word_difficulty(4)] -> 36262
        # word ngram
        for feature in self.ngram:
            if str(feature) in self.word_dic:
                fdic[int(self.word_dic[str(feature)]) - 1] = self.ngram[
                    feature
                ] / float(self.stats[1])
            else:
                pass

        # pos ngram
        for feature in self.pos_ngram:
            if str(feature) in self.pos_dic:
                fdic[
                    int(self.pos_dic[str(feature)]) - 1 + len(self.word_dic)
                ] = self.pos_ngram[feature] / float(self.stats[1])
            else:
                pass

        # grm item(501種類)*4 に投げるようにする
        # ただし
        for key, value in self.grmitem.items():
            fdic[int(key) - 1 + len(self.pos_dic) + len(self.word_dic)] = value / float(
                self.stats[1]
            )

        # 誤り操作
        for key, value in self.operations.items():
            fdic[
                key - 1 + len(self.pos_dic) + len(self.word_dic) + 501 * 2
            ] = value / float(self.stats[1])

        # word diff
        for number, feature in enumerate(self.word_difficulty):
            fdic[
                number + len(self.pos_dic) + len(self.word_dic) + 501 * 2 + 244 * 3
            ] = feature

        return fdic

    def concat(self):
        ngrams = self.ngram2vec()
        vec_size = len(self.pos_dic) + len(self.word_dic) + 501 * 2 + 244 * 3 + 4
        inputs = np.zeros([1, vec_size])

        for k, v in ngrams.items():
            inputs[0, k] = v

        return inputs[0]


# 置換，脱落，余剰の操作抽出（内容語なら品詞，機能語なら単語）
# まずアライメントの情報を持ってきてからここに入れている
# POSはtreetaggerのposリストから
def detect_operate_pos(ori_sen, gec_sen, dp_sen):
    # 機能語読み込み
    # 素性として使うのでdic
    function_dic = {}
    with open("../dat/treetagger_function.word", "r") as f:
        for num, i in enumerate(f):
            function_dic[str(i.rstrip())] = str(num + 1)

    # 機能語品詞読み込み（treetagger:21種類)
    # 機能語だったら弾くのでリスト
    function_pos = []
    with open("../dat/treetagger_function.list", "r") as f:
        for function_pos_word in f:
            function_pos.append(function_pos_word.rstrip())

    # 内容語読み込み（treetagger:37種類）
    # 素性として振り分けるのでdic
    content_dic = {}
    with open("../dat/treetagger_content.list", "r") as f:
        for num, i in enumerate(f):
            content_dic[str(i.rstrip())] = str(num + 1)

    # [単語\t品詞\t原形, .... のような形式]
    # item.split('\t')[0] -> 単語
    # item.split('\t')[1] -> 品詞
    ori_tagged = tagger.TagText(ori_sen)
    gec_tagged = tagger.TagText(gec_sen)

    ori_sen_list = ori_sen.split()
    gec_sen_list = gec_sen.split()

    # tango.tango変のパターンが有るため変更
    ori_pos_list = []
    for x in ori_tagged:
        sp = x.split("\t")
        if len(sp) > 1:
            ori_pos_list.append(sp[1])
    gec_pos_list = []
    for x in gec_tagged:
        sp = x.split("\t")
        if len(sp) > 1:
            gec_pos_list.append(sp[1])

    # ori_pos_list = [x.split('\t')[1] for x in ori_tagged]
    # gec_pos_list = [x.split('\t')[1] for x in gec_tagged]

    # add/msf/oms_word = タグ付き<add>xxx</add>
    # 中身と単語を特定（機能語なら単語，内容語なら品詞）したい
    # add:gec後の文章から抽出する
    ori_w_tag = []
    gec_w_tag = []
    for word in dp_sen.split():
        if "<add>" in word:
            gec_w_tag.append(word)
        elif "<oms>" in word:
            ori_w_tag.append(word)
        elif "<msrcrr" in word:
            ori_w_tag.append(word)
            gec_w_tag.append(word)
        else:
            ori_w_tag.append(word)
            gec_w_tag.append(word)

    feature_len = len(content_dic) + len(function_dic)

    # 機能語単語リスト(207単語)×3 + 内容語リスト(37種類)×3
    # oms_func -> oms_content -> add -> msf
    # オリジナル（oms)
    out_list = []
    for ori_word, ori_tag_word, ori_pos in zip(ori_sen_list, ori_w_tag, ori_pos_list):
        if "<oms>" in ori_tag_word:
            # 機能語であれば単語
            if ori_pos in function_pos:
                if ori_word.lower() in function_dic.keys():
                    out_list.append(int(function_dic[ori_word.lower()]))
            # 内容語
            elif ori_pos in content_dic.keys():
                out_list.append(int(content_dic[ori_pos]) + len(function_dic))
            else:
                pass

    # gec後（add, msf)
    # msf(置換）は修正後のものを採用
    for gec_word, gec_tag_word, gec_pos in zip(gec_sen_list, gec_w_tag, gec_pos_list):
        if "<add>" in gec_tag_word:
            # 機能語であれば単語
            if gec_pos in function_pos:
                if gec_word.lower() in function_dic.keys():
                    out_list.append(int(function_dic[gec_word.lower()]) + feature_len)
            # 内容語
            elif gec_pos in content_dic.keys():
                out_list.append(
                    int(content_dic[gec_pos]) + len(function_dic) + feature_len
                )
            else:
                pass

        if "<msfcrr" in gec_tag_word:
            # 機能語であれば単語
            if gec_pos in function_pos:
                if gec_word.lower() in function_dic.keys():
                    out_list.append(
                        int(function_dic[gec_word.lower()]) + 2 * feature_len
                    )
            # 内容語
            elif gec_pos in content_dic.keys():
                out_list.append(
                    int(content_dic[gec_pos]) + len(function_dic) + 2 * feature_len
                )
            else:
                pass

    return out_list


def extract_dp_sentence(xml):  # xmlデータから入力，出力，アライメント情報を抽出
    dp_sentence = []
    ori_sentence = []
    correct_sentence = []
    tmp_line = ""
    for line_ in xml:
        line = line_.rstrip()
        if tmp_line == '<trial no="01a">':
            dp_sentence.append(line)
            tmp_line = line.rstrip()
        elif tmp_line == '<sentence psn="ns">':
            ori_sentence.append(line)
            tmp_line = line.rstrip()
        elif tmp_line == '<sentence psn="st">':
            correct_sentence.append(line)
            tmp_line = line.rstrip()
        else:
            tmp_line = line.rstrip()

    return dp_sentence, ori_sentence, correct_sentence


def get_gec_items(original, gec_out, aligned):
    # 置換，脱落，余剰検出
    original_text = ""
    operation_features = []
    grmitem_features = []
    for ori_sen, gec_sen, dp_sen in zip(original, gec_out, aligned):
        dp_sen = dp_sen.replace("<msf crr", "<msfcrr")
        original_text += ori_sen.capitalize() + " "
        # 内容語品詞dic, 機能語単語dic, 機能語品詞リスト
        operations = detect_operate_pos(ori_sen, gec_sen, dp_sen)  # [726, 704, 702]
        operation_features.extend(operations)

        use_grm = GrmItem_gec(
            str(ori_sen), str(gec_sen), str(dp_sen)
        ).compare()  # [143, 350, 880, 888]
        grmitem_features.extend(use_grm)
    # e.g.
    # operation_features: [211, 726, 704, 702, 702, 484, 702, 699]
    # grmitem_features: [21, 37, 38, 41, 57, 107, 111, 115, 143, 219, 269, 333, 350, 352, 361, 379, 385, 588, 592, 656, 748, 810, 822, 863, 876, 888]

    # 頻度でまとめる
    operations_feat = dict(Counter(operation_features))
    grmitem_feat = dict(Counter(grmitem_features))

    return operations_feat, grmitem_feat, original_text
