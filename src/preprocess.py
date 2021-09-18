import glob
import numpy as np
import random
import math
from collections import Counter
from sklearn.model_selection import train_test_split

from utils import Surface, GrmItem, Feature, output, GrmItem_gec, Feature_gec, extract_dp_sentence, detect_operate_pos

random.seed(22)


def preprocess(args):
    # To get files same shuffle order
    if args.data == "essay":
        output_path = "../cefrj/train/"
        files = glob.glob('../cefrj/original/**/*.raw')
    else:
        output_path = "../textbook/10_instance/train/"
        files = glob.glob('../textbook/10_instance/raw/**/*.dat')
    shuf_list = random.sample(files, len(files))

    if not args.gec:
        x = []
        y = []
        for dat in shuf_list:
            print(dat)
            data = ''
            with open(dat, 'r') as f:
                for i in f:
                    data += i.rstrip() + ' '

            surface = Surface(str(data))
            ngram, stats, diff = surface.features()
            grmitem = GrmItem(str(data))
            grm, pos_ngram, use_list = grmitem.features()
            inputs = Feature(ngram=ngram, pos_ngram=pos_ngram, grmitem=grm, word_difficulty=diff, stats=stats).concat()

            if 'A1' in dat:
                x.append(inputs)
                y.append(1)
            elif 'A2' in dat:
                x.append(inputs)
                y.append(2)
            elif 'B1' in dat:
                x.append(inputs)
                y.append(3)
            elif 'B2' in dat:
                x.append(inputs)
                y.append(4)
            elif 'C' in dat:
                x.append(inputs)
                y.append(5)
            # elif 'C2' in dat:
            #    y.append(6)

    # GEC
    else:
        print("GEC")
        assert args.data == "essay"

        # ファイルもってくる&シャッフルする
        # xmlファイルからもってくる
        if args.gec == "stat":
            output_path = '../cefrj/train_statgec/'
            for i in range(len(shuf_list)):
                shuf_list[i] = shuf_list[i].replace("original", "ori_statgec_pairs_xml")
                shuf_list[i] = shuf_list[i].replace("raw", "out")
        elif args.gec == "nn":
            output_path = '../cefrj/train_nngec/'
            for i in range(len(shuf_list)):
                shuf_list[i] = shuf_list[i].replace("original", "ori_nngec_pairs_xml")
        elif args.gec == "correct":
            output_path = '../cefrj/train_correct/'
            for i in range(len(shuf_list)):
                shuf_list[i] = shuf_list[i].replace("original", "ori_correct_pairs_xml")

        # 文法項目の読み込み
        grmlist = []  # TODO Grmitem以下にselfとして追加してあるんだけど別の場所で必要になってしまったため重複している
        num_grm_dic = {}
        num_list_dic = {}
        with open('../dat/grmitem.txt', 'r') as f:
            for num, i in enumerate(f, 1):
                grmlist.append(i.rstrip().split('\t')[1])
                num_grm_dic[num] = i.rstrip().split('\t')[1]
                num_list_dic[num] = i.rstrip().split('\t')[0]
        # 機能語読み込み
        # 素性として使うのでdic
        function_word_dic = {}
        with open("../dat/treetagger_function.word", "r") as f:
            for num, i in enumerate(f):
                function_word_dic[str(i.rstrip())] = str(num + 1)

        # 機能語品詞読み込み（treetagger:21種類)
        # 機能語だったら弾くのでリスト
        function_pos_list = []
        with open('../dat/treetagger_function.list', 'r') as f:
            for function_pos_word in f:
                function_pos_list.append(function_pos_word.rstrip())

        # 内容語読み込み（treetagger:37種類）
        # 素性として振り分けるのでdic
        content_pos_dic = {}
        with open('../dat/treetagger_content.list', 'r') as f:
            for num, i in enumerate(f):
                content_pos_dic[str(i.rstrip())] = str(num + 1)
        x = []
        y = []

        # xmlデータ読み込み（入力，出力，アライメント結果）
        for dat in shuf_list:
            print(dat)
            with open(dat, 'r') as f_xml:
                aligned, original, gec_out = extract_dp_sentence(f_xml)

            # 置換，脱落，余剰検出
            original_text = ''
            operation_features = []
            grmitem_features = []
            for ori_sen, gec_sen, dp_sen in zip(original, gec_out, aligned):
                original_text += ori_sen.capitalize() + ' '
                # 内容語品詞dic, 機能語単語dic, 機能語品詞リスト
                operations = detect_operate_pos(ori_sen, gec_sen, dp_sen, content_pos_dic, function_word_dic,
                                                function_pos_list)
                operation_features.extend(operations)

                grmitem = GrmItem_gec(str(ori_sen), str(gec_sen), str(dp_sen))
                use_grm = grmitem.compare(grmlist, num_list_dic)
                grmitem_features.extend(use_grm)

            # 頻度でまとめる
            operations_feat = dict(Counter(operation_features))
            grmitem_feat = dict(Counter(grmitem_features))
            # original文に対して行う
            surface = Surface(str(original_text))
            ngram, stats, diff = surface.features()
            grmitem = GrmItem(str(original_text))
            _, pos_ngram, use_list = grmitem.features()
            # 文法項目のinput
            inputs = Feature_gec(ngram=ngram, pos_ngram=pos_ngram, grmitem=grmitem_feat, word_difficulty=diff,
                                 stats=stats, operations=operations_feat).concat()

            # ngram_inputs = inputs[:26591 + 7933]
            # grm_gec_inputs = inputs[26591 + 7933:26591 + 7933 + 1002 + 732]
            # diff_inputs = inputs[26591 + 7933 + 1002 + 732:]

            if 'A1' in dat:
                x.append(inputs)
                y.append(1)
            elif 'A2' in dat:
                x.append(inputs)
                y.append(2)
            elif 'B1' in dat:
                x.append(inputs)
                y.append(3)

    length = len(y)
    ratio = [0.8, 0.1, 0.1]
    splits = [math.floor(ratio[0] * length), math.floor((ratio[0] + ratio[1]) * length)]

    with open(output_path + "train.csv", "w") as train, open(output_path + "dev.csv", "w") as dev, open(
            output_path + "test.csv", "w") as test:
        for i in range(length):
            if i < splits[0]:
                train.write(",".join(list(map(str, x[i]))) + "," + str(y[i]) + "\n")
            elif i < splits[1]:
                dev.write(",".join(list(map(str, x[i]))) + "," + str(y[i]) + "\n")
            else:
                test.write(",".join(list(map(str, x[i]))) + "," + str(y[i]) + "\n")

    print("all size: (", length, ", ", len(x[0]), ")")
    print("train size:", splits[0])
    print("dev size:", splits[1] - splits[0])
    print("test size:", length - splits[1])
