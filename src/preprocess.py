import glob
import random
import math

from prepro_utils import Surface, GrmItem, Feature, Feature_gec, extract_dp_sentence, get_gec_items

random.seed(22)


def preprocess(args):
    # To get files same shuffle order
    if args.data == "essay":
        output_path = "../essay/train/"
        files = glob.glob('../essay/original/**/*.raw')
    else:
        output_path = "../textbook/train/"
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

            surface = Surface(data)
            ngram, stats, diff = surface.features()
            grmitem = GrmItem(data)
            grm, pos_ngram, _ = grmitem.features()
            inputs = Feature(ngram=ngram, pos_ngram=pos_ngram,
                             grmitem=grm, word_difficulty=diff, stats=stats).concat()
            x.append(inputs)
            if 'A1' in dat:
                y.append(1)
            elif 'A2' in dat:
                y.append(2)
            elif 'B1' in dat:
                y.append(3)
            elif 'B2' in dat:
                y.append(4)
            elif 'C' in dat:
                y.append(5)

    # GEC
    else:
        print("GEC")
        assert args.data == "essay"

        # change path name
        if args.gec == "stat":
            output_path = '../essay/train_statgec/'
            for i in range(len(shuf_list)):
                shuf_list[i] = shuf_list[i].replace(
                    "original", "ori_statgec_pairs_xml")
                shuf_list[i] = shuf_list[i].replace("raw", "out")
        elif args.gec == "nn":
            output_path = '../essay/train_nngec/'
            for i in range(len(shuf_list)):
                shuf_list[i] = shuf_list[i].replace(
                    "original", "ori_nngec_pairs_xml")
        elif args.gec == "correct":
            output_path = '../essay/train_correct/'
            for i in range(len(shuf_list)):
                shuf_list[i] = shuf_list[i].replace(
                    "original", "ori_correct_pairs_xml")

        # loading xml（input，output，alignment）
        x = []
        y = []
        for dat in shuf_list:
            print(dat)
            with open(dat, 'r') as f_xml:
                aligned, original, gec_out = extract_dp_sentence(f_xml)

            operations_feat, grmitem_feat, original_text = get_gec_items(
                original, gec_out, aligned)

            # original文に対して行う
            ngram, stats, diff = Surface(original_text).features()
            pos_ngram = GrmItem(original_text).features()
            # 文法項目のinput
            inputs = Feature_gec(ngram=ngram, pos_ngram=pos_ngram, grmitem=grmitem_feat, word_difficulty=diff,
                                 stats=stats, operations=operations_feat).concat()

            x.append(inputs)
            if 'A1' in dat:
                y.append(1)
            elif 'A2' in dat:
                y.append(2)
            elif 'B1' in dat:
                y.append(3)

    # save data to csv
    length = len(y)
    ratio = [0.8, 0.1, 0.1]
    splits = [math.floor(ratio[0] * length),
              math.floor((ratio[0] + ratio[1]) * length)]

    with open(output_path + "train.csv", "w") as train, open(output_path + "dev.csv", "w") as dev, open(
            output_path + "test.csv", "w") as test:
        for i in range(length):
            if i < splits[0]:
                train.write(
                    ",".join(list(map(str, x[i]))) + "," + str(y[i]) + "\n")
            elif i < splits[1]:
                dev.write(
                    ",".join(list(map(str, x[i]))) + "," + str(y[i]) + "\n")
            else:
                test.write(
                    ",".join(list(map(str, x[i]))) + "," + str(y[i]) + "\n")

    print("all size: (", length, ", ", len(x[0]), ")")
    print("train size:", splits[0])
    print("dev size:", splits[1] - splits[0])
    print("test size:", length - splits[1])
