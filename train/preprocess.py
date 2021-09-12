import glob
import numpy as np
import random
import math
from sklearn.model_selection import train_test_split

from utils import Surface, GrmItem, Feature, output


def preprocess(args):
    if args.data == "textbook":
        PATH = "../textbook/10_instance/"
        files = glob.glob(PATH + 'raw/**/*.dat')
    else:
        PATH = '../cefrj/'
        files = glob.glob(PATH + 'original/**/*.raw')

    shuf_list = random.sample(files, len(files))
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
        # elif 'C1' in dat:
        #    y.append(5)
        # elif 'C2' in dat:
        #    y.append(6)

    length = len(y)
    ratio = [0.8, 0.1, 0.1]
    split = [math.floor(ratio[0] * length), math.floor((ratio[0] + ratio[1]) * length)]

    cnt1, cnt2, cnt3 = 0, 0, 0
    with open(PATH + "train/train.csv", "w") as train, open(PATH + "train/dev.csv", "w") as dev, open(
            PATH + "train/test.csv", "w") as test:
        for i in range(length):
            if i < split[0]:
                train.write(", ".join(list(map(str, x[i]))) + ", " + str(y[i]) + "\n")
                cnt1 += 1
            elif i < split[1]:
                dev.write(", ".join(list(map(str, x[i]))) + ", " + str(y[i]) + "\n")
                cnt2 += 1
            else:
                test.write(", ".join(list(map(str, x[i]))) + ", " + str(y[i]) + "\n")
                cnt3 += 1

    print("all size: (", length, ", ", len(x[0]), ")")
    print("train size:", cnt1)
    print("dev size:", cnt2)
    print("test size:", cnt3)
