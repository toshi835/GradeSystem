# データごとの統計情報を調べる
# 1ファイルごとの文の平均, 中央値, min, max
# 1文ごとの単語数の平均, 中央値, min, max
import glob
import statistics

uncount_words = [[",", ""], [".\n", "\n"], ["!\n", "\n"], ["?\n", "\n"]]
data_types = ["JEFLL",
              "WI+LOCNESS",
              "Textbook",
              "Textbook_before",
              "Textbook_Original"]
paths = ["../essay/original/*/*.raw",
         "../essay/wi+locness/original/*/*.raw",
         "../textbook/paragraph/raw/*/*.dat",
         "../textbook/10_instance/raw/*/*.dat",
         "../textbook/raw/*.raw", ]
for data_type, path in zip(data_types, paths):
    num_sen = []
    num_words = []
    files = glob.glob(path)
    # if data_type != "WI+LOCNESS":  # "JEFLL":
    #    continue
    assert files
    for file in files:
        with open(file, "r") as f:
            text = f.read()
            for un_wo, tr_wo in uncount_words:
                text = text.replace(un_wo, tr_wo)
            #sens = text.strip().split("\n")
            sens = list(filter(lambda x: x.strip(), text.split('\n')))
            num_sen.append(len(sens))
            # if len(sens) < 5:
            #    print(text)
            #    print(file)
            for sen in sens:
                if sen == "":
                    continue
                words = sen.strip().split(" ")
                num_words.append(len(words))

    print("----------Static Information----------")
    print("{}".format(data_type))
    print("Sentence")
    print("Max: {}".format(max(num_sen)))
    print("Min: {}".format(min(num_sen)))
    print("Mean: {:.02f}".format(sum(num_sen)/len(num_sen)))
    print("Median: {}".format(statistics.median(num_sen)))
    print("Sum: {}".format(sum(num_sen)))
    print()
    print("Word")
    print("Max: {}".format(max(num_words)))
    print("Min: {}".format(min(num_words)))
    print("Mean: {:.02f}".format(sum(num_words)/len(num_words)))
    print("Median: {}".format(statistics.median(num_words)))
    print("Sum: {}".format(sum(num_words)))
    print("--------------------------------------")
