import numpy as np


def show_output(grade, stats, word_diff, grmitem):
    grade_class = {1: 'A1', 2: 'A2', 3: 'B1', 4: 'B2', 5: 'C1'}
    output_dic = dict()
    output_dic['grade'] = grade_class[grade[0]]
    output_dic['stats'] = stats
    output_dic['word_diff'] = word_diff
    output_dic['grmitem'] = grmitem

    return output_dic


def data_loader(PATH):
    print("loaded from ", PATH)
    x = []
    y = []
    with open(PATH, "r") as f:
        for line in f:
            lines = list(map(float, line.split(",")))
            x.append(np.array(lines[:-1]))
            y.append(int(lines[-1]))

    x = np.array(x)
    y = np.array(y)
    print("x.shape", x.shape)
    print("y.shape", y.shape)
    return x, y
