import argparse

from preprocess import preprocess, add_widata
from train import train
from test import test, test_raw


def main(args):
    if args.mode == 'prepro':
        preprocess(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'test_raw':
        test_raw(args)
    elif args.mode == 'add_wi':
        add_widata(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True,
                        choices=['prepro', 'train', 'test', 'test_raw', 'add_wi', ''])
    parser.add_argument('-data', required=True, choices=['essay', 'textbook'])
    parser.add_argument("-gec", default='',
                        choices=['stat', 'nn', 'correct', ''])
    parser.add_argument('-clf', default="nn", choices=["nn", "lr"])
    parser.add_argument('-gpus', default=0, type=int)
    parser.add_argument('-epoch', default=3000, type=int)
    parser.add_argument('-model', default="../models/essay/test.pth")
    parser.add_argument('-wi', default=False, type=bool)
    parser.add_argument('-wo_ngram', default=False, type=bool)
    parser.add_argument('--input')
    args = parser.parse_args()
    main(args)
