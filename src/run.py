import argparse

from preprocess import preprocess, preprocess_bert, add_widata
from train import train, train_bert
from test import test, test_raw, test_bert


def main(args):
    if args.clf == "bert":
        if args.mode == 'prepro':
            preprocess_bert(args)
        elif args.mode == 'train':
            train_bert(args)
        elif args.mode == 'test':
            test_bert(args)
    else:
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
    parser.add_argument('-data', required=True,
                        choices=['essay', 'textbook', 'wi'])
    parser.add_argument("-gec", default='',
                        choices=['stat', 'nn', 'correct', ''])
    parser.add_argument('-clf', default="mlp", choices=["mlp", "lr", "bert"])
    parser.add_argument('-gpus', default=0, type=int)
    parser.add_argument('-epoch', default=3000, type=int)
    parser.add_argument('-lr', default=2e-5, type=float)
    parser.add_argument('-model', default="../models/essay/test.pth")
    parser.add_argument('-wi', default=False, type=bool)
    parser.add_argument('-wo_ngram', default=False, type=bool)
    parser.add_argument('-embed', default=False, type=bool)
    parser.add_argument('-feature', default=False, type=bool)
    args = parser.parse_args()
    main(args)
