import argparse

from preprocess import preprocess
from train import train
from test import test, test_raw


def main(args):
    if args.mode == 'prepro':
        preprocess(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'test':
        test_raw(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', required=True, choices=['prepro', 'train', 'test', 'test_raw'])
    parser.add_argument('-data', required=True, choices=['essay', 'textbook'])
    parser.add_argument("-gec", default='', choices=['stat', 'nn', 'correct', ''])
    parser.add_argument('-clf', default="nn", choices=["nn", "lr"])
    parser.add_argument('-gpus', default=0, type=int)
    parser.add_argument('-epoch', default=3000, type=int)
    parser.add_argument('-out', default="../model/essay/test.pth")
    parser.add_argument('--input')
    args = parser.parse_args()
    main(args)
