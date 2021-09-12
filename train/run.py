import argparse

from preprocess import preprocess
from train import train
from test import test, test_raw


def main(args):
    if args.MODE == 'prepro':
        preprocess(args)
    elif args.MODE == 'train':
        train(args)
    elif args.MODE == 'test':
        test(args)
    elif args.MODE == 'test':
        test_raw(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--MODE', required=True, choices=['prepro', 'train', 'train_gec', 'test', 'test_raw'])
    parser.add_argument('-d', '--data', required=True, choices=['essay', 'textbook'])
    parser.add_argument('-c', '--clf', default="nn", choices=["nn", "ll"])
    parser.add_argument('-g', '--visible_gpus', default=0, type=int)
    parser.add_argument('-o', '--OUT', default="../model/essay/test.pth")
    parser.add_argument('-i', '--INPUT')
    args = parser.parse_args()
    main(args)
