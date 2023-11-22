import warnings
warnings.filterwarnings('ignore')
import argparse
from train_model import train_model
from test_model import test_models


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--prediction_file", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train_model(args.train_file)
    elif args.mode == "test":
        if args.prediction_file is not None:
            test_models(args.test_file)
        else:
            test_models(args.test_file)
    else:
        print("wrong mode")
