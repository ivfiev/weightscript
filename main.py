import argparse
from model import *
from lang import *
from parser import parse_T


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", type=str, help="path to program")
    parser.add_argument("--run", type=str, help="path to program")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.run:
        try:
            with open(args.run, "r") as f:
                code = f.read()
                T = parse_T(code)
                print(T("^abba$"))
        except FileNotFoundError as fee:
            print(fee)

    # todo maybe rid of the list ast
