import argparse
from model import *
from lang import *
from parser import parse


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
                T = run(parse(code), "hello")
        except FileNotFoundError as fee:
            print(fee)
