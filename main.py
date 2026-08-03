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


def main(args):
    if args.run and args.input:
        try:
            with open(args.run, "r") as f:
                code = f.read()
                t = parse_T(code)
                # print(T("^abba$"))
                return t(f"^{args.input}$")
        except FileNotFoundError as fee:
            print(fee)


if __name__ == "__main__":
    main(get_args())
