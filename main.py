import argparse
from model import *
from lang import *
from parser import parse_transformer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", type=str, help="path to program")
    parser.add_argument("--run", type=str, help="path to program")
    parser.add_argument("--input", type=str, help="program input")
    args = parser.parse_args()
    return args


def main(args):
    if args.run and args.input:
        try:
            with open(args.run, "r") as f:
                code = f.read()
                t = parse_transformer(code)
                return t(f"^{args.input}$")
        except FileNotFoundError as fee:
            print(fee)
    else:
        print("Usage: --run %filename% --input %string%")


if __name__ == "__main__":
    result = main(get_args())
    print(f"Output: {result}")
