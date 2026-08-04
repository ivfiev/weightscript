from argparse import Namespace
from main import main

palindrome = "./test/palindrome.yml"
reverse = "./test/reverse.yml"
kv = "./test/kv.yml"


def run_test(file, input, expected):
    args = Namespace()
    args.run = file
    args.input = input
    actual = main(args)
    if actual != expected:
        raise Exception(f"{actual} != {expected}")


if __name__ == "__main__":
    run_test(reverse, "abc", "cba")
    run_test(reverse, "c0c0a", "a0c0c")
    run_test(reverse, "1", "1")
    run_test(reverse, "01|2|01", "10|2|10")
    run_test(reverse, "deca42", "24aced")
    run_test(palindrome, "abc", "0")
    run_test(palindrome, "c1e1c", "1")
    run_test(palindrome, "c2c1c", "0")
    run_test(palindrome, "1", "1")
    run_test(kv, "a1b2c0|abc", "120")
    run_test(kv, "a0b1c0|acb", "001")
    run_test(kv, "a2b1c0|aac", "220")
    run_test(kv, "c0a1b2|bca", "201")
    run_test(kv, "eadbc0|dec", "ba0")
    print("pass!")
