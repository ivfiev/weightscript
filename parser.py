import sys
import traceback

from lang import *
from model import P, E

LINE_NUM = 0


def parse(code: str) -> list[list]:
    try:
        lines = [s.strip(" \t") for s in code.split("\n")]
        parsed = []
        fa = FeatureAllocator()
        while lines:
            line = first(lines)
            match line:
                case "Features:":
                    parse_features(lines, fa)
                case "Block:":
                    parsed.append(parse_block(lines, fa))
                case "Unembed:":
                    parsed.append(parse_unembed(lines, fa))
                case _:
                    continue
        return parsed
    except Exception as e:
        traceback.print_exc()
        return fail(e)


def parse_features(lines: list[str], fa: FeatureAllocator):
    while line := first(lines):
        key, size = line.split(": ")
        r = len(P) if size == "number" else len(E) if size == "char" else 1
        fa.alloc(key, r)
        if lines[0] == "":
            break


def parse_block(lines: list[str], fa: FeatureAllocator) -> list:
    parsed = []
    line = first(lines)
    if line == "Attention:":
        parsed.append(parse_attention(lines, fa))
    else:
        fail(f"unexpected '{line}'")
    line = first(lines)
    if line == "FeedForward:":
        parsed.append(parse_feedforward(lines, fa))
    else:
        fail(f"unexpected '{line}'")
    return parsed


def parse_attention(lines: list[str], fa: FeatureAllocator):
    parsed = []
    line = first(lines)
    if line == "- Head:":
        q = first(lines).split()
        k = first(lines).split()
        v = first(lines).split()
        p = first(lines).split()
        match [q, k, v, p]:
            case [["Query:", q], ["Key:", k], ["Value:", v], ["Proj:", p]]:
                parsed.append(
                    [
                        ["QUERY", resolve(fa, q)],
                        ["KEY", resolve(fa, k)],
                        ["VALUE", resolve(fa, v)],
                        ["PROJ", resolve(fa, p)],
                    ]
                )
    return parsed


def parse_feedforward(lines: list[str], fa: FeatureAllocator):
    parsed = []
    while (line := first(lines)).startswith("-"):
        w = line.split()
        if "-=" in w:
            x0, x1 = var(fa, w[1])
            y0, y1 = var(fa, w[3])
            parsed.append(sub_one_hot(x0, y0, x1))
    return parsed


def parse_unembed(lines: list[str], fa: FeatureAllocator):
    parsed = []
    while line := first(lines):
        [k, v] = line.split(": ")
        match k:
            case "Char":
                parsed.append(["CHAR", resolve(fa, v)])
            case "Tokens":
                parsed.append(v)
    return parsed


def resolve(fa: FeatureAllocator, f: str) -> list | tuple:
    match f:
        case "POS":
            return slice(fa.POS, len(P))
        case "EMB":
            return slice(fa.EMB, len(E))
        case _ if len(f) == 3 and f[0] == "'" and f[2] == "'":
            c = f[1]
            if c.isdigit():
                return (fa.POS, len(P), one_hot(len(P), c))
            else:
                return (fa.EMB, len(E), one_hot(len(E), c))
        case _ if info := fa.info(f):
            return [info[0]]
        case _:
            return fail(f"unknown feature '{f}'")


def fail(e):
    print(f"{e} at line {LINE_NUM}", file=sys.stderr, flush=True)
    sys.exit(1)


def first(lines: list) -> str:
    global LINE_NUM
    LINE_NUM += 1
    return lines.pop(0)


def var(fa: FeatureAllocator, key) -> tuple:
    x = fa.info(key)
    if not x:
        return fail(f"unknown variable '{key}'")
    return x
