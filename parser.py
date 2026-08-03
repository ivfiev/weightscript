import sys
import traceback

from lang import *
from model import P, E

LINE_NUM = 0


def parse_transformer(code: str) -> Transformer:
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
        return build_transformer(parsed, lambda m, s: print_columns(fa, m, s))
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
    while lines[0].startswith("-"):
        line = first(lines)
        if line == "-":
            break
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
        match w[1:]:
            case [xpp] if xpp.endswith("++"):
                x, size = var(fa, xpp[:-2])
                parsed.extend(inc_one_hot(x, size))
            case [y, _, "not", x]:
                y, _ = var(fa, y)
                x, _ = var(fa, x)
                parsed.append(["NOT", [x], [y]])
            case [y, _, "and", xs]:
                y, _ = var(fa, y)
                xs, _ = zip(*var(fa, w[4].split(",")))
                parsed.append(["AND", xs, [y]])
            case [y, _, "nand", xs]:
                y, _ = var(fa, y)
                xs, _ = zip(*var(fa, w[4].split(",")))
                parsed.append(["NAND", xs, [y]])
            case [y, _, "nor", xs]:
                y, _ = var(fa, y)
                xs, _ = zip(*var(fa, w[4].split(",")))
                parsed.append(["NOR", xs, [y]])
            case [y, _, u, "!=", v]:
                y, _ = var(fa, y)
                [u, v], [size, _] = zip(*var(fa, [u, v]))
                parsed.extend(neq_one_hot(u, v, size, y))
            case [y, _, u, "<", v]:
                y, _ = var(fa, y)
                [u, v], [size, _] = zip(*var(fa, [u, v]))
                parsed.extend(lt_one_hot(u, v, size, y))
            case [y, _, u, ">", v]:
                y, _ = var(fa, y)
                [u, v], [size, _] = zip(*var(fa, [u, v]))
                parsed.extend(gt_one_hot(u, v, size, y))
            case [y, _, u, "-", v]:
                y, _ = var(fa, y)
                uv, [size, _] = zip(*var(fa, [u, v]))
                parsed.extend(sub_one_hot(uv[0], uv[1], size, y))
            case [y, _, u, "+", v]:
                y, _ = var(fa, y)
                uv, [size, _] = zip(*var(fa, [u, v]))
                parsed.extend(add_one_hot(uv[0], uv[1], size, y))
            case [y, _, u, "==", v]:
                y, _ = var(fa, y)
                uv, [size, _] = zip(*var(fa, [u, v]))
                parsed.extend(eq_one_hot(uv[0], uv[1], size, y))
    return parsed


def parse_unembed(lines: list[str], fa: FeatureAllocator):
    parsed = []
    while line := first(lines):
        [k, v] = line.split(": ")
        match k:
            case "Char":
                parsed.append(["CHAR", *resolve(fa, v)])
            case "Tokens":
                parsed.append(v)
            case "Binary":
                parsed.append(["BINARY", *resolve(fa, v)])
    return [parsed]


def resolve(fa: FeatureAllocator, f: str) -> list | tuple:
    match f:
        case "POS":
            return slice(fa.POS, len(P))
        case "EMB":
            return slice(fa.EMB, len(E))
        case _ if len(f) == 5 and f[0:2] == "['" and f[3:] == "']":
            c = f[2]
            if c.isdigit():
                return (fa.POS, len(P), one_hot(len(P), c))
            else:
                return (fa.EMB, len(E), one_hot(len(E), c))
        case _ if len(f) == 3 and f[0] == "'" and f[2] == "'":
            c = f[1]
            return [who(c)]
        case _ if info := fa.info(f):
            return [info[0] + i for i in range(info[1])]
        case _ if f[0] == "[" and f[-1] == "]":
            lists = [resolve(fa, g) for g in f[1:-1].split(",")]
            result = []
            for x in lists:
                if not isinstance(x, list):
                    raise TypeError()
                result.extend(x)
            return result
        case _:
            return fail(f"unknown feature '{f}'")


def print_columns(fa: FeatureAllocator, m: mat, label: str):
    print(label)
    d = 0
    while info := fa.info(d):
        key, dims = info
        print(key, end="\t")
        for col in m:
            if dims == len(V):
                ix = [i for i in range(dims) if col[d + i] == 1.0]
                if ix:
                    print(V[ix[0]], end="\t")
                else:
                    print("-", end="\t")
            if dims == R:
                ix = [i for i in range(dims) if col[d + i] == 1.0]
                if ix:
                    print(ix[0], end="\t")
                else:
                    print("-", end="\t")
            if dims == 1:
                print(int(col[d]), end="\t")
        d += dims
        print()
    print()


def fail(e):
    print(f"{e} at line {LINE_NUM}", file=sys.stderr, flush=True)
    sys.exit(1)


def first(lines: list) -> str:
    global LINE_NUM
    LINE_NUM += 1
    return lines.pop(0)


def var(fa: FeatureAllocator, key):
    if isinstance(key, list):
        vs = []
        for k in key:
            vs.append(var(fa, k))
        return vs
    elif len(key) == 3 and key[0] == "'" and key[2] == "'":
        return (V.index(key[1]), 1)
    else:
        x = fa.info(key)
        if not x:
            return fail(f"unknown variable '{key}'")
        return x
