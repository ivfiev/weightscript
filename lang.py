from model import *


def build_attn(codes: list[list]) -> Attention:
    q, k, v, p = [], [], [], []
    for code in codes:
        c = code[0]
        features = code[1]
        m = {
            "QUERY": q,
            "KEY": k,
            "VALUE": v,
            "PROJ": p,
        }[c]
        if isinstance(features, list):
            for f in features:
                m.append([0] * D)
                m[-1][f] = 1.0
        if isinstance(features, tuple):
            for _ in range(features[1]):
                m.append([0] * D)
            for i in range(features[0], features[1]):
                for j in range(features[1]):
                    m[j][i] = features[2][j]
    return Attention(t(q), t(k), t(v), p)


def build_ffn(codes: list[list]) -> FFN:
    logic = ["AND", "OR", "NOT", "NOR", "NAND", "TWO"]
    other = ["ZERO"]
    n = max(1, sum(code[0] in [*logic, *other] for code in codes))
    input = [[0.0] * D for _ in range(n)]
    output = [[0.0] * n for _ in range(D)]
    bias_in = [0.0] * n
    bias_out = [0.0] * D
    for ip, code in enumerate(codes):
        c = code[0]
        if c in ["AND", "OR", "NOT"]:
            features = code[1]
            write = code[2]
            for f in features:
                input[ip][f] = 1.0
            bias_in[ip] = -len({*features}) + 1.0 if c == "AND" else 0.0
            for w in write:
                output[w][ip] = -3.0 if c == "NOT" else 1.0
                bias_out[w] = 1.0 if c == "NOT" else 0.0
        if c == "NOR":
            features = code[1]
            write = code[2]
            for f in features:
                input[ip][f] = -1.0
            bias_in[ip] = 1.0
            for w in write:
                output[w][ip] = 1.0
                bias_out[w] = 0.0
        if c == "NAND":
            features = code[1]
            write = code[2]
            for f in features:
                input[ip][f] = -1.0
            bias_in[ip] = len({*features})
            for w in write:
                output[w][ip] = 1.0
                bias_out[w] = 0.0
        if c == "TWO":
            features = code[1]
            write = code[2]
            for f in features:
                input[ip][f] = 1.0
            bias_in[ip] = -1.0
            for w in write:
                output[w][ip] = 1.0
                bias_out[w] = 0.0
        if c == "ZERO":
            bit = code[1]
            input[ip][bit] = 1.0
            output[bit][ip] = -1.0
    return FFN(t(input), bias_in, t(output), bias_out)


def build_unembed(code: list) -> tuple[mat, vec, str]:
    r = code[-1]
    m = [[0.0 for _ in range(len(V))] for _ in range(D)]
    b = [0.0 for _ in range(len(V))]
    if code[0][0] == "BINARY":
        f = code[0][1]
        zero = V.index("0")
        one = V.index("1")
        m[f][one] = 2.0
        b[zero] = 1.0
    if code[0][0] == "NUM":
        f = code[0][1]
        z = V.index("0")
        for i in range(10):
            m[i + f][i + z] = 1.0 + i
    if code[0][0] == "CHAR":
        f = code[0][1]
        for i in range(len(V)):
            m[i + f][i] = 1.0
    return (m, b, r)


def run(program: list[list], input: str) -> str:
    blocks = []
    unembed = None
    for code in program:
        if len(code) == 2:
            blocks.append(Block([build_attn(code) for code in code[0]], build_ffn(code[1])))
        elif len(code) == 1:
            unembed = build_unembed(code[0])
    return Transformer(blocks, E, P, unembed)(input)  # pyright: ignore[reportArgumentType]


def who(c):
    return next(i for i, e in enumerate(E[c]) if e == 1.0)


def where(n):
    return next(i for i, e in enumerate(P[n]) if e == 1.0)


def slice(base, count):
    return [x for x in range(base, base + count)]


def one_hot(n, c):
    i = V.index(c) if isinstance(c, str) else c
    return [1.0 if j == i else 0.0 for j in range(n)]


# a = a - b
def sub_one_hot(a, b, c):
    return [
        *[["ZERO", i] for i in range(a, a + c)],
        *[["AND", [i, j], [a + (i - a) - (j - b)]] for j in range(b, b + c) for i in range(a, a + c) if (i - a) >= (j - b)],
    ]


# a = a + b
def add_one_hot(a, b, c):
    return [
        *[["ZERO", i] for i in range(a, a + c)],
        *[["AND", [i, j], [a + (i - a) + (j - b)]] for j in range(b, b + c) for i in range(a, a + c) if (i - a) + (j - b) < c],
    ]


# a = a + 1
def inc_one_hot(a, b):
    return [
        *[["ZERO", i] for i in range(a, a + b)],
        *[["AND", [i, i], [i + 1]] for i in range(a, a + b)],
    ]


# d = a[0:c] == b[0:c], assume only one 1
def eq_one_hot(a, b, c, d):
    return [["AND", [a + i, b + i], [d]] for i in range(c)]


# d = a[0:c] != b[0:c], assume only one 1
def neq_one_hot(a, b, c, d):
    x = [["TWO", [a + i, *[b + j for j in range(c) if j != i]], [d]] for i in range(c)]
    return x


# d = a[0:c] < b[0:c], assume only one 1
def lt_one_hot(a, b, c, d):
    return [["AND", [a + j, b + i], [d]] for i in range(c) for j in range(i)]


# d = a[0:c] > b[0:c], assume only one 1
def gt_one_hot(a, b, c, d):
    return [["AND", [a + i, b + j], [d]] for i in range(c) for j in range(i)]


class FeatureAllocator:
    def __init__(self):
        self.EMB = 0
        self.POS = len(E)
        self.n = len(E) + len(P)

    def alloc(self, range=1):
        assert self.n + range < D
        n = self.n
        self.n += range
        return n
