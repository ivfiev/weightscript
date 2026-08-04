type vec = list[float]
type mat = list[list[float]]

D = 96
V = "^$abcde01234|"
R = 16
E = {e: [1.0 if j == i else 0.0 for j in range(D)] for i, e in enumerate(V)}
P = {p: [1.0 if j == p + len(E) else 0.0 for j in range(D)] for p in range(R)}


def mm(a: mat, b: mat) -> mat:
    assert all(len(r) == len(b) for r in a)
    return [[sum(a[r][i] * b[i][c] for i in range(len(a[0]))) for c in range(len(b[0]))] for r in range(len(a))]


def ma(a: mat, b: mat) -> mat:
    assert len(a) == len(b)
    return [[a[r][c] + b[r][c] for c in range(len(a[0]))] for r in range(len(a))]


def va(a: mat, u: vec) -> mat:
    assert len(a[0]) == len(u)
    return [[u[i] + a[r][i] for i in range(len(a[r]))] for r in range(len(a))]


def t(a: mat) -> mat:
    b = []
    for i in range(len(a[0])):
        b.append([])
        for j in range(len(a)):
            b[-1].append(a[j][i])
    return b


def binary_norm(a: mat) -> mat:
    return [[0.0 if x < 0.9 else 1.0 for x in r] for r in a]


def relu(a: mat) -> mat:
    return [[0.0 if x < 0.0 else x for x in r] for r in a]


def pm(a: mat):
    for u in a:
        print(u)
    print()


class Attention:
    def __init__(self, queries: mat, keys: mat, values: mat, proj: mat):
        self.queries = queries  # d x a
        self.keys = keys  # d x a
        self.values = values  # d x v
        self.proj = proj  # v x d

    def __call__(self, x) -> mat:
        q = mm(x, self.queries)  # n x a
        k = mm(x, self.keys)  # n x a
        v = mm(x, self.values)  # n x v
        qs = list(map(sum, q))
        qk = mm(q, t(k))  # n x n
        qk = self._hardmax(qk, qs)
        x = mm(qk, v)  # n x v
        x = mm(x, self.proj)
        return x

    def _hardmax(self, qk: mat, qs: vec) -> mat:
        return [[1.0 if 0 < s <= x else 0.0 for x in r] for r, s in zip(qk, qs)]


class FFN:
    def __init__(self, input: mat, bias_in: vec, output: mat, bias_out: vec):
        self.input = input
        self.bias_in = bias_in
        self.output = output
        self.bias_out = bias_out

    def __call__(self, x: mat) -> mat:
        x = mm(x, self.input)
        x = va(x, self.bias_in)
        x = relu(x)
        x = mm(x, self.output)
        x = va(x, self.bias_out)
        return x


class Block:
    def __init__(
        self,
        attn: list[Attention],
        ffn: FFN,
        output=None,
    ):
        self.attn = attn
        self.ffn = ffn
        self.output = output or (lambda _, __: None)

    def __call__(self, x: mat) -> mat:
        attns = [attn(x) for attn in self.attn]
        for a in attns:
            x = ma(x, a)
        x = binary_norm(x)
        self.output(x, "Attention:")
        x = ma(x, self.ffn(x))
        x = binary_norm(x)
        self.output(x, "FeedForward:")
        return x


class Transformer:
    def __init__(
        self,
        blocks: list[Block],
        emb: dict,
        pos: dict,
        unembed: tuple,
        output=None,
    ):
        self.blocks = blocks
        self.emb = emb
        self.pos = pos
        self.unembed = unembed
        self.output = output or (lambda _, __: None)

    def __call__(self, x: str):
        m = self._embed(x)
        self.output(m, "Initial:")
        for b in self.blocks:
            m = b(m)
        return self._unembed(m)

    def _embed(self, x: str):
        e = [self.emb[c] for c in x]
        p = [self.pos[i] for i in range(len(x))]
        m = ma(e, p)
        return m

    def _unembed(self, m: mat):
        u, b, r = self.unembed
        ij = [int(x) for x in r.split(":") if x]
        # TODO - pass the range at runtime instead
        match ij:
            case []:
                m = m[:]
            case [i]:
                m = m[i:]
            case [i, j]:
                m = m[i:j]
        m = mm(m, u)
        m = va(m, b)
        preds = [V[v.index(max(v))] for v in m]
        return "".join(preds)
