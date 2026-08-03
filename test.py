from lang import *
from parser import parse_transformer


def run(code, input):
    return build_transformer(code, None)(input)


def is_palindrome(s: str) -> str:
    a = FeatureAllocator()
    S = a.alloc("S")
    POS = a.alloc("POS", len(P))
    ID = a.alloc("ID", len(E))
    CMP = a.alloc("CMP")
    RESULT = a.alloc("RESULT")
    code = [
        [
            [
                [
                    ["QUERY", (a.EMB, len(E), one_hot(len(E), "$"))],
                    ["KEY", slice(a.EMB, len(E))],
                    ["VALUE", slice(a.POS, len(P))],  # if match, then copy pos
                    ["PROJ", slice(POS, len(P))],  # project into POS subspace
                ],
            ],
            [
                ["NOR", [who("^"), who("$")], [S]],  # mark non-special tokens
                *sub_one_hot(POS, a.POS, len(P)),  # subtract actual pos from mirrored
            ],
        ],
        [
            [
                [
                    ["QUERY", slice(POS, len(P))],  # looking for mirrored pos
                    ["KEY", slice(a.POS, len(P))],  # ... in each tokens actual pos
                    ["VALUE", slice(a.EMB, len(E))],  # copy matching token's identity
                    ["PROJ", slice(ID, len(E))],  # into ID
                ]
            ],
            [
                *neq_one_hot(a.EMB, ID, len(E), CMP),  # compare one-hot into CMP
            ],
        ],
        [
            [
                [
                    ["QUERY", [who("^")]],  # Q = 1 for ^, 0 else
                    ["KEY", [S]],  # K = non-special tokens, ie broadcast
                    ["VALUE", [CMP]],  # aggregate 1 if non-match, else 0
                    ["PROJ", [RESULT]],  #  OR them into 42nd, 0 of palindrome, 1 if not
                ],
            ],
            [
                ["NOT", [RESULT], [RESULT]],
            ],
        ],
        [
            [["BINARY", RESULT], "0:1"],
        ],
    ]
    return run(code, f"^{s}$")


# def count_letter(s: str, c: str) -> str:
#     a = FeatureAllocator()
#     POS = a.alloc(len(P))
#     code = [
#         [
#             [
#                 [
#                     ["QUERY", [who("^")]],
#                     ["KEY", [who(c)]],
#                     ["VALUE", slice(a.POS, len(P))],
#                     ["PROJ", slice(POS, len(P))],
#                 ],
#             ],
#             [],
#         ],
#         [
#             [["NUMERIC", POS], "0:1"],
#         ],
#     ]
#     return run(code, f"^{s}$")


def reverse(s: str) -> str:
    a = FeatureAllocator()
    POS = a.alloc("POS", len(P))
    BRO = a.alloc("BRO", len(E))
    code = [
        [
            [
                [
                    # copy pos of $ into each token
                    ["QUERY", (a.EMB, len(E), one_hot(len(E), "$"))],
                    ["KEY", slice(a.EMB, len(E))],
                    ["VALUE", slice(a.POS, len(P))],
                    ["PROJ", slice(POS, len(P))],
                ],
            ],
            [
                *sub_one_hot(POS, a.POS, len(P)),  # obtain mirrored pos
            ],
        ],
        [
            [
                [
                    # copy the twin token embedding
                    ["QUERY", slice(POS, len(P))],
                    ["KEY", slice(a.POS, len(P))],
                    ["VALUE", slice(a.EMB, len(E))],
                    ["PROJ", slice(BRO, len(E))],
                ],
            ],
            [],
        ],
        [
            [["CHAR", BRO], "1:-1"],
        ],
    ]
    return run(code, f"^{s}$")


def kv(s: str, r: str) -> str:
    a = FeatureAllocator()
    COMMA = a.alloc("COMMA", len(P))
    EOF = a.alloc("EOF", len(P))
    GT = a.alloc("GT")
    LT = a.alloc("LT")
    BETWEEN = a.alloc("BETWEEN")
    OUTSIDE = a.alloc("OUTSIDE")
    POS = a.alloc("POS", len(P))
    VALUE = a.alloc("VALUE", len(E))
    code = [
        [
            [
                [
                    # copy pos of , into each token
                    ["QUERY", (a.EMB, len(E), one_hot(len(E), "|"))],
                    ["KEY", slice(a.EMB, len(E))],
                    ["VALUE", slice(a.POS, len(P))],
                    ["PROJ", slice(COMMA, len(P))],
                ],
                [
                    # copy pos of $ into each token
                    ["QUERY", (a.EMB, len(E), one_hot(len(E), "$"))],
                    ["KEY", slice(a.EMB, len(E))],
                    ["VALUE", slice(a.POS, len(P))],
                    ["PROJ", slice(EOF, len(P))],
                ],
            ],
            [
                *gt_one_hot(a.POS, COMMA, len(P), GT),  # flag if to the right of ,
                *lt_one_hot(a.POS, EOF, len(P), LT),  # flag if to the left of $
            ],
        ],
        [
            [],
            [
                ["AND", [GT, LT], [BETWEEN]],
                ["NAND", [GT, LT], [OUTSIDE]],
            ],
        ],
        [
            [
                [
                    # copy twin token's position
                    ["QUERY", [BETWEEN, *slice(a.EMB, len(E))]],
                    ["KEY", [OUTSIDE, *slice(a.EMB, len(E))]],
                    ["VALUE", slice(a.POS, len(P))],
                    ["PROJ", slice(POS, len(P))],
                ],
            ],
            [
                *inc_one_hot(POS, len(P)),
            ],
        ],
        [
            [
                [
                    # copy target token's position
                    ["QUERY", slice(POS, len(P))],
                    ["KEY", slice(a.POS, len(P))],
                    ["VALUE", slice(a.EMB, len(E))],
                    ["PROJ", slice(VALUE, len(E))],
                ],
            ],
            [],
        ],
        [
            [["CHAR", VALUE], r],
        ],
    ]
    return run(code, f"^{s}$")


def run_tests():
    print(
        "palindrome",
        all(
            [
                is_palindrome("a") == "1",
                is_palindrome("ababa") == "1",
                is_palindrome("cababa2") == "0",
                is_palindrome("01b") == "0",
            ]
        ),
    )
    # print("strawberry", count_letter("strawberry", "r") == 3.0)
    print(
        "reverse   ",
        all(
            [
                reverse("abc") == "cba",
                reverse("0ac2c1b1") == "1b1c2ca0",
            ]
        ),
    )
    print(
        "kv        ",
        all(
            [
                kv("a0b1c2|cba", "-4:-1") == "210",
                kv("c2a2b0|bbc", "-4:-1") == "002",
                kv("c2a2b0|ab", "-3:-1") == "20",
            ]
        ),
    )


if __name__ == "__main__":
    run_tests()
