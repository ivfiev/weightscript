# WeightScript
Tiny YAML-like language for building transformer programs and watching them execute step by step.

## Examples
A program that reverses a string: (there are more examples in ./test/ directory)

```yml
Features:                               # feature vectors are allocated explicitly. each token begins with standard EMB + POS vector
  position: number                      # each token will need a feature to hold the position of the opposite token
  mirrored: char                        # we'll also need to hold opposite token's embedding/character
   
Block:                                  # each block consists of 0 or more attention and feed-forward layers
                                        # in this block we compute the position of the mirrored/opposite token
  Attention:
    - Head:
        Query: ['$']                    # looking for end-of-input token ($)
        Key: EMB                        # compare against each token
        Value: POS                      # take its position
        Proj: position                  # store it in 'position' feature
  FeedForward:
    - position = position - POS         # then FFN obtains mirrored position using subtraction

Block:                                  # in this block we're copying the mirrored token's embedding
  Attention:
    - Head:
        Query: position                 # we already know what position we want
        Key: POS                        # find the token at that position
        Value: EMB                      # copy its character
        Proj: mirrored                  # into 'mirrored'
  FeedForward:
    -                                   # we don't need any FFN computation in this block

Unembed:                                # unembed specifies how to output the results
  Char: mirrored                        # here we're requesting the character stored in 'mirrored' feature space
  Tokens: 1:-1                          # over all tokens, excluding special input start/end (^ and $) tokens (python slice syntax)
```

### Sample output:
`python main.py --run ./test/reverse.yml --input abcd`

```
Initial:
EMB         ^           a           b           c           d           $           
POS         0           1           2           3           4           5           
position    -           -           -           -           -           -           
mirrored    -           -           -           -           -           -           

Attention:
EMB         ^           a           b           c           d           $           
POS         0           1           2           3           4           5           
position    5           5           5           5           5           5           
mirrored    -           -           -           -           -           -           

FeedForward:
EMB         ^           a           b           c           d           $           
POS         0           1           2           3           4           5           
position    5           4           3           2           1           0           
mirrored    -           -           -           -           -           -           

Attention:
EMB         ^           a           b           c           d           $           
POS         0           1           2           3           4           5           
position    5           4           3           2           1           0           
mirrored    $           d           c           b           a           ^           

FeedForward:
EMB         ^           a           b           c           d           $           
POS         0           1           2           3           4           5           
position    5           4           3           2           1           0           
mirrored    $           d           c           b           a           ^           

Output: dcba
```

### Dumping parameters:
`python main.py --weights ./test/reverse.yml`


## What you can do:
- Deterministically construct the model parameters
- Inspect every intermediate state during forward pass
- Watch attention move information between tokens
- Watch FFNs perform symbolic computation within tokens

## Deliberate simplifications:
Some parts of the modern transformer are intentionally simplified to aid human understanding.
- There is no training
- Features are represented as explicit orthogonal one-hot basis vectors
- Discretized attention - non-matching keys always contribute a 0 value
- Simple custom "binary-norm" instead of LayerNorm/RMSNorm

## Related projects
- https://github.com/google-deepmind/tracr
- https://arxiv.org/pdf/2106.06981
