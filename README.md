# WeightScript
Create transformer model weights by hand from a yaml-like scripting language and observe residual states gradually evolve during the forward pass.

## Examples
Sample program below constructs 2-block transformer for reversing strings. (there are more examples in ./test/ directory)

```yml
Features:                               # feature vectors are allocated explicitly. each token begins with standard EMB + POS vector
  position: number                      # we'll need a feature to hold the position of the opposite token
  mirrored: char                        # we'll also need to hold opposite token's embedding
   
Block:                                  # each block consists of 0 or more attention and feed-forward layers
                                        # in this block we compute the position of the mirrored/opposite token
  Attention:
    - Head:
        Query: ['$']                    # looking for end-of-input token ($)
        Key: EMB                        # it gets compared with each tokens actual embedding
        Value: POS                      # if match - position gets copied
        Proj: position                  # Proj is equivalent to W_output. mirrored token's position is written to 'position' feature
  FeedForward:
    - position = position - POS         # then FFN subtracts token's current position from $'s position

Block:                                  # in this block we're copying the mirrored token's embedding
  Attention:
    - Head:
        Query: position                 # looking for 'position' number
        Key: POS                        # in token's actual position (POS)
        Value: EMB                      # if match - copy it's embedding (EMB)
        Proj: mirrored                  # into 'mirrored'
  FeedForward:
    -                                   # we don't need any FFN computation in this block

Unembed:                                # unembed specifies how to output the results
  Char: mirrored                        # here we're requesting the character stored in 'mirrored' feature space
  Tokens: 1:-1                          # over all tokens, excluding special input start/end (^ and $) tokens
```

Sample output from running an example:
`python main.py --run ./test/reverse.yml`
```
terminal output... (maybe put a smaller example higher as well)
```

Dump parameters:
`python main.py --weights ./test/reverse.yml`


## What you can do:
- Program transformer layers using a simple YAML-like language
- Compile programs directly into attention and FFN parameters
- Inspect every intermediate state during forward pass
- Watch attention move information between tokens
- Watch FFNs perform symbolic computation within tokens
- Inspect the generated weights

## Deliberate simplifications:
Some parts of the modern transformer are intentionally simplified to aid human understanding.
- There is no training
- Features are represented as explicit orthogonal one-hot basis vectors
- Discretized attention - non-matching keys always contribute a 0 value
- Simple custom "binary-norm" instead of LayerNorm/RMSNorm

