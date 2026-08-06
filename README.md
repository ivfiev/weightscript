# WeightScript
A simple yaml-like language that gets translated to transformer weights.


Sample program below constructs 2-block transformer for checking whether a string is a palindrome. (there are more examples in ./test/ directory)


```yml
Features:
  special: bool 
  pos: number 
  id: char
  neq: bool
  result: bool
   
Block:
  Attention:
    - Head:
        Query: ['$']
        Key: EMB
        Value: POS
        Proj: pos
  FeedForward:
    - special = nor '$','^'
    - pos = pos - POS

Block:
  Attention:
    - Head:
        Query: pos 
        Key: POS
        Value: EMB
        Proj: id
  FeedForward:
    - neq = id != EMB 

Block:
  Attention:
    - Head:
        Query: '^'
        Key: special
        Value: neq
        Proj: result
  FeedForward:
    - result = not result

Unembed:
  Binary: result
  Tokens: 0:1
```
