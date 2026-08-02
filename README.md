# T-lang
WIP

Sample program below constructs 2-block transformer for reversing strings.

Features are 1-hot vectors superimposed in 72-dimensional space.


```yml
Features:
  pos: number 
  bro: char
   
Block:
  Attention:
    - Head:
        Query: ['$']
        Key: EMB
        Value: POS
        Proj: pos
  FeedForward:
    - pos -= POS

Block:
  Attention:
    - Head:
        Query: pos
        Key: POS
        Value: EMB
        Proj: bro
  FeedForward:
    - 

Unembed:
  Char: bro 
  Tokens: 1:-1

```
