# ballistic

## glossary 
- given an architecture y = mx + b 
- observable variables: y, x
- latent variables: cannot be observed directly - m, b
- prior distribution: predetermined beliefs about weights  
    - p(m) 
    - p(b)
- likelihood: inferred beliefs about observable variables 
    - p(y | m, x, b)  
    - p(x | y, m, b)  
- posterior distribution: synthesized beliefs about latent variables
    - p(m | x, y, b)
    - p(b | x, m, y)
- program synthesis: finding distributions for m, b, that maximize P(y = mx + b) 

## TODO
[ ] what are the possible architectures the language should use 
[ ] what are the possible distributions the language should use 
[ ] what are the possible learning algorithms the language should use 
[ ] how will the language represent these concepts succinctly