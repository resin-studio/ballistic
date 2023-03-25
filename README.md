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
- plate: collection of variables governed by the same bayesian description 
- program synthesis: finding distributions for m, b, that maximize P(y = mx + b) 
- neural network:
    - vertical list of data mapping to vertical list of data 
    - mapped via 2-dimensional matrix representing size(output) in vertical and size(input) horizontal 
- guide program: describes the ideal behavior of latent variable
    - may be parameterized by neural network
    

## TODO
- should we narrow domain to time series modeling?
- what are the possible architectures the language should use 
- what are the possible distributions the language should use 
    - LogNormal
    - Bernoulli
    - Normal
    - HalfNormal
    - Distribution (variational)
        - may be parameterized by neural network
- what are the possible learning algorithms the language should use 
    - mcmc
    - svi
        - close distance between posterior p(theta|y) and guide q(theta)
        - using ELBO: find a theta distribution that maximizes P(y, theta)/q(theta) <: log p(y)
        - uses a guide parameterized by neural network 
    - experimental design:
        -learn optimal design, given specification of distribution shape
        - that decrease posterior entropy
        - using information gain / EIG
            - measure difference in entropy between prior and posterior
            - relies on some initial model of how y behaves to sample from
- how will the language represent these concepts succinctly
    - look at examples from video and determine how to represent as bayesian graph + data
    - should we specify priors + architecture or do a search for priors along architecture 
    - how to represent specification of likelihood distribution with examples (bayesian graphs and data together)
    - experimental design problem?