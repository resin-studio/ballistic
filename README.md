# ballistic

## TODO 
- define structural constraint interface 
    - the constraint language is akin to the predicate universe in the AFTA (abstract finite tree automata) refinement work.
    - the abstract finite tree automata (AFTA) is constructed from the predicate universe AND grammar, to efficiently represent the search space. 
        - the automaton represents both the predicates/types AND the static/abstract semantics combined
        - the concrete automaton represents the criterion that an expression evaluates to some value   
        - the abstract automaton represents the criterion that an expression evaluates to some type 
    - however the constraints represent abstract bayesian network. ("abstract bayesian network automata"?)
    - define Sum/Product etc constraints
- synthesize architecture from constraints 
    - akin to AFTA refinement algorithm 
- generalize from scalars to tensors

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

## language design
- a distribution is an expression representing density values
- primitive distribution constructors include: 
    - `normal`
    - `bernoulli`
- a procedure is a function from scalars to a distribution 
    - ```
    fun myDist(x) =  
        for m in Normal(0,1)
        for b in Norma(0,1)
        let y = m * x + b
        y
    ```
    - ```
    fun myDist(x) =  
        bind Normal(0,1) as m =>
        bind Norma(0,1) as b =>
        wrap m * x + b
    ```
- a script is a sequence of commands 
    ```
    for x in Normal(0,1) 
    for y in Normal(0,1) 
    let z = x + y 
    z
    ```
- commands include
    - `sample`
    - `plot`
- "evaluation" is really "sampling" a distribution
- distribution function could look like `p(x : X) : [0, 1] = ...`
- probabilities could look like `Pr[x + y = z | ... ]`
- bayesian network corresponds to type/DSL
- operations in DSL correspond to constraints in bayesian types  
- data can be represented as special type
    - `x : X -> {y : Y | (x, y) : data(file)}` 
- bayesian network can be represented as constraint type and an expression: 
    - `x : a, z : c, y : {b | a, b, c : Sum} ~> y = z - x`
- samples are represented as variables in types 
- distributions are not in types directly
- bayesian network constraint can be compiled into a DSL refinement
- the latent variables (M, B) are indicated by existential quantifiers
    - All X . X -> (Some Y M B. Y :: Y = M * X + B)   
- restrict to basic arithmetic and real comparison operators

## synthesis goal 
- synthesize distributional function from type/data
## synthesis approach 
- have a default DSL consisting of some basic arithmetic and plate concepts
- represent bayesian network as constraint type
- compile constraint type into DSL refinement
- compile DSL into tree automaton
- repeat:
    - search for tree that is accepted by tree automata (Syngar) 
    - construct model from tree and dataset with priors 
    - use stochastic variational inference to learn posteriors (Pyro)

## Example
- learn from time series unemployment data without specifying any structure
    - automatically learn to use the plate concepts
- learn from sparse data with bayesian constraint 
    

## Open questions 
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
