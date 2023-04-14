# ballistic

# type universe
- the type universe is the language of types
- either a type consists of of a base and a predicate, or a type is Top 
- the base is with respect to a category in the grammar
- the predicate could be a range or some way of narrowing the scope the type
- the bases include input numbers, distributions, and reals

# type abstraction
- a type abstraction is a set of types that may be used in verification
- the set of types is refined with each iteration 

# concrete tree automata
- the concrete automaton verifies that an expression would evaluate to an acceptable value   

# abstract semantics
- the meaning of expressions in terms of types
- applying an expression to an input is not included

# abstract evaluation 
- the meaning of expressions in terms of types evaluated on input
- a simple wrapper around abstract semantics

# abstract tree automata
- the abstract automaton verifies that an expression would evaluate to an acceptable type 
- constructed from the data, grammar, type abstraction, and abstract semantics 
- the type for an expression in the automaton is an over-approximation the the abstract semantics of the expression
    - more precisely, it is the strongest conjunction from the type abstraction 
    - that is the supertype of the abstract semantics of that expression   

# algorithm 
0. hyperparams: sampling number, fuel for iterations.
1. input: data, initial type abstraction: Top, InputBase, TerminalBase 
2. construct automaton from grammar, data, and type abstraction 
3. if automaton is empty then return none.  
4. get prog in automaton ranked by ast-cost 
5. learn weights via pyro (i.e. stochastic variational inference) 
6. compute "fitness" of program w.r.t data by sampling.
    - maximize fitness: P(f)*Prod_i[P(out_i | f, in_i)]
    - minimize inverse transform: -log P(g) + Sum_i[-log(P(out_i | f, in_i))]
    - correctness model: probability increases exponentially as standard deviation increases?
    - simplicity model: probability decreases exponentially as length increases
    - reference: https://people.csail.mit.edu/asolar/SynthesisCourse/Lecture20.htm
    - TODO: 
        - lookup statistical/machine learning methods for measuring fit
        - can we use standard deviation to measure this? 
7. terminate if no better solution can be found.
8. find rows in data with really bad fit / poor correctness. 
    - e.g. examples at the standard deviation boundary.
9. annotate program with types weakened by examples 
10. update universe with conjunction of new weakened types from the annotated program and goto 2 

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
- operations in DSL correspond to types in bayesian types  
- data can be represented as special type
    - `x : X -> {y : Y | (x, y) : data(file)}` 
- bayesian network can be represented as type type and an expression: 
    - `x : a, z : c, y : {b | a, b, c : Sum} ~> y = z - x`
- samples are represented as variables in types 
- distributions are not in types directly
- bayesian network type can be compiled into a DSL refinement
- the latent variables (M, B) are indicated by existential quantifiers
    - All X . X -> (Some Y M B. Y :: Y = M * X + B)   
- restrict to basic arithmetic and real comparison operators

## synthesis goal 
- synthesize distributional function from type/data
## synthesis approach 
- have a default DSL consisting of some basic arithmetic and plate concepts
- represent bayesian network as type type
- compile type type into DSL refinement
- compile DSL into tree automaton
- repeat:
    - search for tree that is accepted by tree automata (Syngar) 
    - construct model from tree and dataset with priors 
    - use stochastic variational inference to learn posteriors (Pyro)

## Example
- learn from time series unemployment data without specifying any structure
    - automatically learn to use the plate concepts
- learn from sparse data with bayesian type 
    

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
