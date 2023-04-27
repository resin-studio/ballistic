# ballistic

Bayesian regression synthesis

## TODO
- see if expressions over plates are propagated properly.
    - might need to convert output col to output matrix.

## notice
- the relation between inputs and outputs is connected via smt constraints and terms
    - the loss is defined by the noise formula, which is defined from output terms and sample data 
    - the solver slows down when constraints are added over the loss 
- lots of trial and error with constraints to improve SMT performance; of little avail
- need to distinguish between fresh vars and fresh weights
- need to maintain context mapping variables to weights
    - this is not part of Unsupervised work by Ellis et al.
- need to associated constraint with each choice, due to distributions
- need to use min and max of samples to form boundary of distributions
- solving hangs forever: 
    - tried removing various distribution constraints. any one causes major slow down.
    - tried making constraints only be required if control is on; still slow.
    - tried using terms instead of constraints; still slow
    - tried switching between solved weights and enumerated digits; both slow 
    - tried removing powers/exponents; still slow

## TODO
- reproduce time series unemployment claim example
    - why doesn't the line fit the data properly?
    - does the log transform help/hurt? 
- add vector concept to type universe


## type universe
- the type universe is the language of types
- either a type consists of of a base and a predicate, or a type is Top 
- the base is with respect to a category in the grammar
- the predicate could be a range or some way of narrowing the scope the type
- the bases include input numbers, distributions, and reals

## type abstraction
- a type abstraction is a set of types that may be used in verification
- the set of types is refined with each iteration 

## concrete tree automata
- the concrete automaton verifies that an expression would evaluate to an acceptable value   

## abstract semantics
- the meaning of expressions in terms of types
- applying an expression to an input is not included

## abstract evaluation 
- the meaning of expressions in terms of types evaluated on input
- a simple wrapper around abstract semantics

## abstract tree automata
- the abstract automaton verifies that an expression would evaluate to an acceptable type 
- constructed from the data, grammar, type abstraction, and abstract semantics 
- the type for an expression in the automaton is an over-approximation the the abstract semantics of the expression
    - more precisely, it is the strongest conjunction from the type abstraction 
    - that is the supertype of the abstract semantics of that expression   


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
- inspired by: Ellis et al. Unsupervised Learning by Program Synthesis
    - reference: https://people.csail.mit.edu/asolar/SynthesisCourse/Lecture20.htm
- define a grammar consisting of some basic arithmetic and plate concepts
- define a denotation into SMT language
- associated grammar with uniform distribution for each node
- repeat over production rule, threshold:
    - generate SMT formula representing space of programs from grammar/production rules and denotation semantics
        - with constraint that description length must be less than threshold
    - draw a program from grammar/description length distribution 
    - learn posteriors for latent variables using SVI in Pyro
    - let threshold' be discrepancy between mean output and observed output 
    - if threshold' >= threshold: terminate
    - else: update threshold to threshold' 

## synthesis approach questions
- we can use the actual input data in the SMT formula as supervision
- do we need to encode dependency between sample variables and distribution (and distribution arguments)?
    - probably yes. we can construct a range or equality constraint from the distribution type, its mean (and possibly its standard deviation)
    - can't encode the soft constraints of distributions
- how does the constraint between description length constraint and reconstruction error influence SMT search? 
    - the description length is tied to the control bits in the SMT formula
    - this is how we encode soft constraints
- how do we encode an AST into an SMT formula?
- how do we decode an AST back from the SMT formula?



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
