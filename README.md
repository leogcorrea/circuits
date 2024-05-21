# Probabilistic and Logic Circuits

A pytorch implementation of inference in probabilistic and logic circuits. The main pipeline for using the library is, given an Answer Set Programming (ASP) description in a source file, it's converted into a Conjunctive Normal Form (CNF) and then into a Negation Normal Form (NNF) in order to build a circuit which corresponds to the original ASP program. The circuit representation is a sequence of pythorch nn.Module layers which paralelize inference on the circuit. Furthermore, it's possible to setup weights interpreted as probabilities associated to a given input. It'a also possible to calculate these input probabilities by another approach, for instance, using neural networks.

# Usage

Run logicc.py with two input arguments: 
    1. The path to the ASP source program;
    2. The path to c2d compiler executable.

