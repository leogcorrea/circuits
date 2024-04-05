# Logic Circuits

A pytorch implementation of inference in probabilistic and logic circuits. The pipeline is: given a PASP source file, it's converted to a CNF format and then to a NNF format. The output of this last step is then used to build a circuit representation comprising layers of pythorch nn.Module objects in order to paralelize execution. 

# Usage

Run logicc.py with two input arguments: 
    1. The path to the PASP source program;
    2. The path to c2d compiler executable.
