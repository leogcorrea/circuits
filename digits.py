from logicc import build_circuit_from_file, pasp2cnf, cnf2nnf
import torch
import sys
import os

if __name__ == '__main__':
    
    filename = sys.argv[1] if len(sys.argv) > 1 else os.getcwd() + '/digits.pasp'
    c2d_executable = sys.argv[2] if len(sys.argv) > 2 else os.getcwd() + '/c2d_linux'

    filename, sym2lit = pasp2cnf(filename)
    filename = cnf2nnf(filename, c2d_executable)

    import time
    start_time = time.time()

    circuit = build_circuit_from_file(filename)
    print("Circuit: ", filename)
    print("Time to build the circuit: {} s".format(time.time() - start_time))

    lit2idx = lambda lit: lit-1

    probs = torch.ones(circuit.nliterals)

    for lit in range(1, 20+1):
        probs[lit2idx(lit)] = 0.1 
    
    circuit.set_input_weights(probs)

    print("Weights (probabilities): ", probs)

    for lit in range(21, 39+1):
        start_time = time.time()
        output = circuit.query([lit])
        clause = ""
        for k,v in sym2lit.items():
            if v == lit:
                clause = k 
        print("Query: ", clause)
        print("Result: ", output)

    print("Time to compute queries: {} s".format(time.time() - start_time))