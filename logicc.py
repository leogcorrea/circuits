import torch
from torch import nn
import nnf
import cnf
from pasp2cnf.program import Program
from collections import deque
import sys
from subprocess import Popen, PIPE

""" Base class for the probabilistic circuit. Represents an abstract layer.  """
class Layer(nn.Module):
    def __init__(self, id, matrix):
        super().__init__()
        self.id = id
        self.weight = nn.Parameter(matrix, requires_grad=False)

""" The input layer which operates a linear transform on the inputs. It receives: 
    a) an input configuration (comprising literal values); or 
    b) the input node values themselves. """
class InputLayer(Layer):
    def __init__(self, id, matrix, negated, input_mask, gains):
        super().__init__(id, matrix)
        self.linearTransform = nn.Linear(in_features=matrix.size(dim=1), out_features=matrix.size(dim=0), bias=False)
        #self.linearTransform.weight = nn.Parameter(self.weight, requires_grad=False)
        self.negated = negated
        self.input_mask = input_mask
        self.gains = None
        self.gain_set = None
        self.set_gains(gains)
        self.gain_set = False

    """ Gains (weights) applied by the linear transform on the inputs """
    def set_gains(self, gains):
        xgains = torch.matmul(self.input_mask, gains)
        sel = self.negated - xgains
        self.gains = abs(sel) + (sel == 0).type(torch.float) 
        self.linearTransform.weight = nn.Parameter(torch.matmul(torch.diag(self.gains), self.weight), requires_grad=False)
        self.gain_set = True

    """ Mask to obtain the negation of input literal according to the circuit setup """
    def set_negated(self, value):
        self.negated = value

    """ The linear transform application """
    def forward(self, input):
        if len(input) > 1:
            """ Transforms literal values into node states according to the circuit setup """
            configuration = input[0]
            mask = input[1].t()
            negated = input[2]
            x = torch.matmul(configuration, mask)
            y = abs(negated - x)
        else:
            y = input[0]
            """ Or operates on the node states informed directly """
        if self.gain_set:
            y = self.linearTransform(y)
        
        #y = torch.log(x)
        return y
    

""" A circuit layer which calculate an AND operation (product) of the input """
class AndLayer(Layer):
    def __init__(self, id, matrix):
        super().__init__(id, matrix)
        #self.linearTransform = nn.Linear(in_features=matrix.size(dim=1), out_features=matrix.size(dim=0), bias=False)
        #self.linearTransform.weight = nn.Parameter(matrix, requires_grad=False)

    def forward(self, input):
        #return self.linearTransform(input)
        x = torch.mul(self.weight, input)
        mask = (self.weight == 0)
        x += mask
        y = torch.prod(x, dim = 1)
        return y

""" A OR layer which sum up the input """
class OrLayer(Layer):
    def __init__(self, id, matrix):
        super().__init__(id, matrix)
        self.linearTransform = nn.Linear(in_features=matrix.size(dim=1), out_features=matrix.size(dim=0), bias=False)
        self.linearTransform.weight = self.weight

    def forward(self, input):
        # x = torch.exp(input)
        # x = self.linearTransform(x)
        # y = torch.log(x)
        
        # # xx = torch.mul(self.weight, input)
        # # mask = torch.tensor(self.weight != 0)
        # # xx = xx + torch.tensor(torch.finfo().tiny)
        # # yy = torch.logsumexp(xx, dim=1)

        return self.linearTransform(input)


""" Currently not being used. Could be use in the future for reading out the results of the circuit just calculated """
class OutputLayer(OrLayer):
    def __init__(self, id, matrix):
        super().__init__(id, matrix)

    def forward(self, input) -> torch.Tensor:
        y = super().forward(input)
        #y = torch.exp(x)
        return y
    

""" Get the max level (depth in terms of layers) found in the circuit """
def get_max_level(node, level = 0):
    max_level = level
    for child in node.children:
        l = get_max_level(child, level+1)
        if l > max_level:
            max_level = l
    return max_level
    
""" Determines the depth level (an increasing integer, zero based) in which each NNF node appears in the circuit.
    Input: 
        node: the NNF node to start from
        levels: a possibly empty dict of levels associating each node ID to a integer value (the level)
    Output: a populated dict of levels 
"""
def populate_levels(node, levels, level = 0):
    if node.id in levels:
        current = levels[node.id]
        if current < level:
            levels[node.id] = level
    else:
        levels[node.id] = level

    for child in node.children:
        populate_levels(child, levels, level+1)

""" Construct a dict of levels and related sets of nodes separating them in layers and organized as intercaling OR and AND layers. 
    May possibly create the so called 'virtual nodes' to aggregate multiple nodes in one layer belonging to the same
    layer operation (OR or AND).

    Input: 
        root: NNF node of the tree representing the circuit
"""
def build_layers(root):
    levels = {}
    populate_levels(root, levels)

    max_level = get_max_level(root)
    layers = {i: set() for i in range(max_level+1)}     

    level = 0
    root_t = type(root)
    other_t = nnf.OrNode if root_t is nnf.AndNode else nnf.AndNode # Used for intercalating the node/layer types
    
    queue = deque()
    queue.append((root, level))

    new_nodes = {}
    leaves = set()

    get_layer_type = lambda level: root_t if level % 2 == 0 else other_t

    # start a BFS in the tree
    while len(queue):
        node, level = queue.popleft()

        layer_t = get_layer_type(level)
        node_t = type(node)
        nlevel = levels[node.id] # Get the max level the node appears in
        ldiff = nlevel-level

        if (node_t == nnf.LiteralNode): # Literal nodes are leaves of the tree representing the circuit
            if (node not in leaves):
                leaves.add(node) 
                levels[node.id] = max_level
                for l in range(level, max_level+1):
                    layers[l].add(node)
        else:
            if (node_t != layer_t): # If the node type is different than the layer type, creates a 'virtual node' representation
                nlevel = level+1
                levels[node.id] = nlevel

            if level > nlevel:
                levels[node.id] = level # Update the max level this node appears just in case it has been moved down due to a new layer added
            elif level < nlevel:  # this case requires that if the node is not in the current layer, creates multiple virtual nodes to populate
                                  # the 
                if node in layers[level]:
                    continue
                if (node_t != get_layer_type(nlevel)):
                    nlevel += 1  # A new layer should be created to include the current node
                    levels[node.id] = nlevel  # Update it
                ldiff = nlevel-level   #
                # To create IDs for the virtual nodes, we use the same integer part and add some increasing unique decimal part 
                max_virtual_nodes = 10.
                while ldiff >= max_virtual_nodes:
                    max_virtual_nodes *= 10.0
                tflag = ldiff % 2
                for i in range(1, ldiff+1):
                    new_nodeid = hash(node) + i/max_virtual_nodes # creates new nodes ID based on the hash(integer part of the ID) + decimal part
                    nnode_t = layer_t if i % 2 == tflag else other_t 
                    node = nnode_t(new_nodeid, [node]) # instantiates a new virtual node which has the previous node as its child
                    levels[node.id] = nlevel-i # assigns a level to it
                levels[node.id] = nlevel-i
            layers[level].add(node)

        if len(node.children):
            new_level = level + 1
            if new_level > max_level: # should it be necessary, creates a new layer and put all leaves in it
                max_level += 1
                layers[new_level] = leaves.copy()
                for leaf in leaves: # assigns each leaf node a new level
                    levels[leaf.id] = max_level

            for child in node.children: # visits children nodes
                queue.append((child, new_level))

    return layers, levels

""" Creates matrices for connecting consecutive layers to be used by the layer operation itself. """
def build_connections(layers, levelDict, nvars):
    max_level = len(layers) - 1

    connections = [None]*(max_level + 1)
    negated = torch.zeros(len(layers[max_level]))
    parent_dict = {}
    child_dict = {}

    for level, nodeSet in layers.items():
        nodes = sorted(nodeSet, key=lambda node: hash(node))
        rows = len(nodes)
        childSet = set()

        if level == max_level:
            input_mask = torch.zeros(rows, nvars) 

        for node in nodes:
            if type(node) is nnf.LiteralNode:
                childSet.add(node)
            else:
                childSet.update(node.children)

        cols = len(childSet)

        connections[level] = torch.zeros(rows, cols) 
        parent_count = 0
        child_count = 0
        parent_dict = child_dict
        child_dict = {}

        for node in nodes:

            if node in parent_dict:
                parent_idx = parent_dict[node]
            else:
                parent_idx = parent_count
                parent_count += 1
          
            if levelDict[node] == max_level:   
                if level == max_level:
                    input_mask[parent_idx, node.literal-1] = 1
                    negated[parent_idx] = node.negated  #torch.tensor(torch.finfo().tiny)
                children = [node]
            else:
                children = sorted(node.children, key=lambda node: hash(node))

            for child in children:
                if child in child_dict:
                    child_idx = child_dict[child]
                else:
                    child_idx = child_count
                    child_dict[child] = child_idx
                    child_count +=1
                connections[level][parent_idx, child_idx] = 1

    return connections, negated, input_mask

""" A Logic circuit is a sequence of intercalating OR and AND layers operating on a sequence of literal values (via evaluate() method) 
    or an initial node setup as input (via __call__() operator which is dispatched to the inherited forward() method of nn.Sequential)
 """
class LogicCircuit(nn.Sequential):
    def __init__(self, layers, nliterals):
        super().__init__()
        self.layers = layers
        for layer in layers:
            self.append(layer)
        self.nliterals = nliterals

    """ Calls the input layer with a configuration of literal values """
    def evaluate(self, configuration):
        if len(self.layers) == 0:
            raise IndexError("No input layer defined")

        """ Calls the forward() method with two arguments, the input configuration and an input mask """
        return self((configuration, self.layers[0].input_mask, self.layers[0].negated))

    """ Define gains or input weights most likely representing probabilities """
    def set_input_weights(self, value):
        if len(self.layers) == 0:
            raise IndexError("No input layer defined")
        self.layers[0].set_gains(value)

    def get_input_size(self):
        if len(self.layers) == 0:
            raise IndexError("No input layer defined")
        return self.layers[0].linearTransform.weight.size(dim=0)
    
    # def infer(self, literals):
    #     if len(self.layers) == 0:
    #         raise IndexError("No input layer defined")
    #     idxs = torch.nonzero(circuit.layers[0].input_mask[:, literals], as_tuple=True)[0]
    #     neg = torch.zeros_like(circuit.layers[0].negated)
    #     neg[idxs] = circuit.layers[0].negated[idxs]

    #     return self((torch.ones(1, self.nliterals), self.layers[0].input_mask, neg))
    
    def infer(self, literals):
        if len(self.layers) == 0:
            raise IndexError("No input layer defined")
        lit = torch.abs(torch.tensor(literals)) - 1
        idxs = torch.nonzero(self.layers[0].input_mask[:, lit], as_tuple=True)[0]
        neg = torch.zeros_like(self.layers[0].negated)
        neg[idxs] = self.layers[0].negated[idxs]

        conf = torch.ones(1, self.nliterals)
        #conf[0, literals] = 0.0
        for l in literals:
            if l < 0:
                conf[0,abs(l)-1] = 0.0

        return self((conf, self.layers[0].input_mask, neg))
""" Main method to build a LogicCircuit instance based on a NNF file format """
def build_circuit(filename):
    rootId, _, nodeDict, nvars = nnf.parse(filename)

    root = nodeDict[rootId]

    layerSet, levelDict = build_layers(root)   

    connections, negated, input_mask = build_connections(layerSet, levelDict, nvars)

    root_t = type(root)
    rlayer_t = OrLayer if root_t is nnf.OrNode else AndLayer
    olayer_t = OrLayer if root_t is nnf.AndNode else AndLayer 

    max_layers = len(connections) 
    last = max_layers-1

    layers = []

    inputLayer = InputLayer(last, connections[last], negated, input_mask, torch.ones(nvars))
    layers.append(inputLayer)

    for i in range(last-1, -1, -1):
        layer_t = rlayer_t if i % 2 == 0 else olayer_t
        layers.append(layer_t(i, connections[i])) # creates a layer with intercalating types (OR / AND) and corresponding connection matrix
    
    #outputLayer = OutputLayer(connections[0])
    #net.append(outputLayer)

    return LogicCircuit(layers, nvars)

""" Given a file name with both CNF and NNF formats variations, generated a set of inputs and test them against the represented formulas """
def test_configurations(filename = 'simple_w_constraint_opt'):

    clauses, nvars  = cnf.build(filename + '.cnf')

    circuit = build_circuit(filename + '.nnf')

    nconf = 2 ** nvars
    
    print('Total configurations to test = {}'.format(nconf))

    for conf in range(nconf):
        print('\rTesting {}', conf, end='')

        bconf = '{0:b}'.format(conf).zfill(nvars)
        input = []

        for b in bconf:
            input.append(int(b))

        result1 = cnf.evaluate(clauses, input)
        result2 = circuit.evaluate(torch.FloatTensor([input]))

        if result1 or result2:
            print("\nInput: {}, Output1: {} Output2: {}".format(input, result1, result2))


def test_probabilities(filename = 'smoke_pre.cnf'):
    EPSILON = 0.00005

    circuit = build_circuit(filename + '.nnf')
    print("Circuit being tested: ", filename + '.nnf')

    probs = torch.tensor([0.25, 0.25, 0.25, 0.2, 0.2, 0.3, 0.4, 1., 1., 1., 1., 1., 1., 1., 0.333, 0.4992503748125937, 1., 1., 1., 1., 1., 1., 1., 1.]) 
    circuit.set_input_weights(probs)

    output = circuit.infer([14])
    print("Output: ", output)
    print(" [PASSED]" if torch.abs(output - torch.tensor([0.0117])) < EPSILON else " [REJECTED]")

    b = circuit.infer([11])
    output = circuit.infer([11, 14]) / b
    print("Output: ", output)
    print(" [PASSED]" if torch.abs(output - torch.tensor([0.0600])) < EPSILON else " [REJECTED]")

    b = circuit.infer([-9])
    a = circuit.infer([-9, -14])
    output= a/b
    print("Output: ", output)
    print(" [PASSED]" if torch.abs(output - torch.tensor([0.9925])) < EPSILON else " [REJECTED]")

""" Usage examples """
if __name__ == '__main__':

    #test_configurations()
    test_probabilities()

    filename = sys.argv[1]
    c2d_executable = sys.argv[2]
    program_str = ''
    with open(filename) as infile:
        program_str = infile.read()
    database_str = ''
    # if len(sys.argv) > 2:
    #     with open(sys.argv[2]) as infile:
    #         database_str = infile.read()
    program = Program(program_str, database_str)
    if program.grounded_program.check_tightness():
        cnf = program.clark_completion()
        str_cnf = str(cnf).replace("w", "c")
        filename += ".cnf"
        with open(filename, "w") as outfile:
            outfile.write(str_cnf)

    process = Popen([c2d_executable, "-in", filename, "-dt_method", "4"], stdout=PIPE)# ,"-smooth_all"], stdout=PIPE)
    (poutput, perr) = process.communicate()
    exit_code = process.wait()

    if exit_code != 0:
        if poutput:
            print(poutput.decode("utf-8"))
        if perr:
            print(perr.decode("utf-8"))
        exit(exit_code)

    import time
    start_time = time.time()

    circuit = build_circuit(filename + '.nnf')
    print("Time to build the circuit (s): ", time.time() - start_time)

    probs = torch.ones(circuit.nliterals)
    for i in range(20):
          probs[i] = 0.01
    probs[9] = 0.91
    probs[12] = 0.42 
    probs[17] = 0.5 
        
    circuit.set_input_weights(probs)

    input = [32]
    
    start_time = time.time()
    output = circuit.infer(input)

    print("Time to compute inference (s): ", time.time() - start_time)
    print("Circuit: ", filename + '.nnf')
    print("Input (literals): ", input)
    print("Output: ", output)
    print("Weights(probabilities): ", probs)