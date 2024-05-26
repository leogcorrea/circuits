import torch
from torch import nn
import nnf
import cnf
from pasp2cnf.program import Program
from collections import deque
import sys
from subprocess import Popen, PIPE
from os import getcwd

class Layer(nn.Module):
    """ Base class for the probabilistic circuit. Represents an abstract layer.  """
    def __init__(self, id, matrix):
        super().__init__()
        self.id = id
        self.weight = nn.Parameter(matrix, requires_grad=False)


class InputLayer(Layer):
    """ The input layer which operates a linear transform on the inputs. It receives: 
    a) an input configuration (comprising literal values); or 
    b) the input node values themselves. """

    def __init__(self, id, matrix, negated, input_mask, gains):
        super().__init__(id, matrix)
        self.linear_transform = nn.Linear(in_features=matrix.size(dim=1), out_features=matrix.size(dim=0), bias=False)
        #self.linear_transform.weight = nn.Parameter(self.weight, requires_grad=False)
  
        self.register_buffer("negated", negated)
        self.register_buffer("input_mask", input_mask)
        self.register_buffer("_mask", input_mask.t())
        self.register_buffer("gains", None)
        self.gain_set = None
        self.set_gains(gains, [])
        self.gain_set = False
    
    def set_gains(self, gains, surrogate):
        """ Gains (weights) applied by the linear transform on the inputs """
        self.gains = gains # store gains for future reference
        xgains = torch.matmul(self.input_mask, gains) # map gain values to internal nodes organization
        sel = self.negated - xgains # build a negation mask for gains
        # select those node acting as surrogate facts for annotated disjunctions 
        idxs = torch.nonzero(self.input_mask[:, surrogate], as_tuple=True)[0] 
        if len(idxs):
            sel[idxs] = torch.where(sel[idxs] > 0, 1, sel[idxs]) # the surrogate facts have their negation equalled to 1.0
        
        xgains = abs(sel) + (sel == 0).float() # applies the negation mask to gains
        # set the linear transform accordingly
        self.linear_transform.weight = nn.Parameter(torch.matmul(torch.diag(xgains), self.weight), requires_grad=False) 
        self.gain_set = True

    
    def set_negated(self, value):
        """ Mask to obtain the negation of input literal according to the circuit setup """
        self.negated = value

    
    def forward(self, input):
        """ The linear transform application """

        if len(input) > 1:
            # transforms literal values into internal node states according to the circuit setup
            configuration = input[0]
            negated = input[1]
            x = torch.matmul(configuration, self._mask) # transform the input to a internal state vector
            y = abs(negated - x) # aplies a negation mask to the nodes

        else:
            # or operates on the node states directly
            y = input[0] 
        if self.gain_set:
            y = self.linear_transform(y)
        
        #y = torch.log(x)
        return y
    

class AndLayer(Layer):
    """ A circuit layer which calculate an AND operation (product) of the input """

    def __init__(self, id, matrix):
        super().__init__(id, matrix)
        #self.linear_transform = nn.Linear(in_features=matrix.size(dim=1), out_features=matrix.size(dim=0), bias=False)
        #self.linear_transform.weight = nn.Parameter(matrix, requires_grad=False)
        self.register_buffer("_mask", (matrix == 0))
        
    def forward(self, input):
        #return self.linear_transform(input)
        x = torch.mul(self.weight, input) # transform input to the layer internal state 
        # set zero-valued elements to one so as to make them act as neutral elements in the product of columns calculated next
        x += self._mask 
        y = torch.prod(x, dim = 1)
        return y


class OrLayer(Layer):
    """ A OR layer which sum up the input """

    def __init__(self, id, matrix):
        super().__init__(id, matrix)
        self.linear_transform = nn.Linear(in_features=matrix.size(dim=1), out_features=matrix.size(dim=0), bias=False)
        self.linear_transform.weight = self.weight

    def forward(self, input):
        # x = torch.exp(input)
        # x = self.linear_transform(x)
        # y = torch.log(x)
        
        # # xx = torch.mul(self.weight, input)
        # # mask = torch.tensor(self.weight != 0)
        # # xx = xx + torch.tensor(torch.finfo().tiny)
        # # yy = torch.logsumexp(xx, dim=1)

        # apply a simple linear transform to sum-up input elements (internally bringing input to the internal layer representation)
        return self.linear_transform(input)


class OutputLayer(OrLayer):
    """ Currently not being used. Could be use in the future for reading out the results of the circuit just calculated """
    def __init__(self, id, matrix):
        super().__init__(id, matrix)

    def forward(self, input) -> torch.Tensor:
        y = super().forward(input)
        #y = torch.exp(x)
        return y
    

def get_max_level(node, level = 0):
    """ Get the max level (depth in terms of layers) found in the circuit """
    max_level = level
    for child in node.children:
        l = get_max_level(child, level+1)
        if l > max_level:
            max_level = l
    return max_level
    

def populate_levels(node, levels, level = 0):
    """ Determines the depth level (an increasing integer, zero based) in which each NNF node appears in the circuit.
    Input: 
        node: the NNF node to start from
        levels: a possibly empty dict of levels associating each node ID to a integer value (the level)
    Output: a populated dict of levels 
    """
    if node.id in levels:
        current = levels[node.id]
        if current < level:
            levels[node.id] = level
    else:
        levels[node.id] = level

    for child in node.children:
        populate_levels(child, levels, level+1)


def build_layers(root):
    """ Construct a dict of levels and related sets of nodes separating them in layers and organized as intercaling OR and AND layers. 
    May possibly create the so called 'virtual nodes' to aggregate multiple nodes in one layer belonging to the same
    layer operation (OR or AND). 

    Algorithm insights:
        The algorithm transform individual nodes in layers of nodes operating the same (AND/OR) logical function on nodes with similar types. 
        In order to organize the layers types, it starts by assigning the first layer the type of the root node. From there down on the node tree,
        it alternates between AND/OR layer types, adding a current processed node to the current layer if its type is the same of that of the layer
        or creating a 'virtual node' with only one child (the node itself) so as to delegate to the next layer the operation on the referred
        node. The algorithm outputs a dictionary of layers containing sets of nodes in each layer and a index of levels where each node is located
        inside this layering organization. 
    Input: 
        root: NNF node of the tree representing the circuit
    Output:
        layers: a dict of layer level (0-based, increasing integers) to Node sets
        levels: a dict of Node ids to layer levels (0-based, increasing integers)
    """
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
            if node in layers[level]:
                continue
            if level > nlevel:
                levels[node.id] = level # Update the max level this node appears just in case it has been moved down due to a new layer added
            elif level < nlevel:  # this case requires that if the node is not in the current layer, creates multiple virtual nodes to populate
                                  # the 

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


def build_connections(layers, levelDict, nvars):
    """ Creates matrices for connecting consecutive layers of the circuit. The matrices have elements equal to 1 if there exists
    a connection between a i-th node of the current layer to the j-th node of the next layer (i: row index, j: column index)
    and 0 elsewhere.
        The 'negated' mask output is used to keep track of input nodes with a negated value of the corresponding literal.
        The 'input_mask' is used to transform a node input to the circuit internal state.  
    Inputs:
        layers: a dict of layer ids and node sets
        levelDict: a index (dict) of node ids to layer ids
        nvars: number of literals in the circuit
    Outputs: 
        connections: the connection matrices for each layer 
        negated: mask to obtain negation of input values
        input_mask: mask to transform input values into node states """

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


class LogicCircuit(nn.Sequential):
    """ A Logic circuit is a sequence of intercalating OR/AND layers. It can be operated on a sequence of input node values 
   (via evaluate() method) or on a sequence of literals intended for making a probabilistic inference (via query() method).
   By associating weights to the inputs, this can be interpreted as probabilistic evidence on input literals, and then the 
   circuit behaves as a probabilistic inference calculator.
    """
    def __init__(self, layers, nliterals):
        super().__init__()
        self.layers = layers
        for layer in layers:
            self.append(layer)
        self.nliterals = nliterals
        self.probnorm = 1.0
        self._device = torch.device("cpu")

    def to(self, device):
        """ Used to move the circuit to the CPU/GPU """
        super().to(device)
        self._device = device

    def evaluate(self, configuration):
        """ Calls the input layer with a configuration of node values
        Input:
            configuration: an input torch.tensor with dimensions given by a call to the get_input_size() method
        Output:
            a float point torch.tensor representing the circuit output
        """
        if len(self.layers) == 0:
            raise IndexError("No input layer defined")

        """ Calls the forward() method with two arguments, the input configuration and an negation mask """
        return self((configuration, self.layers[0].negated))

    def set_input_weights(self, value, surrogate = []):
        """ Define gains or input weights representing probabilities. 
        Surrogate defines probabilistic surrogate facts with negation equals to 1.0 (used to implement annotaded disjunctions)"""
        
        if len(self.layers) == 0:
            raise IndexError("No input layer defined")
        self.layers[0].set_gains(value, surrogate)
        ones = torch.ones(1, self.get_input_size()).to(self._device)
        self.probnorm = self(ones)
   
    def get_input_size(self):
        """ Returns the expected size of an input configuration """
        if len(self.layers) == 0:
            raise IndexError("No input layer defined")
        return self.layers[0].linear_transform.weight.size(dim=0)
   
    def query(self, literals = []):
        """ Makes an inference or query for the provided literals
        Input:
            literals: a list of numerical literals ids
        Output:
            normalized probability of query
        """ 

        if len(self.layers) == 0:
            raise IndexError("No input layer defined")
        
        conf = torch.ones(1, self.nliterals).to(self._device)
        neg = torch.zeros_like(self.layers[0].negated).to(self._device)

        if len(literals):
             literals = torch.tensor(literals).to(self._device)
             lit = torch.abs(literals) - 1
             conf[0, lit[literals < 0]] = 0.0
             idxs = torch.nonzero(self.layers[0].input_mask[:, lit], as_tuple=True)[0].to(self._device)
             neg[idxs] = self.layers[0].negated[idxs]

        return self((conf, neg)) / self.probnorm
    


def build_circuit(root, nvars):
    """ Main method to build a LogicCircuit instance given a NNF root node and the number of literals """

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


def build_circuit_from_file(filename):
    """ Main method to build a LogicCircuit out of a NNF file """

    rootId, _, nodeDict, nvars = nnf.parse(filename)
    
    return build_circuit(nodeDict[rootId], nvars)


def test_configurations(filename = 'simple_w_constraint_opt'):
    """ Given a file name with both CNF and NNF formats variations, generate a set of inputs and test them against the represented formulas """

    clauses, nvars  = cnf.build(filename + '.cnf')

    circuit = build_circuit_from_file(filename + '.nnf')

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


def make_query(expr, symbols):
    """ Helper function to make a query given a expression and a dict of symbols """
    return [symbols[expr] if not 'not' in expr else -symbols[expr.replace('not', '').lstrip()]]


def test_probabilities(c2d_executable):
    """ Tester function to the ASP program given in the file smoke.pasp. Expects a path to the c2d compiler """

    filename = 'smoke.pasp'
    EPSILON = 0.00005

    filename, symbols = pasp2cnf(filename)
    filename = cnf2nnf(filename, c2d_executable)
    circuit = build_circuit_from_file(filename)
    print("Circuit being tested: ", filename + '.nnf')
    
    lit2idx = lambda lit: lit-1
    sym2lit = lambda sym: symbols[sym] if not 'not' in sym else -symbols[sym.replace('not', '').lstrip()]
    sym2idx = lambda sym: lit2idx(sym2lit(sym))

    probs = torch.ones(circuit.nliterals)
    probs[sym2idx('a(bill)')] = 0.25
    probs[sym2idx('b(carol)')] = 0.25
    probs[sym2idx('c(daniel)')] = 0.25
    probs[sym2idx('d(carol,anna)')] = 0.2
    probs[sym2idx('e(bill,anna)')] = 0.2
    probs[sym2idx('influences(bill,anna)')] = 0.3
    probs[sym2idx('influences(carol,anna)')] = 0.4
    probs[sym2idx('stress(bill)')] = 0.333
    probs[sym2idx('stress(carol)')] = 0.333
    probs[sym2idx('stress(daniel)')] = 0.334

    idx1 = sym2idx('stress(bill)')
    idx2 = sym2idx('stress(carol)')
    idx3 = sym2idx('stress(daniel)')
    surrogate = [idx1, idx2, idx3]

    circuit.set_input_weights(probs, surrogate)

    output = circuit.query([sym2lit('smokes(anna)')])

    print("Query: ", 'smokes(anna)')
    print("Output: ", output)
    print(" [PASSED]" if torch.abs(output - torch.tensor([0.0117])) < EPSILON else " [REJECTED]")

    b = circuit.query([sym2lit('smokes(bill)')])
    output = ( circuit.query([sym2lit('smokes(anna)'), sym2lit('smokes(bill)')])) / b

    print("Query: ", 'smokes(anna) | smokes(bill)')
    print("Output: ", output)
    print(" [PASSED]" if torch.abs(output - torch.tensor([0.0600])) < EPSILON else " [REJECTED]")

    b = circuit.query([sym2lit('not stress(carol)')])
    a = circuit.query([sym2lit('not smokes(anna)'), sym2lit('not stress(carol)')])
    output= a/b

    print("Query: ", 'not smokes(anna) | not stress(carol)')
    print("Output: ", output)
    print(" [PASSED]" if torch.abs(output - torch.tensor([0.9925])) < EPSILON else " [REJECTED]")


def pasp2cnf(filename):
    """ Given an ASP (or annotated ASP) program, outputs a CNF form of it with a dict of symbols (clauses in the program) """
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

    return filename, program.grounded_program.symbol2literal


def cnf2nnf(filename, c2d_executable):
    """ Given a CNF as an input file, writes out a NNF version of it. """
    process = Popen([c2d_executable, "-in", filename, "-dt_method", "4"], stdout=PIPE)
    (poutput, perr) = process.communicate()
    exit_code = process.wait()

    if exit_code != 0:
        if poutput:
            print(poutput.decode("utf-8"))
        if perr:
            print(perr.decode("utf-8"))
        exit(exit_code)
    
    return filename + '.nnf'



if __name__ == '__main__':
    """ Usage example """

    filename = sys.argv[1] if len(sys.argv) > 1 else  getcwd() + '/digits.pasp'
    c2d_executable = sys.argv[2] if len(sys.argv) > 2 else getcwd() + '/c2d_linux'

    TESTS = 1
    if TESTS:
        test_configurations()
        test_probabilities(c2d_executable)