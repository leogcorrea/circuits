import torch
from torch import nn
import nnf
import cnf
import collections


class Layer(nn.Module):
    def __init__(self, id, matrix):
        super().__init__()
        self.id = id
        self.weight = nn.Parameter(matrix, requires_grad=False)


class InputLayer(Layer):
    def __init__(self, id, matrix, negated, input_mask, gains):
        super().__init__(id, matrix)
        self.linearTransform = nn.Linear(in_features=matrix.size(dim=1), out_features=matrix.size(dim=0), bias=False)
        #self.linearTransform.weight = nn.Parameter(self.weight, requires_grad=False)
        self.negated = negated
        self.input_mask = input_mask
        self.gains = None
        self.set_gains(gains)

    def set_gains(self, gains):
        xgains = torch.matmul(self.input_mask, gains)
        sel = self.negated - xgains
        self.gains = abs(sel) + (sel == 0).type(torch.float) 
        self.linearTransform.weight = nn.Parameter(torch.matmul(torch.diag(self.gains), self.weight), requires_grad=False)

    def set_negated(self, value):
        self.negated = value

    def forward(self, input):
        if len(input) > 1:
            x = torch.matmul(input[0], input[1].t())
            y = abs(self.negated - x)
        else:
            y = self.linearTransform(input[0])
        
        #y = torch.log(x)
        return y
    

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

    
class OutputLayer(OrLayer):
    def __init__(self, id, matrix):
        super().__init__(id, matrix)

    def forward(self, input) -> torch.Tensor:
        y = super().forward(input)
        #y = torch.exp(x)
        return y
    

def get_max_level(node, level = 0):
    max_level = level
    for child in node.children:
        l = get_max_level(child, level+1)
        if l > max_level:
            max_level = l
    return max_level
    

def populate_levels(node, levels, level = 0):
    if node.id in levels:
        current = levels[node.id]
        if current < level:
            levels[node.id] = level
    else:
        levels[node.id] = level

    for child in node.children:
        populate_levels(child, levels, level+1)


def build_layers(root):
    levels = {}
    populate_levels(root, levels)

    MAX_VIRTUAL_NODES = 10.

    max_level = get_max_level(root)
    layers = {i: set() for i in range(max_level+1)}     

    level = 0
    root_t = type(root)
    other_t = nnf.OrNode if root_t is nnf.AndNode else nnf.AndNode 
    
    queue = collections.deque()
    queue.append((root, level))

    new_nodes = {}
    leaves = set()

    get_layer_type = lambda level: root_t if level % 2 == 0 else other_t

    while len(queue):
        node, level = queue.popleft()

        layer_t = get_layer_type(level)
        node_t = type(node)
        nlevel = levels[node.id]
        ldiff = nlevel-level

        if (node_t == nnf.LiteralNode):
            if (node not in leaves):
                leaves.add(node) 
                levels[node.id] = max_level
                for l in range(level, max_level+1):
                    layers[l].add(node)
        else:
            if (node_t != layer_t):
                nlevel = level+1
                levels[node.id] = nlevel

            if level > nlevel:
                levels[node.id] = level 
            elif level < nlevel:
                if node in layers[level]:
                    continue
                if (node_t != get_layer_type(nlevel)):
                    nlevel += 1
                    levels[node.id] = nlevel
                ldiff = nlevel-level
                while ldiff >= MAX_VIRTUAL_NODES:
                    MAX_VIRTUAL_NODES *= 10.0
                tflag = ldiff % 2
                for i in range(1, ldiff+1):
                    new_nodeid = -(hash(node) + i/MAX_VIRTUAL_NODES)
                    nnode_t = layer_t if i % 2 == tflag else other_t 
                    node = nnode_t(new_nodeid, [node])
                    levels[node.id] = nlevel-i
                levels[node.id] = nlevel-i
            layers[level].add(node)

        if len(node.children):
            new_level = level + 1
            if new_level > max_level:
                max_level += 1
                layers[new_level] = leaves.copy()
                for leaf in leaves:
                    levels[leaf.id] = max_level

            for child in node.children:
                queue.append((child, new_level))

    return layers, levels


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
        #    cols = nvars
            input_mask = torch.zeros(rows, nvars) 
        #else:
        for node in nodes:
            if type(node) is nnf.LiteralNode:
                childSet.add(node)
            else:
                for child in node.children:
                    childSet.add(child)

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
                    #child_idx = node.literal-1
                    input_mask[parent_idx, node.literal-1] = 1
                    negated[parent_idx] = node.negated  #torch.tensor(torch.finfo().tiny)
                #else:
                if node in child_dict:
                    child_idx = child_dict[node]
                else:
                    child_idx = child_count
                    child_dict[node] = child_idx
                    child_count +=1
                connections[level][parent_idx, child_idx] = 1
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
    def __init__(self, layers, nliterals):
        super().__init__()
        self.layers = layers
        for layer in layers:
            self.append(layer)
        self.nliterals = nliterals

    def evaluate(self, configuration):
        if len(self.layers) == 0:
            raise IndexError("No input layer defined")

        return self((configuration, self.layers[0].input_mask))

    def set_input_weights(self, value):
        if len(self.layers) == 0:
            raise IndexError("No input layer defined")
        self.layers[0].set_gains(value)


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
        layers.append(layer_t(i, connections[i]))        
    
    #outputLayer = OutputLayer(connections[0])
    #net.append(outputLayer)

    return LogicCircuit(layers, nvars)


def test(filename):

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


if __name__ == '__main__':
    if 1:
        test('simple_w_constraint_opt')
    
    filename = 'smoke_pre.cnf'
    import time
    start_time = time.time()
    circuit = build_circuit(filename + '.nnf')
    print("Time to build the circuit (s): ", time.time() - start_time)

    probs = torch.tensor([0.25, 0.25, 0.25, 0.2, 0.2, 0.3, 0.4, 1., 1., 1., 1., 1., 1., 1., 0.333, 0.4992503748125937, 1., 1., 1., 1., 1., 1., 1., 1.]) 

    circuit.set_input_weights(probs)

    input = torch.ones(1, 48)
    input[0, 14+23] = 0.0
    output = circuit(input)
    print("Output: ", output)

    input = torch.ones(1, 48)
    input[0, 11+23] = 0.0
    b = circuit(input)

    input[0, 14+23] = 0.0
    output = circuit(input) / b
    print("Output: ", output)

    input = torch.ones(1, 48)
    input[0, 8] = 0.0
    b = circuit(input)

    start_time = time.time()

    input[0, 13] = 0.0
    a = circuit(input)
    output= a/b

    print("Time to compute inference (s): ", time.time() - start_time)
    print("Circuit: ", filename + '.nnf')
    print("Input: ", input)
    print("Output: ", output)