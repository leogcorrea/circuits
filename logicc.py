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
    def __init__(self, id, matrix, negated):
        super().__init__(id, matrix)
        self.linearTransform = nn.Linear(in_features=matrix.size(dim=1), out_features=matrix.size(dim=0), bias=False)
        self.linearTransform.weight = self.weight
        self.negated = negated

    def forward(self, input):
        y = self.linearTransform(input)
        y = abs(self.negated - y)
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
            if (node_t != layer_t):  # if the node type is different from other nodes in this layer
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
                tflag = ldiff % 2
                for i in range(1, ldiff+1):
                    new_nodeid = -(hash(node) + i/100.0)
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


def build_connections(layers, levelDict, nvars, marginals):
    max_level = len(layers) - 1

    connections = [None]*(max_level + 1)
    negated = torch.zeros(len(layers[max_level]))
    parent_dict = {}
    child_dict = {}

    for level, nodeSet in layers.items():
        nodes = sorted(nodeSet, key=lambda node: abs(int(node.id)))
        rows = len(nodes)
        childSet = set()

        if level == max_level:
            cols = nvars
        else:
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
                    child_idx = node.literal-1
                    negated[parent_idx] = False if marginals[child_idx] else node.negated #torch.tensor(torch.finfo().tiny)
                else:
                    if node in child_dict:
                        child_idx = child_dict[node]
                    else:
                        child_idx = child_count
                        child_dict[node] = child_idx
                        child_count +=1
                connections[level][parent_idx, child_idx] = 1
            else:
                children = sorted(node.children, key=lambda node: abs(node.id))
                for child in children:
                    if child in child_dict:
                        child_idx = child_dict[child]
                    else:
                        child_idx = child_count
                        child_dict[child] = child_idx
                        child_count +=1
                    connections[level][parent_idx, child_idx] = 1

    return connections, negated


class LogicCircuit:
    def __init__(self, layers, nliterals):
        super().__init__()
        self.sequential = nn.Sequential()
        self.layers = layers
        for layer in layers:
            self.sequential.append(layer)
        self.nliterals = nliterals

    def evaluate(self, input):
        return self.sequential(input)


def build_circuit(filename, marginals = None):
    rootId, nodeList, nodeDict, nvars = nnf.parse(filename)

    root = nodeDict[rootId]
    layerSet, levelDict = build_layers(root)   
    margVec= [False]*nvars

    if marginals:
        for interval in marginals:
            for literal in range(interval[0], interval[1]):
                margVec[literal-1] = True

    connections, negated = build_connections(layerSet, levelDict, nvars, margVec)

    root_t = type(root)
    rlayer_t = OrLayer if root_t is nnf.OrNode else AndLayer
    olayer_t = OrLayer if root_t is nnf.AndNode else AndLayer 

    max_layers = len(connections) 
    last = max_layers-1

    layers = []

    inputLayer = InputLayer(last, connections[last], negated)
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
    
    print('Total configurations to test = '.format(nconf))

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

    #filename = 'simple'
    #filename = 'simple_w_constraint_opt'
    #filename = 'montyhall_w_constraint_pre'
    filename = 'smoke_pre.cnf'

    if 0:
        test(filename)

    marginals = [[8, 14], [18, 24]]
    #marginals = None
    circuit = build_circuit(filename + '.nnf', marginals)

    input = torch.tensor([0.25, 0.25, 0.25, 0.2, 0.2, 0.3, 0.4, 1., 1., 1., 1., 1., 1., 1., 0.333, 0.4992503748125937, 1., 1., 1., 1., 1., 1., 1., 1.])
    #input = torch.ones(1, circuit.nliterals)

    output = circuit.evaluate(input)

    print("Circuit: ", filename)
    print("Input:", input)
    print("Output:", output)