
class Node():
    """ Abstract base node """
    def __init__(self, id, children):
        super().__init__()
        self.id = id
        self.children = children
    
    def __hash__(self) -> int:
        return abs(int(self.id))
    
    def __eq__(self, __value: object) -> bool:
        return abs(int(self.id)) == __value

    def __str__(self) -> str:
        return self.id

class AndNode(Node):
    """ AND node in a NNF """
    def __init__(self, id, children):
        super().__init__(id, children)


class OrNode(Node):
    """ OR node in a NNF """
    def __init__(self, id, children, conflictingVar = 0):
        super().__init__(id, children)
        self.conflictingVar = conflictingVar
 

class LiteralNode(Node):
     """ LITERAL node in a NNF """
     def __init__(self, id, literal, negated = False):
        super().__init__(id, [])
        self.literal = literal
        self.negated = negated


def _parseNodes(filename):
    foundHeader = False

    nodeIndex = 0
    allNodes = []
    nodeList = []
    refNodes = set()

    nodeDict = {}
    
    with open(filename) as f:
        for line in f:
            nodeId = nodeIndex
            nodeInfo = line.split()

            if not foundHeader:
                if nodeInfo[0] == 'nnf':
                    nvertices = int(nodeInfo[1])
                    nedges = int(nodeInfo[2])
                    nvars = int(nodeInfo[3] )
                    foundHeader = True
                    continue
                else:
                    raise TypeError("Missing header!")
 
            if nodeInfo[0] == 'L':
                var = int(nodeInfo[1])
                node = LiteralNode(nodeId, abs(var), var < 0)
            else:
                if nodeInfo[0] == 'A':
                    node = AndNode(nodeId, [])    
                    offset = 1
                elif nodeInfo[0] == 'O':
                    j = int(nodeInfo[1])
                    node = OrNode(nodeId, [], j)
                    offset = 2
                else:
                    raise TypeError('Unknown node type!')
                n = int(nodeInfo[offset]) 
                for i in range(n):
                    childId = int(nodeInfo[offset+1+i])
                    refNodes.add(childId)
            
            nodeDict[nodeId] = node
            allNodes.append(nodeInfo)
            nodeList.append(nodeId)
            nodeIndex += 1

    rootSet = set(nodeList) - refNodes
    noutput = len(rootSet)

    if noutput != 1:
        raise TypeError("Root note should be unique.")
    
    rootId = rootSet.pop()

    return rootId, allNodes, nodeDict, nvars


def parse(filename):
    """ Parse an input NNF file into a tree of nodes """
    rootId, allNodes, nodeDict, nvars = _parseNodes(filename)

    for nodeId, nodeInfo in enumerate(allNodes):
        children = []
        node = nodeDict[nodeId]
        
        if nodeInfo[0] == 'A':
            offset = 1
        elif nodeInfo[0] == 'O':
            offset = 2
        else:
            continue

        n = int(nodeInfo[offset]) 
        for i in range(n):
            childId = int(nodeInfo[offset+1+i])
            child = nodeDict[childId]
            children.append(child)
        node.children = children

    return rootId, allNodes, nodeDict, nvars