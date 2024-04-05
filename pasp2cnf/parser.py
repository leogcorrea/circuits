from lark import Lark, Transformer
import re

GRAMMAR = r'''
    program : ( rule | prob_rule )*

    rule : ( normal_rule | disjunctive_rule | fact | constraint ) "."

    prob_rule : annotated_disjunction "."

    fact : literal

    disjunction : literal (";" literal)+

    disjunctive_rule : disjunction constraint

    normal_rule : literal constraint

    annotated_disjunction : weight "::" literal (";" weight "::" literal)*

    constraint : ":-" body

    body : [ ( literal | arithmetic_atom ) ( "," ( literal | arithmetic_atom ) )* ]

    NEGATION : "not"

    literal : [NEGATION] /[a-z]([a-zA-Z0-9_])*/ [ "(" input ")" ]

    arithmetic_atom : arithmetic_expression comparator arithmetic_expression

    !comparator : "=" | "!=" | "<=" | "<" | ">=" | ">" 

    arithmetic_expression : ( variable | /[0-9\(\)+\-\/\*]+/ )+

    input : term ( "," term )*

    term : literal | /[0-9_\/<>=+"-]([a-zA-Z0-9_\/<>=+".-]*)/ | variable 

    variable : /[A-Z][a-zA-Z0-9]*/

    weight :  /[+-]?([0-9]*[.])?[0-9]+/ | variable

    COMMENT : ("%"|"#")/[^\n]*/
    %ignore COMMENT
    %import common.WS
    %ignore WS
    
'''

class Rule(object):
    """A class for nonprobabilistic rules.

    Implements a custom `__str__` method.

    Args:        
        - head (:obj:`list`): The list of head atoms. May be empty.
        - body (:obj:`list`): The list of body atoms. May be empty.

    Attributes:
        head (:obj:`list`): The list of head atoms. May be empty.
        body (:obj:`list`): The list of body atoms. May be empty.
    """
    def __init__(self, head, body):
        self.head = head
        self.body = body if body is not None else []

    def __str__(self):
        res = ""
        if self.head is not None:
            #res += f"{str(self.head[0])}"
            res += f"{';'.join([str(x) for x in self.head])}"
        if len(self.body) > 0:
            res +=f":-{','.join([str(x) for x in self.body])}."
        else:
            res += "."
        return res

    def __repr__(self):
        return str(self)


class ProbabilisticRule(object):
    """A class for probabilistic rules of the form

        weight1::atom1; ...; weightN::atomN :- body1, ..., not bodyM.

    where weights are nonnegative number whose sum is <= 1.

    Implements a custom `__str__` method.

    Args:        
        - head (:obj:`list`): The list of head atoms. 
        - body (:obj:`list`): The list of body atoms. May be empty.
        - weights (:obj:`list`): The list of weights of the head atoms. 

    Attributes:
        head (:obj:`list`): The list of head atoms. 
        body (:obj:`list`): The list of body atoms. May be empty.
        weights (:obj:`list`): The list of weights of the head atoms. 
    """
    def __init__(self, head, body, weights):
        self.head = head
        self.body = body if body is not None else []
        self.weights = weights

    def __str__(self):
        res = ";".join([ f"{self.weights[i]}::{self.head[i]}" for i in range(len(self.head)) ])
        if len(self.body) > 0:
            res +=f":-{','.join([str(x) for x in self.body])}."
        else:
            res += "."
        return res

    def __repr__(self):
        return str(self)
    
    def get_asp_string(self):
        """Generates an ASP representation of the rule.

        Implements a custom `__str__` method.
        
        Returns:
            :obj:`string`: The representation of this rule as an ASP rule.
        """
        res = ""
        #if len(self.head) == 1: # probabilistic fact
        #    res += f"{{{self.head[0]}}}"
        #else: # annotated disjunction
            # lower = 0
            # if abs(sum(float(w) for w in self.weights)-1) < 1e-9: # if disjunction is exhaustive
            #     lower = 1                       # exactly one atom must be true
            #res += f"{lower}{{{';'.join([ str(atom) for atom in self.head ])}}}1"
        res += f"{{{';'.join([ str(atom) for atom in self.head ])}}}"
        # if self.head is not None:
        #     if self.weights is not None:
        #         res += f"1{{{','.join([ str(atom) for atom in self.head ])}}}1"
        #     else:
        #         res += str(self.head[0])
        if len(self.body) > 0:
            res +=f":-{','.join([str(x) for x in self.body])}."
        else:
            res += "."
        return res


class Literal(object):
    """A class for literals.

    Implements a custom `__str__` method.
    
    Args:
        - predicate (:obj:`string`): The predicate of the atom.
        - inputs (:obj:`list`, optional): The inputs of the atom. These may be strings or other atoms. Defaults to `None`.
        - negated (:obj:`bool`, optional): Whether the atom is negated. Defaults to `False`.

    Attributes:
        predicate (:obj:`string`): The predicate of the atom.
        inputs (:obj:`list`, optional): The inputs of the atom. 
        These may be strings or other atoms. 
        negated (:obj:`bool`, optional): Whether the atom is negated.
    """
    def __init__(self, predicate, inputs = None, negated=False):
        self.predicate = predicate
        self.inputs = inputs if inputs is not None else []
        def replace_quotes(term):
            if type(term) != Literal:
                return term.replace("'", '"')
            return term
        self.inputs = [ replace_quotes(term) for term in self.inputs ]
        self.negated = negated

    def __str__(self):
        res = ""
        if self.negated:
            res += "not "
        res += f"{self.predicate}"
        if len(self.inputs) > 0:
            res += f"({','.join([ str(term) for term in self.inputs ])})"
        return res

    def __repr__(self):
        return str(self)

    def get_variables(self):
        """Recursively finds all the variables used in the atom.

        Returns:
            :obj:`list`: The list of variables as strings.
        """
        vars = set()
        for term in self.inputs:
            if type(term) == Literal:
                vars.update(term.get_variables())
            elif re.match(r"[A-Z][a-zA-Z0-9]*", term):
                vars.add(term)
        return vars

class ArtithmeticAtom(Literal):
    """A class for arithmetic atoms.

    Implements a custom `__str__` method.
    
    Args:
        - predicate (:obj:`string`): The predicate of the atom.
        - inputs (:obj:`list`, optional): The inputs of the atom. 
        - These may be strings or other atoms. Defaults to `None`.
        - negated (:obj:`bool`, optional): Whether the atom is negated. Defaults to `False`.

    Attributes:
        predicate (:obj:`string`): The predicate of the atom.
        inputs (:obj:`list`, optional): The inputs of the atom. 
        These may be strings or other atoms. 
        negated (:obj:`bool`, optional): Whether the atom is negated.
    """
    def __init__(self, predicate, inputs, variables):
        assert(len(inputs) == 2)
        self.variables = set(variables)
        super().__init__(predicate, inputs, negated = False)

    def __str__(self):
        res = str(self.inputs[0])
        res += f" {self.predicate} "
        res += str(self.inputs[1])
        return res

    def get_variables(self):
        """Recursively finds all the variables used in the atom.

        Returns:
            :obj:`list`: The list of variables as strings.
        """
        self.variables


class PaspTransformer(Transformer):
    """The corresponding PASP semantics class for the PASP grammar GRAMMAR.
    
    See the lark documentation for how this works.
    """
    def program(self, ast):  # noqa
        return ast # sort out the comments

    def prob_rule(self, ast):  # noqa
        return ProbabilisticRule(ast[0]['head'], ast[0]['body'], ast[0]['weights'])

    def annotated_disjunction(self, ast): # noqa
        weights = ast[::2]
        head = ast[1::2]
        return { 'head' : head, 'weights' : weights, 'body' : None }

    def rule(self, ast):  # noqa
        return Rule(ast[0]['head'], ast[0]['body'])

    def disjunctive_rule(self, ast):
        #return { 'head': [ast], 'body' : None}
        head = ast[0]
        body = ast[1]
        # print("head:", head, "body:", body)
        return { 'head' : head, 'body': body['body'] }

    def disjunction(self, ast):
        #return { 'head': [ast], 'body' : None}
        return ast

    def fact(self, ast): #noqa
        ast = ast[0]
        return { 'head' : [ast], 'body' : None }
        # if type(ast) == Atom: # we found an atom
        #     return { 'head' : [ast], 'body' : None }
        # else: # we found an annotated disjunction
        #     return ast

    def normal_rule(self, ast):  # noqa
        head = ast[0]
        body = ast[1]
        # print("head:", head, "body:", body)
        return { 'head' : [head], 'body': body['body'] }
        # return { 'head' : ast[0]['head'], 'weights' : ast[0]['weights'], 'body': ast[1]['body'] }

    def body(self, ast):  # noqa
        if len(ast) == 1 and ast[0] == None:
            return None
        return ast

    def constraint(self, ast): #noqa
        return { 'head' : None, 'weights' : None, 'body' : ast[0] }

    def literal(self, ast):  # noqa
        negated = str(ast[0]) == 'not'
        if len(ast) == 3:
            return Literal(str(ast[1]), inputs = ast[2], negated = negated)
        else:
            return Literal(str(ast[1]), negated = negated)      
    
    def arithmetic_atom(self, ast):
        return ArtithmeticAtom(ast[1], [ast[0][0], ast[2][0]], ast[0][1] + ast[2][1])
    
    def comparator(self, ast):
        return ast[0]

    def arithmetic_expression(self, ast):
        str_rep = ""
        variables = []
        for sub in ast:
            if re.match(r"[A-Z][a-zA-Z0-9]*", sub):
                variables.append(sub)
            str_rep += sub
        return (str_rep, variables)

    def input(self, ast):  # noqa
        return ast

    def term(self, ast):  # noqa
        ast = ast[0]
        if type(ast) == Literal:
            return ast
        if "." in ast and (ast[0] != '"' or ast[-1] != '"'):
            return '"' + ast + '"'
        return str(ast)

    def variable(self, ast): # noqa
        return str(ast[0])

    def weight(self, ast):  # noqa
        return str(ast[0])


if __name__ == '__main__':
    import sys
    parser = Lark(GRAMMAR, start='program', parser='lalr', transformer=PaspTransformer())
    with open(sys.argv[1]) as infile:
        tree = parser.parse(infile.read())
        for r in tree:
            print(r, type(r), sep = '\t')
