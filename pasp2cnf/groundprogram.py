from dataclasses import dataclass
import clingo
from typing import List, Iterable, Union, Dict
import networkx as nx
from .wcnf import WCNF

class GroundObject(object):
    pass

@dataclass
class GroundAtom(GroundObject):
    symbol: clingo.Symbol
    atom: int
    order: int = 0

    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return (self.symbol, self.atom) < (other.symbol, other.atom)
        elif isinstance(other, GroundObject):
            return self.order < other.order
        raise Exception("Incomparable type")

@dataclass
class GroundRule(GroundObject):
    choice: bool
    head: List[int]
    body: List[int]
    order: int = 1
    
    def __lt__(self, other):
        if self.__class__ == other.__class__:
            return (self.choice, self.head, self.body) < (other.choice, other.head, other.body)
        elif isinstance(other, GroundObject):
            return self.order < other.order
        raise Exception("Incomparable types")
    
    def __str__(self):
        head = "; ".join([f"X{a}" for a in self.head])
        if self.choice:
            head = "{" + head + "}"
        body = ", ".join([ ("not X" if b < 0 else "X") + str(abs(b)) for b in self.body])
        if body:
            head += ":-" + body 
        return head + "."


class Observer:
    def __init__(self, program):
        self.program = program

    def rule(self, choice: bool, head: List[int], body: List[int]) -> None:
        self.program.rules.append(GroundRule(choice=choice, head=head, body=body))

    def output_atom(self, symbol: clingo.Symbol, atom: int) -> None:
        self.program.atoms.append(GroundAtom(symbol=symbol, atom=atom))

@dataclass
class GroundProgram():
    ''' A class to compactly represent a ground program, as one output by gringo.

    Atoms are represented as positive integers, and rules are 
    represented as lists of signed integers, with negative numbers
    denoting negated literals.
    '''
    rules: List[GroundRule]
    atoms: List[GroundAtom]
    _atom_to_txt: Dict[int,str]
    _derived_from: Dict[int,GroundRule]
    _atom_to_weight: Dict[int,float]

    ''' Initializes and obtains a ground program by calling gringo
    on a given ASP program string. Ground atoms that have weights are interpreted
    as probabilistic choices and expected not to unify with a single rule head.
    '''  
    def __init__(self, program_str:str, weight_function:Dict[str,float]):
        # initialize variables
        self.rules = []
        self.atoms = []   
        # maps each atom index to its weight
        self._atom_to_weight = {}
        # maps each atom index to its textual representation
        self._atom_to_txt = {}
        # for each head atom, which rules can derive it?
        self._derived_from = {}
        # strongly connected components of positive dependncy graph
        self._components = None
        # if string is empty, there's nothing to do
        if not program_str:
            return    
        self._ground(program_str, weight_function)

    def _ground(self, program_str:str, weight_function:Dict[str,float]) -> None:    
        # ground program
        control = clingo.Control()
        control.add("base", [], program_str)
        control.register_observer(Observer(self))
        control.ground([('base', [])]) 
        self._process_output(control, weight_function)

    def _process_output(self, control:clingo.Control, weight_function:Dict[str,float]) -> None:
        print("Literal : program clause")
        # process output
        for sym in control.symbolic_atoms:
            symbol = str(sym.symbol)
            self._atom_to_txt[sym.literal] = symbol
            if symbol in weight_function:
                self._atom_to_weight[sym.literal] = weight_function[symbol]
            print(sym.literal, ": " + symbol)
        for r in self.rules: 
            if not(r.choice and len(r.head) > 0 and len(r.body) == 0 and self._atom_to_txt[abs(r.head[0])] in weight_function):
                
                # it is a normal rule        
                for a in r.head:
                    if a not in self._derived_from:
                        self._derived_from[a] = []
                    self._derived_from[a].append(r)
        choices = set(self._atom_to_weight.keys())
        if len(choices.intersection(self._derived_from.keys())) != 0:
            raise Exception("Syntax Error: Probabilistic choice unifies with rule head")

    def add_rule(self, choice: bool = False, head: Iterable[int] = [], body: Iterable[int] = []) -> None: # pylint: disable=dangerous-default-value
        self.rules.append(GroundRule(choice=choice, head=list(head), body=list(body)))

    def add_rules(self, rules: Iterable[GroundRule]) -> None:
        self.rules.extend(rules)

    def _remove_tautologies(self):
        rules = []
        for r in self.rules:
            if set(r.head).intersection(set(r.body)) == set():
                rules.append(r)
        self.rules = rules

    def _compute_scc(self):
        ' Computes the SCCS of the positive dependency graph of grounded program. '
        self.dep = nx.DiGraph()
        self.dep.add_nodes_from(self._atom_to_txt.keys())
        for r in self.rules:
            for a in r.head:
                for b in r.body:
                    if b > 0:
                        self.dep.add_edge(b, a)
        comp = nx.algorithms.strongly_connected_components(self.dep)
        self._components = list(comp)
        self._condensation = nx.algorithms.condensation(self.dep, self._components)

    def check_tightness(self) -> bool:
        'Checks if grounded program is tight (or head-cycle free)'
        if not self._components:
            self._compute_scc()
        for comp in self._components:
            if len(comp) > 1:
                return False
        return True

    def shift(self):
        ''''
        Apply shifting to disjunctive rules (Ben-Eliyahu and Dechter, 1994), 
        Assumes (but does not check) that the program is head cycle free.
        '''
        new_rules = []
        for r in self.rules:
            if len(r.head) > 1:          
                for a in r.head:
                    ext = [b for b in r.head if b != a]
                    new_rules.append( GroundRule(a, r.body + ext) )
            else:
                new_rules.append(r)
        self.rules = new_rules

    def clark_completion(self) -> WCNF:
        ''' Obtains the Clark completion of the program

        Assumes (but does not check) that the program is tight.
        Returns a (weighted) CNF.        
        '''
        # TODO: obtain clark completion of program in CNF form
        wcnf = WCNF()
        wcnf.num_vars = len(self._atom_to_txt) # no. of ground atoms

        # handle derived atoms
        for head in self._derived_from:
            # encode head <=> body_1 or body_2 or ... for each rule deriving atom head
            ors = []
            for r in self._derived_from[head]:
                ors.append(wcnf.new_aux_var()) # create auxiliary variable for rule
                ands = [-x for x in r.body]
                wcnf.clauses.append([ors[-1]] + ands) # body => aux_var
                for l in r.body: # literal => aux_var
                    wcnf.clauses.append([-ors[-1], l]) 
            # head => aux_var_1 or aux_var_2 or ...
            wcnf.clauses.append([-head] + ors)
            for o in ors: # aux_var => head
                wcnf.clauses.append([head, -o])

        # handle the constraints
        constraints = [r for r in self.rules if len(r.head) == 0]
        for r in constraints:
            wcnf.clauses.append([-x for x in r.body])
         
        # handle atoms that never appear as heads - gringo should do that for us
        # false_atoms = set(self._atom_to_txt.keys())
        # false_atoms.difference_update(self._atom_to_weight.keys())
        # false_atoms.difference_update(self._derived_from.keys())
        # for a in false_atoms:
        #     wcnf.clauses.append([-a])

        # handle probabilistic facts
        for a in self._atom_to_weight:
            wcnf.weights[a] = self._atom_to_weight[a]
            if -a not in self._atom_to_weight:
                wcnf.weights[-a] = 1.0-self._atom_to_weight[a]

        # TODO handle the annotated disjunctions by enconding head as exactly-one-of constraint
        # We need to fix the parsing of weights first (which only allows for prob facts currently)
        # for r in self.rules:
        #     if r.choice and len(r.head) > 1:
        #         # at least one
        #         wcnf.clauses.append(list(r))
        #         # at most one
        #         for v in r:
        #             for vp in r:
        #                 if v < vp:
        #                     wcnf.clauses.append([-v, -vp])

        return wcnf
        # TODO: use graph too guide completion
        # if not self._components:
        #     self._compute_scc()
        # # process nodes in topological ordering (if possible)
        # ts = nx.topological_sort(self._condensation)
        # ancs = {}
        # decs = {}
        # for t in ts:
        #     comp = self._condensation.nodes[t]["members"]
        #     for v in comp:
        #         ancs[v] = set([vp[0] for vp in self.dep.in_edges(nbunch=v) if vp[0] in comp])
        #         decs[v] = set([vp[1] for vp in self.dep.out_edges(nbunch=v) if vp[1] in comp])        

    def _pretty_print_literal(self, literal:int) -> str:
        if literal < 0:
            atom_str = "not "
            literal = abs(literal)
        else:
            atom_str = ""
        if literal in self._atom_to_txt:
            atom_str += self._atom_to_txt[literal]
        else:
            # i = abs(literal)
            # while True:
            #     lit = Function('x_' + str(i))
            #     if lit not in self.symbols:
            #         break
            #     i += 1
            atom_str += "UNKNOWN"
        return atom_str
    
    def _pretty_print_rule(self,r: GroundRule) -> str:
        if r.choice and len(r.head) > 0 and len(r.body) == 0 and abs(r.head[0]) in self._atom_to_weight:
            # probabilistic choice
            return ';'.join( f"{self._atom_to_weight[a]}::{self._atom_to_txt[abs(a)]}" for a in r.head) + "."
        # non-probabilistic rule
        head = ';'.join(self._pretty_print_literal(literal) for literal in r.head)
        body = ','.join(self._pretty_print_literal(literal) for literal in r.body)
        if body:
            return head + ':-' + body + '.'
        else:
            return head + '.'

    def __str__(self):
        return '\n'.join(self._pretty_print_rule(r) for r in self.rules)

    def __iter__(self):
        return iter(self.rules)