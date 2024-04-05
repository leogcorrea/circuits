'''
Represents and manipulates a Probabilistic Answer Set Program

heavily inpisred by ASPM: https://github.com/raki123/aspmc/blob/main/aspmc/programs/program.py
'''
from dataclasses import dataclass
from lark import Lark
from .parser import PaspTransformer as PaspParser, GRAMMAR as PaspGrammer, Rule, ProbabilisticRule
 
import clingo
import re
from typing import List, Iterable, Union, Dict
import networkx as nx
from .groundprogram import GroundProgram
from .wcnf import WCNF

class Program:
    ''' A class for manipulating PASP Programs.

    Language features supported:
        - Ground annotaded disjunctions (body-free).
        - Disjunctive rules (with variables) with arithmetic constraints.
        - Choice rules.
        - Constraints.

    No direcrives are supported (query, semantics, etc), as the purpose is to
    compile the program into a circuit (which can then be used to answer queries
    under an appropriate semantics).

    Args:
        - program_str: A (possibly empty) string containing the possibly non-ground part of the program. 
        - database_str: A (possibly empty) string containing the ground part of the program. 
       
    '''
    def __init__(self, program_str:str, database_str:str, verbosity=0):
        # self._anotated_disjunctions = []
        self._weight_function = {} # choices to weights
        self._rules = []
        self.grounded_program = None
        if not program_str and not database_str:
            return
        # parse program string
        self.parse(program_str + database_str)
        # then ground it
        self.ground()

    def parse(self,program_str:str) -> None:
        the_parser = Lark(PaspGrammer, start='program', parser='lalr', transformer=PaspParser())
        self.program = the_parser.parse(program_str)

    def _to_asp(self) -> str:
        ''' Translates PASP to ASP for grounding.

        Converts annotated disjunction into choice rules
        and stores weights of probabilistic choices.
        '''
        # TO-DO: translate choice rules to normal rules
        # TO-DO: accept probabilistic rules with (positive) bodies.
        asp_program = []
        for line,r in enumerate(self.program):
            if isinstance(r, ProbabilisticRule):
                # rewrite probabilistic rules as choice rules
                asp_program.append(r.get_asp_string())
                # we assume that probabilistic rules are ground and their atoms do not 
                # unify with atoms in other rule's head, i.e., 
                # there is a function probabilistic fact -> weight
                for atom, weight in zip(r.head, r.weights):
                    #TODO: allow for annotated disjunctions by assigning weights and exactly_one_of contraints
                    if len(atom.get_variables()) > 0:
                        raise Exception(f"Syntax Error: Found non-ground probabilistic rule in Line {line+1}\n>>\t" + str(r))
                    # self._probabilistic_atoms.add(atom)
                    self._weight_function[str(atom)] = float(weight)
            elif isinstance(r, Rule):
                asp_program.append(str(r))
        return asp_program

    def ground(self) -> None:
        'Ground program' 
        # apply translation to ASP (so that gringo can be called on)        
        asp_program_str = '\n'.join(str(r) for r in self._to_asp())
        self.grounded_program = GroundProgram(asp_program_str, self._weight_function)

    def shift(self):
        ''''
        Apply shifting to disjunctive rules (Ben-Eliyahu and Dechter, 1994), 
        Assumes (but does not check) that the program is head cycle free.
        '''
        if self.grounded_program:
            return self.grounded_program.shift()

    def clark_completion(self) -> WCNF:
        ''' Obtains the Clark completion of the program

        Assumes (but does not check) that the program is tight.
        Returns a (weighted) CNF.        
        '''
        if self.grounded_program:
            return self.grounded_program.clark_completion()
            
    def __str__(self):
        max_lines = 30 # TODO: make this configurable somewhere else
        lines = []
        for i,r in enumerate(self.program):
            if i >= max_lines:
                lines.append(f'and {len(self.grounded_program.rules)-i+1} other rules...') 
                break
            lines.append(str(r))
        return '\n'.join(lines)

    #     self.print_grounded_program()

if __name__ == '__main__':
    '''
    Verbosity Level: 
        0 - no output
        1 - print input program
        2 - print input program and grounded program
        3 - print input program, grounded program and CNF
    '''
    verbosity = 3
    import sys
    program_str = ''
    with open(sys.argv[1]) as infile:
        program_str = infile.read()
    database_str = ''
    if len(sys.argv) > 2:
        with open(sys.argv[2]) as infile:
            database_str = infile.read()
    program = Program(program_str, database_str)
    if verbosity > 0:
        print('--- INPUT PROGRAM ---') 
        print(program)       
        if verbosity > 1:
                print('--- GROUNDED PROGRAM --- ')
                print(program.grounded_program)
        # Is program tight?
        if program.grounded_program.check_tightness():
            print("\nPROGRAM IS TIGHT\n")
            cnf = program.clark_completion()
            if verbosity > 2:
                print('--- CLARK COMPLETION --- ')
                print(cnf)
        else:
            print("PROGRAM IS NOT TIGHT")