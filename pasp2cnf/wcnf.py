'''
Represents and manipulates weighted Boolean formulas in CNF format

heavily inpisred by ASPM: https://github.com/raki123/aspmc/blob/main/aspmc/compile/cnf.py
'''

class WCNF:
    ''''
    Represents a weighted CNF Boolean formula, that is, a conjunction of clauses, 
    plus a weight function that assigns non-negative real values to literals in
    the formula. It also distinguishes regular and auxiliary variables; the latter
    can be projected away without altering the logical equivalence or model count of
    the Boolean formula (such variables are generally inserted for representing the 
    truth-value of rules, to obtain a more succinct encoding). The main use of this 
    class is to interact with knowledge compilation tools such as C2D.

    Attributes:
        - num_vars: an integer specifying the number of variables (represented as integers 1,...,num_vars)
        - clauses: a list of clauses, each clause being a list of integers denoting literals
        - auxiliary: a set of auxiliary variables
        - weights: a dict mapping literals to floats
    '''
    def __init__(self):
        self.num_vars = 0
        self.clauses = []
        self.auxiliary = set()
        self.weights = {}

    def new_aux_var(self):
        ' helper method to get a new auxilliary variable '
        self.num_vars += 1
        self.auxiliary.add(self.num_vars)
        return self.num_vars        

    def __str__(self):
        ret = f"p cnf {self.num_vars} {len(self.clauses)}\n"
        for idx in self.weights:
            ret += f"w {idx} {self.weights[idx]} 0\n"
        for clause in self.clauses:
            ret += f"{' '.join([str(l) for l in clause])} 0\n"
        if self.auxiliary:
            ret += f"c p auxiliary {' '.join([str(x) for x in self.auxiliary])} 0\n"
        return ret

    def to_file(self, filename, weighted = True, format = "MC2023"):
        """ Write the CNF to given filename in DIMACS format.        

        Args:
            filename (string): The filename where the CNF should be written to.
            weighted (bool, optional): If true (default), outputs a wcnf format; otherwise writes only cnf.
            format (string): format to write literal weights (if weighted); either "MC2023" or "MC2021", according to the Model Counting Competitions
        Returns:
            None
        """
        pass # TODO

    def from_file(self, filename):
        ''' Reads formula from file in DIMACS format.

        Assumes weighted CNF are formatted according to the Model Counting Competitions of 2021 or 2023
        (e.g., for the MC2021 format see https://mccompetition.org/assets/files/2021/competition2021.pdf)
        '''
        with open(filename) as in_file:
            for line in in_file:                
                line = line.strip()
                if not line:
                    continue
                if line[0] == 'p':
                    if line.startswith('p cnf') or line.startswith('p wcnf'):
                        line = line.split()
                        self.num_vars = int(line[2])            
                elif line[0] == 'c':
                    if line.startswith('c p weight'):
                        # If in MC2021 format
                        line = line.split()
                        self.weights[int(line[3])] = float(line[4])
                    if line.startswith('c p auxilliary'):
                        # parse auxiliary variable
                        line = line.split()
                        self.auxiliary.update([int(x) for x in line[3:-1]])
                elif line[0] == 'w':
                        # If in MC2023 format
                        line = line.split()
                        self.weights[int(line[1])] = float(line[2])
                else:
                    line = [int(l) for l in line.split()]
                    self.clauses.append(line[:-1])        

if __name__ == '__main__':
    import sys 
    if len(sys.argv) < 2:
        print("Usage", sys.argv[0], "filename")
        exit(1)
    wcnf = WCNF()
    wcnf.from_file(sys.argv[1])
    print(wcnf)
           