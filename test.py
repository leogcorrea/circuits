def buildCNF(filename):  
    nvars = 0
    ndisj = 0

    cnf = []
    clause = []

    with open(filename) as f:
        for line in f:
            if line[0] == 'c':
                continue
            if nvars == 0:
                header =  line.split()
                if header[0] != 'p' or header[1] != 'cnf':
                   return
                nvars = int(header[2])
                ndisj = int(header[3])
                continue

            #line = line.strip()
            disj = line.split()

            for item in disj:
                if item == '0':
                    cnf.append(clause)
                    clause = []         
                else:
                    clause.append(int(item))
            
    if ndisj != len(cnf):
        raise IndexError("Number of clauses found in CNF differ from the header!")
    
    return cnf, nvars

def evaluateCNF(cnf, input):
    for disj in cnf:
        result = 0
        for item in disj:
            literal = abs(item)
            value = input[literal-1]
            if item < 0:
                value = not value
            result += value
        if not result:
            return False
    return True


if __name__ == '__main__':
    filename = 'simple_w_constraint_opt.cnf'
    #filename = 'montyhall_w_constraint_pre.cnf'
    cnf, nvars  = buildCNF(filename)

    #input = [1]*nvars
    #result = evaluateCNF(cnf, input)

    nconf = 2 ** nvars
    
    for conf in range(nconf):
        bconf = '{0:b}'.format(conf).zfill(nvars)
        input = []

        for b in bconf:
            input.append(int(b))

        result = evaluateCNF(cnf, input)

        if result:
            print("Input: {0}, Output: {1}", input, result)
    