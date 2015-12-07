

import sys


def main():

    f1 = sys.argv[1]
    f2 = sys.argv[2]

    scores_1 = parse(f1)
    scores_2 = parse(f2)

    # eliminate the lines where they agree
    uniq_1 = []
    uniq_2 = []
    for s1,s2 in sorted(zip(scores_1,scores_2), key=lambda t:int(t[0][1].split('/')[1])):
        if s1[1] != s2[1]:
            uniq_1.append(s1[0])
            uniq_2.append(s2[0])

    with open(f1+'.uniq', 'w') as f:
        for line in uniq_1:
            print >>f, line

    with open(f2+'.uniq', 'w') as f:
        for line in uniq_2:
            print >>f, line




def parse(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[4:184]:
            toks = line.strip().split()
            acc  = toks[-1]
            f1   = toks[-2]
            rec  = toks[-3]
            prec = toks[-4]
            data.append( (line.strip(), rec) )
    return data




if __name__ == '__main__':
    main()
