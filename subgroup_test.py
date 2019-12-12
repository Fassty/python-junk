#!/usr/bin/env python
from sys import argv
from itertools import chain, combinations

def powerset(group):
    return chain.from_iterable(combinations(group, r) for r in range(len(group) + 1))

def GCD(x, y):
    x, y = (x,y) if x > y else (y, x)
    a0, a1 = x, y
    x0, x1 = 1, 0
    y0, y1 = 0, 1

    while(a1 > 0):
        (q, a1), a0 = divmod(a0,a1), a1
        x1, x0 = x0 - x1 * q, x1
        y1, y0 = y0 - y1 * q, y1

    return a0

def create_group(p):
    group = []
    for i in range(p):
        if(GCD(i, p) == 1): group.append(i)
    return group

if __name__ == "__main__":
    p = int(argv[1])
    group = create_group(p)
    ps = list(powerset(group))
    groups = []

    for subs in ps:
        if subs != () and 1 in subs:
            subgr = True
            for (x, y) in combinations(subs, 2):
                elem = (x * y) % p
                if x != y and not elem in subs:
                    subgr = False
                    break
            if subgr:
                groups.append(subs)
                print(subs)
