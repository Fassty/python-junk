#!/usr/bin/env python
from sys import stdin as _in
import argparse

def main(args):
    # Swap if x > y
    x, y = (args.x, args.y) if args.x > args.y else (args.y, args.x)

    # Init a, x, y
    a0, a1 = x, y
    x0, x1 = 1, 0
    y0, y1 = 0, 1

    # Alg loop
    while(a1 > 0):
        (q, a1), a0 = divmod(a0,a1), a1
        x1, x0 = x0 - x1 * q, x1
        y1, y0 = y0 - y1 * q, y1

        print(f" AO:{a0} A1:{a1}\n", f"X0:{x0} X1:{x1}\n", f"YO:{y0} Y1:{y1}\n", f"Q:{q}")
    # Return GCD(x,y), a, b where GCD(x, y) = a * x + b * y
    return (a0, x0, y0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, help="X input")
    parser.add_argument("--y", type=int, help="Y input")
    args = parser.parse_args()
    print(main(args))
