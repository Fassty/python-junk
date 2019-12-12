#!/usr/bin/env python3.8

class SelfTest:
    def __init__(self, **kwargs):
        print(kwargs)


SelfTest(a=1, b=2)
