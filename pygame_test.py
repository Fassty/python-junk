#!/usr/bin/env python
import pygame
import numpy as np
from pygame.locals import *

BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (0,0,255)
RED = (255,0,0)
GREEN = (0,255,0)
count = 100
width = 1000 / count
DISPLAY = None
class BubbleSort:
    def __init__(self, DISPLAY):
        self.DISPLAY = DISPLAY
        self.sorted = False

    def sort(self, arr):
        while not self.sorted:
            self.sorted = True
            for i in range(len(arr) - 1):
                if arr[i] > arr[i +1 ]:
                    arr[i], arr[i+1] = arr[i+1], arr[i]
                    self.draw(arr)
                    self.sorted = False

    def draw(self, arr):
        self.DISPLAY.fill(WHITE)
        for i, elem in enumerate(arr):
            pygame.draw.rect(self.DISPLAY, BLACK, (width*i, 0, width, 600*elem))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()

class InsertSort:
    def __init__(self):
        self.end = 0

    def step(self, arr):
        if self.end >= len(arr) - 1: return 0,0,arr
        j = self.end + 1
        tmp = arr[j]
        while j > 0 and tmp > arr[j-1]:
            arr[j] = arr[j - 1]
            j -= 1
        arr[j] = tmp
        self.end += 1
        return j, self.end, arr

class SelectSort:
    def __init__(self):
        self.end = None

    def step(self, arr):
        if self.end != None and -self.end >= len(arr): return 0,0,arr
        max_i = np.where(arr[0 : self.end] == np.amax(arr[0 : self.end]))[0][0]
        if self.end == None:
            self.end = 0
        self.end -= 1
        arr[max_i], arr[self.end] = arr[self.end], arr[max_i]
        return max_i, self.end, arr

class MergeSort:
    def __init__(self, DISPLAY):
        self.DISPLAY = DISPLAY

    def merge_sort(self,arr):
        unit = 1
        while unit <= len(arr):
            for h in range(0, len(arr), unit * 2):
                l,r = h, min(len(arr), h + 2 * unit)
                mid = h + unit
                p,q = l, mid
                while p < mid and q < r:
                    if arr[p] <= arr[q]: p += 1
                    else:
                        tmp = arr[q]
                        arr[p + 1 : q + 1] = arr[p : q]
                        arr[p] = tmp
                        p, mid, q = p+1, mid + 1, q + 1
            self.draw(arr)
            unit *= 2
        return arr

    def draw(self,arr):
        self.DISPLAY.fill(WHITE)
        for i, elem in enumerate(arr):
            pygame.draw.rect(self.DISPLAY, BLACK, (width*i, 0, width, 600*elem))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
        pygame.time.delay(100)

def main():
    pygame.init()
    DISPLAY = pygame.display.set_mode((1000,600), 0, 32)
    arr = np.random.rand(1,count)[0]

    ms = BubbleSort(DISPLAY)
    ms.sort(arr)

if __name__ == "__main__":
    main()
