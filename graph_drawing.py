#!/usr/bin/env python
import numpy as np
import pygame
import sys
from pygame.locals import *

class Node:
    def __init__(self, x=200, y=200, target=[]):
        super().__init__()
        self.x = x
        self.y = y
        self.weight = 10
        self.target = target

    def add_edge(self, target):
        self.target.extend(target)

def dist(u, v):
    return np.sqrt((v.x - u.x)**2 + (v.y - u.y)**2)

class Graph:
    def __init__(self, nodes=[]):
        self.nodes = nodes

    def draw(self, DISPLAY):
        DISPLAY.fill((255,255,255))
        for node in self.nodes:
            pygame.draw.circle(DISPLAY, (0,0,0), (node.x, node.y), 8)
            for node in self.nodes:
                for neigh in node.target:
                    if node != neigh:
                        pygame.draw.line(DISPLAY, (0,0,0), (node.x, node.y), (neigh.x, neigh.y))
        pygame.display.update()
        pygame.time.delay(100)

    def update_nodes(self):
        for node in self.nodes:
            direction = [0,0]
            for neigh in node.target:
                if node != neigh:
                    d = dist(node, neigh)
                    print(d)
                    if d <= 0: direction += [1,1]
                    else:
                        direction[0] += (node.x - neigh.x) * 1/(2*d)
                        direction[1] += (node.y - neigh.y) * 1/(2*d)
            node.x += int(direction[0])
            node.y += int(direction[1])

    def update(self, DISPLAY):
        while True:
            self.draw(DISPLAY)
            self.update_nodes()
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()


def create_graph():
    node1 = Node(x=150)
    node2 = Node(y=150)
    node3 = Node()

    node1.add_edge([node2, node3])
    node2.add_edge([node1, node3])
    node3.add_edge([node1, node2])

    nodes = [node1, node2, node3]

    graph = Graph(nodes=nodes)
    return graph

def main():
    pygame.init()
    graph = create_graph()
    DISPLAY = pygame.display.set_mode((400,400), 0, 32)
    graph.update(DISPLAY)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.update()

if __name__ == "__main__":
    main()
