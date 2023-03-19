"""
Conway's Game of Life is a cellular automata simulation that follows simple rules to create interesting patterns. 

The two-dimensional board has a grid of "cells", which follow three simple rules:

- Living cells with two or three neigbors stay alive in the next step of the simulation.

- Dead cells with exactly three neighbors become alive in the next step of the simulation.

- Any other cell dies or stays dead in the next step of the simulation.
"""
 
import copy, random, sys, time
 
# Set up the constants:
WIDTH = 79   # The width of the cell grid.
HEIGHT = 20  # The height of the cell grid.

ALIVE = 'X'
DEAD = ' '

nextCells = {}
for x in range(WIDTH):
    for y in range(HEIGHT):
        if random.randint(0, 1) == 0:
            nextCells[(x, y)] = ALIVE 
        else:
            nextCells[(x, y)] = DEAD

while True:

    print('\n' * 50)
    cells = copy.deepcopy(nextCells)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            print(cells[(x, y)], end='')
        print()

    print('Press Ctrl-C to quit.')

    for x in range(WIDTH):
        for y in range(HEIGHT):

            left = (x - 1) % WIDTH
            right = (x + 1)% WIDTH
            above = (y - 1) % HEIGHT
            below = (y + 1) % HEIGHT

            numNeighbors = 0

            if cells[(left, above)] == ALIVE:
                numNeighbors += 1
            if cells[(x, above)]== ALIVE:
                numNeighbors += 1
            if cells[(right, above)] == ALIVE:
                numNeighbors += 1
            if cells[(left, y)] == ALIVE: 
                numNeighbors += 1
            if cells[(right, y)] == ALIVE:
                numNeighbors += 1
            if cells[(left, below)] == ALIVE:
                numNeighbors += 1
            if cells[(x, below)] == ALIVE:
                numNeighbors += 1
            if cells[(right, below)] == ALIVE:
                numNeighbors += 1

            if cells[(x, y)] == ALIVE and (numNeighbors == 2 or numNeighbors == 3):
                nextCells[(x, y)] = ALIVE
            elif cells[(x, y)] == DEAD and numNeighbors == 3:
                nextCells[(x, y)] = ALIVE 
            else: 
                nextCells[(x, y)] = DEAD 

    try:
        time.sleep(0.1)
    except KeyboardInterrupt:
        print("Conway's Game of Life")
        sys.exit()
