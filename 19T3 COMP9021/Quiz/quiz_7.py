# @file = quiz_7.py
# @author = Hongxiao Jin
# @creat_time = 9/11/2019 12:03 pm

# 19T3 COMP9021 19T3 - Rachid Hamadi
# Quiz 7 *** Due Thursday Week 9
#
# Randomly generates a grid of 0s and 1s and determines
# the maximum number of "spikes" in a shape.
# A shape is made up of 1s connected horizontally or vertically (it can contain holes).
# A "spike" in a shape is a 1 that is part of this shape and "sticks out"
# (has exactly one neighbour in the shape).


from random import seed, randrange
import sys

dim = 10


def display_grid():
    for row in grid:
        print('   ', *row)


def find_way(cur_x, cur_y, col, dir):
    grid[cur_x][cur_y] = col
    for (dir_x, dir_y) in dir:
        next_x = cur_x + dir_x
        next_y = cur_y + dir_y
        if 0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]):
            if grid[next_x][next_y] == 1:
                find_way(next_x, next_y, col, dir)


# Returns the number of shapes we have discovered and "coloured".
# We "colour" the first shape we find by replacing all the 1s
# that make it with 2. We "colour" the second shape we find by
# replacing all the 1s that make it with 3.
def colour_shapes():
    colour = 2
    direction = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    # look for the part composed by 1, and colour this part with 2,3...
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                find_way(i, j, colour, direction)
                colour += 1
    return grid
    # Replace pass above with your code


def find_max_colour(shape):
    # get the max colour in coloured grid
    max_col = 2
    for i in range(len(shape)):
        for j in range(len(shape[0])):
            if shape[i][j] > max_col:
                max_col = shape[i][j]
    return max_col


def find_max_part(coloured_shape):
    max_colour = find_max_colour(coloured_shape)
    max_size = 0
    most_col = []
    for col in range(2, max_colour + 1):
        count_size = 0
        for i in range(len(coloured_shape)):
            for j in range(len(coloured_shape[0])):
                if coloured_shape[i][j] == col:
                    count_size += 1
        if count_size >= max_size:
            max_size = count_size
            most_col.append(col)
    return most_col


def check_around(shape, col):
    direction = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    spikes = 0
    for cur_x in range(len(shape)):
        for cur_y in range(len(shape[0])):
            if shape[cur_x][cur_y] == col:
                count_col = 0
                for (dir_x, dir_y) in direction:
                    around_x = cur_x + dir_x
                    around_y = cur_y + dir_y
                    if 0 <= around_x < len(shape) and 0 <= around_y < len(shape[0]):
                        if shape[around_x][around_y] == col:
                            count_col += 1
                if count_col == 1:
                    spikes += 1
    return spikes


def max_number_of_spikes(nb_of_shapes):
    max_part = find_max_part(nb_of_shapes)
    result = 0
    for i in range(len(max_part)):
        res = check_around(nb_of_shapes, max_part[i])
        if res > result:
            result = res
    return result
    # Replace pass above with your code


# Possibly define other functions here


try:
    for_seed, density = (int(x) for x in input('Enter two integers, the second '
                                               'one being strictly positive: '
                                               ).split()
                         )
    if density <= 0:
        raise ValueError
except ValueError:
    print('Incorrect input, giving up.')
    sys.exit()

seed(for_seed)
grid = [[int(randrange(density) != 0) for _ in range(dim)]
        for _ in range(dim)
        ]
print('Here is the grid that has been generated:')
display_grid()
nb_of_shapes = colour_shapes()
print('The maximum number of spikes of some shape is:',
      max_number_of_spikes(nb_of_shapes)
      )
