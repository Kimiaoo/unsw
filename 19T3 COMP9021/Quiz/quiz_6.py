# @file = quiz_6.py
# @author = Hongxiao Jin
# @creat_time = 3/11/2019 6:07 pm

# 19T3 COMP9021 19T3 - Rachid Hamadi
# Quiz 6 *** Due Thursday Week 8
#
# Randomly fills an array of size 10x10 with 0s and 1s, and outputs the size of
# the largest parallelogram with horizontal sides.
# A parallelogram consists of a line with at least 2 consecutive 1s,
# with below at least one line with the same number of consecutive 1s,
# all those lines being aligned vertically in which case the parallelogram
# is actually a rectangle, e.g.
#      111
#      111
#      111
#      111
# or consecutive lines move to the left by one position, e.g.
#      111
#     111
#    111
#   111
# or consecutive lines move to the right by one position, e.g.
#      111
#       111
#        111
#         111


from random import seed, randrange
import sys

dim = 10


def display_grid():
    for row in grid:
        print('   ', *row)


def find_largest_shape(grid_string, shape):
    possible_size = []
    for current_index in range(len(grid_string)):
        temp_example = ''
        current_line = int(current_index / 10)
        # A parallelogram consists of a line with at least 2 consecutive 1s
        if current_line <= 8 and current_index % 10 <= 8:
            if grid_string[current_index] == '1':
                for i in range(current_index, (current_line + 1) * 10):
                    if grid_string[i] == '0':
                        flag_index = i
                        break
                    else:
                        flag_index = (current_line + 1) * 10
                if flag_index - current_index >= 2:
                    for temp_flag in range(current_index + 2, flag_index + 1):
                        temp_example = ''
                        for i in range(current_index, temp_flag):
                            temp_example += grid_string[i]
                            if len(temp_example) > 1:
                                temp_index = current_index
                                temp_flag_index = temp_flag
                                for line in range(current_line + 1, 10):
                                    if shape == 'rectangle':
                                        temp_index += 10
                                        temp_flag_index += 10
                                    elif shape == 'left':
                                        range_left = int((temp_index + 10) / 10) * 10
                                        range_right = int((temp_index + 10) / 10) * 10 + 9
                                        temp = temp_index + 9
                                        if range_left <= temp <= range_right:
                                            temp_index += 9
                                            temp_flag_index += 9
                                        else:
                                            continue
                                    elif shape == 'right':
                                        range_left = int((temp_index + 10) / 10) * 10
                                        range_right = int((temp_index + 10) / 10) * 10 + 9
                                        temp_i = temp_index + 11
                                        temp_f = temp_flag_index + 11
                                        if range_left <= temp_i <= range_right and range_left <= temp_f <= range_right + 1:
                                            temp_index += 11
                                            temp_flag_index += 11
                                        else:
                                            continue
                                    if temp_example == grid_string[temp_index:temp_flag_index]:
                                        possible_size.append(
                                            (temp_flag_index - temp_index) * (line - current_line + 1))
                                    else:
                                        break
    if possible_size:
        return max(possible_size)
    else:
        return 0


def size_of_largest_parallelogram():
    # turn grid to string
    grid_string = ''
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            grid_string += ''.join(str(grid[i][j]))

    # call function find_largest_parallelogram()
    largest_rectangle = find_largest_shape(grid_string, 'rectangle')
    largest_left = find_largest_shape(grid_string, 'left')
    largest_right = find_largest_shape(grid_string, 'right')

    # get the largest size
    largest_size = max(0, largest_rectangle, largest_left, largest_right)

    return largest_size

    # REPLACE PASS ABOVE WITH YOUR CODE


# POSSIBLY DEFINE OTHER FUNCTIONS


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
size = size_of_largest_parallelogram()
if size:
    print('The largest parallelogram with horizontal sides '
          f'has a size of {size}.'
          )
else:
    print('There is no parallelogram with horizontal sides.')
