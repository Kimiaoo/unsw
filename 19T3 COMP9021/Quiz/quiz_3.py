# @file = quiz_3.py
# @author = Hongxiao Jin
# @creat_time = 2019/10/3 22:00

# 19T3 COMP9021 19T3 - Rachid Hamadi
# Quiz 3 *** Due Thursday Week 4


# Reading the number written in base 8 from right to left,
# keeping the leading 0's, if any:
# 0: move N     1: move NE    2: move E     3: move SE
# 4: move S     5: move SW    6: move W     7: move NW
#
# We start from a position that is the unique position
# where the switch is on.
#
# Moving to a position switches on to off, off to on there.

import sys

on = '\u26aa'
off = '\u26ab'
code = input('Enter a non-strictly negative integer: ').strip()
try:
    if code[0] == '-':
        raise ValueError
    int(code)
except ValueError:
    print('Incorrect input, giving up.')
    sys.exit()
nb_of_leading_zeroes = 0
for i in range(len(code) - 1):
    if code[i] == '0':
        nb_of_leading_zeroes += 1
    else:
        break
print("Keeping leading 0's, if any, in base 8,", code, 'reads as',
      '0' * nb_of_leading_zeroes + f'{int(code):o}.'
      )
print()

# INSERT YOUR CODE HERE
step = list('0' * nb_of_leading_zeroes + f'{int(code):o}')
step.reverse()
step = list(map(int, step))
order = {}
path = {}
xy_list = [(0, 0)]
x_values = []
y_values = []
order[0] = (0, 1)
order[1] = (1, 1)
order[2] = (1, 0)
order[3] = (1, -1)
order[4] = (0, -1)
order[5] = (-1, -1)
order[6] = (-1, 0)
order[7] = (-1, 1)
x = y = 0
path[(x, y)] = 1
for i in range(0, len(step)):
    x = order[step[i]][0] + x
    y = order[step[i]][1] + y
    if (x, y) not in path:
        path[(x, y)] = 1
    elif (x, y) in path:
        path[(x, y)] = path[(x, y)]+1
x = y = 0
for i in range(0, len(step)):
    x = order[step[i]][0] + x
    y = order[step[i]][1] + y
    if path[(x, y)] % 2 == 1:
        xy_list.append((x, y))
        path[(x, y)] = 1
    elif path[(x, y)] % 2 == 0:
        while (x, y) in xy_list:
            xy_list.remove((x, y))
        path[(x, y)] = 0

for i in range(0, len(xy_list)):
    x_values.append(xy_list[i][0])
    y_values.append(xy_list[i][1])

if x_values and y_values:
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    for j in range(max_y, min_y-1, -1):
        for i in range(min_x, max_x + 1):
            if (i, j) in path and path[(i, j)] != 0:
                print(on, end='')
            else:
                print(off, end='')
        print()