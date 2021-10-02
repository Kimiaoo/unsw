# encoding: utf-8

# @project = workspace_python
# @file = quiz_1
# @author = Hongxiao Jin
# @creat_time = 2019/9/22 19:32

# 19T3 COMP9021 19T3 - Rachid Hamadi
# Quiz 1 *** Due Thursday Week 2


import sys
from random import seed, randrange

try:
    arg_for_seed, upper_bound = (abs(int(x)) + 1 for x in input('Enter two integers: ').split())
except ValueError:
    print('Incorrect input, giving up.')
    sys.exit()

seed(arg_for_seed)
mapping = {}
for i in range(1, upper_bound):
    r = randrange(-upper_bound // 2, upper_bound)
    if r > 0:
        mapping[i] = r
print('\nThe generated mapping is:')
print('  ', mapping)

mapping_as_a_list = []
one_to_one_part_of_mapping = {}
nonkeys = []

# INSERT YOUR CODE HERE
for key in range(1, upper_bound):
    if key not in mapping:
        nonkeys.append(key)

k = 0
for key in range(0, upper_bound):
    if key not in mapping:
        mapping_as_a_list.append(None)
    else:
        k = k + 1
        mapping_as_a_list.append(mapping[key])

keys = list(mapping.keys())
values = list(mapping.values())
for m in range(0, len(keys)):
    for n in range(0, len(values)):
        if (mapping[keys[m]] == values[n]) and (m != n):
            t = 0
            break
        else:
            t = 1
    if t == 1:
        one_to_one_part_of_mapping.setdefault(keys[m], mapping[keys[m]])

print()
print('\nThe mappings\'s so-called "keys" make up a set whose number of elements is', k, '.')
print('\nThe list of integers between 1 and', upper_bound - 1, 'that are not keys of the mapping is:')
print('  ', nonkeys)
print('\nRepresented as a list, the mapping is:')
print('  ', mapping_as_a_list)
# Recreating the dictionary, inserting keys from smallest to largest,
# to make sure the dictionary is printed out with keys from smallest to largest.
one_to_one_part_of_mapping = {key: one_to_one_part_of_mapping[key]
                              for key in sorted(one_to_one_part_of_mapping)
                              }
print('\nThe one-to-one part of the mapping is:')
print('  ', one_to_one_part_of_mapping)