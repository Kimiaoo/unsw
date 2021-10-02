# @file = quiz_2.py
# @author = Hongxiao Jin
# @creat_time = 2019/9/25 19:46

# 19T3 COMP9021 19T3 - Rachid Hamadi
# Quiz 2 *** Due Thursday Week 3


import sys
from itertools import cycle
from random import seed, randrange
from pprint import pprint

try:
    arg_for_seed, upper_bound = (abs(int(x)) + 1 for x in input('Enter two integers: ').split())
except ValueError:
    print('Incorrect input, giving up.')
    sys.exit()

seed(arg_for_seed)
mapping = {}
for i in range(1, upper_bound):
    r = randrange(-upper_bound // 8, upper_bound)
    if r > 0:
        mapping[i] = r
print('\nThe generated mapping is:')
print('  ', mapping)
# sorted() can take as argument a list, a dictionary, a set...
keys = sorted(mapping.keys())
print('\nThe keys are, from smallest to largest: ')
print('  ', keys)

cycles = []
reversed_dict_per_length = {}

# INSERT YOUR CODE HERE
cycle = []
dic = {}
standard = []
dic = mapping.copy()
for k in mapping:
    if k == dic[k]:
        cycle.append(k)
        cycles.append(cycle.copy())
        cycle.clear()
        dic.pop(k)

def find_cycles(key):
    if dic[key] in dic and dic[key] not in standard and key not in cycle:
        cycle.append(key)
        find_cycles(dic[key])
    elif i == dic[key]:
        cycle.append(key)
        cycles.append(cycle.copy())
        cycle.clear()
        cycles.sort()
    else:
        cycle.clear()

for i in dic:
    if dic[i] in dic:
        standard.append(i)
        find_cycles(i)
    else:
        continue

reversed_dict = {}
reversed_mapping1 = {}
reversed_mapping2 = {}
mappingList = []
nums = []
keys = list(mapping.values())
values = list(mapping.keys())

for i in range(0, len(keys)):
    if keys.count(keys[i]) not in nums:
        nums.append(keys.count(keys[i]))
nums = sorted(nums)

for j in range(0, len(nums)):
    reversed_mapping1.clear()
    for i in range(0, len(keys)):
        num = keys.count(keys[i])
        if num == nums[j]:
            if num <= 1:
                mappingList.append(values[i])
                mappingList.sort()
                reversed_mapping1[keys[i]] = mappingList.copy()
                reversed_mapping2 = reversed_mapping1.copy()
            else:
                mappingList.append(values[i])
                for a in range(0, len(keys)):
                    if (keys[a] == keys[i]) and (i != a):
                        mappingList.append(values[a])
                        mappingList.sort()
                        reversed_mapping1[keys[i]] = mappingList.copy()
                        reversed_mapping2 = reversed_mapping1.copy()
            mappingList.clear()
            reversed_dict_per_length[num] = reversed_mapping2
        else:
            continue

print('\nProperly ordered, the cycles given by the mapping are: ')
print('  ', cycles)
print('\nThe (triply ordered) reversed dictionary per lengths is: ')
pprint(reversed_dict_per_length)
