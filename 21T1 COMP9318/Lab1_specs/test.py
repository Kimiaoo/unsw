import math

import submission as submission
import re


print(submission.nsqrt(0), submission.nsqrt(1))


def f(x):
    return x * math.log(x) - 16.0


def fprime(x):
    return 1.0 + math.log(x)


x = submission.find_root(f, fprime)
print(x)
print(f(x))


def print_tree(root, indent=0):
    print(' ' * indent, root)
    if len(root.children) > 0:
        for child in root.children:
            print_tree(child, indent + 4)


def myfind(s, char):
    pos = s.find(char)
    if pos == -1:  # not found
        return len(s) + 1
    else:
        return pos


def next_tok(s):  # returns tok, rest_s
    if s == '':
        return (None, None)
    # normal cases
    poss = [myfind(s, ' '), myfind(s, '['), myfind(s, ']')]
    min_pos = min(poss)
    if poss[0] == min_pos:  # separator is a space
        tok, rest_s = s[: min_pos], s[min_pos + 1:]  # skip the space
        if tok == '':  # more than 1 space
            return next_tok(rest_s)
        else:
            return (tok, rest_s)
    else:  # separator is a [ or ]
        tok, rest_s = s[: min_pos], s[min_pos:]
        if tok == '':  # the next char is [ or ]
            return (rest_s[:1], rest_s[1:])
        else:
            return (tok, rest_s)


def str_to_tokens(str_tree):
    # remove \n first
    str_tree = str_tree.replace('\n', '')
    out = []

    tok, s = next_tok(str_tree)
    while tok is not None:
        out.append(tok)
        tok, s = next_tok(s)
    return out


# str_tree = '''
# 1 [2 [3 4       5          ]
#    6 [7 8 [9]   10 [11 12] ]
#    13
#   ]
# '''
# str_tree = '''
# * [1 2 + [3 4]]
# '''
# str_tree = '''
# 1 [1 [1 1 [1 1 1] 1 [1 1]] 1]
# '''
# str_tree = '''
# 1 [2 [3 4 [5 6 7] 8 [9 10]] 11 [12 [13 [14 [15 [16 17[18 19] 20] 21] 22 ] 23] 24] 25]
# '''
str_tree =  '''
1 [2 [3 4  [14 [15 [16]]]     5          ]
   6 [7 8 [9]   10 [11 12] ]
   13
  ]
'''
toks = str_to_tokens(str_tree)
# print('------------------------------------ori tokens------------------------------------')
print(toks)
tt = submission.make_tree(toks)
print("--------------------------------------tree--------------------------------------")
print_tree(tt)

depth = submission.max_depth(tt)
print(depth)

