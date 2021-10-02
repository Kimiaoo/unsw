# @file = quiz_4.py
# @author = Hongxiao Jin
# @creat_time = 2019/10/13 20:45

# 19T3 COMP9021 19T3 - Rachid Hamadi
# Quiz 4 *** Due Thursday Week 5
#
# Prompts the user for an arity (a natural number) n and a word.
# Call symbol a word consisting of nothing but alphabetic characters
# and underscores.
# Checks that the word is valid, in that it satisfies the following
# inductive definition:
# - a symbol, with spaces allowed at both ends, is a valid word;
# - a word of the form s(w_1,...,w_n) with s denoting a symbol and
#   w_1, ..., w_n denoting valid words, with spaces allowed at both ends
#   and around parentheses and commas, is a valid word.


import sys


def is_valid(word, arity):
    alpha_num = left_num = right_num = 0
    for i in range(0, len(word)):
        if word[i].isalpha():
            alpha_num = alpha_num + 1
        elif word[i] == '(':
            left_num = left_num + 1
        elif word[i] == ')':
            right_num = right_num + 1
        elif word[i] == '_' or word[i] == ',' or word[i] == ' ':
            continue
        else:
            return False
    if alpha_num == 0:
        return False
    elif arity != 0 and (left_num == 0 or right_num == 0):
        return False
    elif arity == 0 and (left_num != 0 or right_num != 0):
        return False
    elif arity != 0 and (word[-1] != ')' and word[-1] != ' '):
        return False
    elif arity == 0 and (not word[-1].isalpha() and word[-1] != ' '):
        return False

    if '(' in word:
        element = word[:word.find('(')]
        element = element.strip()
        for i in range(0, len(element)):
            if element[i] == ' ':
                return False
    elif arity == 0:
        element = word
        element = element.strip()
        for i in range(0, len(element)):
            if element[i] == ' ':
                return False

    stack = []
    for i in range(0, len(word)):
        if word[i] == '(':
            stack.append('(')
        elif word[i] == ')' and '(' in stack:
            stack.remove(stack[-1])
        elif word[i] == ')' and '(' not in stack:
            return False
    if '(' in stack:
        return False
    else:
        while word.find('(') > 0:
            right_position = word.find(')')
            left_position = word[:right_position].rfind('(')
            element = word[left_position + 1:right_position].split(',')
            if len(element) == arity and ' ' not in element:
                for j in range(0, len(element)):
                    element[j] = element[j].strip()
                    for k in range(0, len(element[j])):
                        if element[j][k] == ' ':
                            return False
                word = word[:left_position] + word[right_position + 1:]
                continue
            else:
                return False
        return True
    # REPLACE THE RETURN STATEMENT ABOVE WITH YOUR CODE


try:
    arity = int(input('Input an arity : '))
    if arity < 0:
        raise ValueError
except ValueError:
    print('Incorrect arity, giving up...')
    sys.exit()
word = input('Input a word: ')
if is_valid(word, arity):
    print('The word is valid.')
else:
    print('The word is invalid.')
