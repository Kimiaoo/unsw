# @file = quiz_5.py
# @author = Hongxiao Jin
# @creat_time = 2019/10/17 18:45

# 19T3 COMP9021 19T3 - Rachid Hamadi
# Quiz 5 *** Due Thursday Week 7
#
# Implements a function that, based on the encoding of
# a single strictly positive integer that in base 2,
# reads as b_1 ... b_n, as b_1b_1 ... b_nb_n, encodes
# a sequence of strictly positive integers N_1 ... N_k
# with k >= 1 as N_1* 0 ... 0 N_k* where for all 0 < i <= k,
# N_i* is the encoding of N_i.
#
# Implements a function to decode a positive integer N
# into a sequence of (one or more) strictly positive
# integers according to the previous encoding scheme,
# or return None in case N does not encode such a sequence.


import sys


def encode(list_of_integers):
    # save code (type is list)
    code_list = []
    # save encode (type is list)
    encode_list = []
    # get list of code
    for i in range(len(list_of_integers)):
        code_list.append(bin(list_of_integers[i])[2:])
    # to double the digital
    for i in range(len(code_list)):
        for j in range(len(code_list[i])):
            encode_list.append(code_list[i][j])
            encode_list.append(code_list[i][j])
        # add '0' between elements and in case extra '0' in the end
        if i != len(list_of_integers) - 1:
            encode_list.append('0')
    # turn list into string
    encode_string = ''.join(encode_list)
    # turn 0b to int
    if encode_string:
        return int(encode_string, 2)
    # REPLACE pass ABOVE WITH YOUR CODE


def decode(integer):
    # save code as list in 0b
    code_list = list(bin(the_input)[2:])
    # save decode (type is list)
    decode_list_0b = []
    decode_list = []

    # if the num of 1 is odd, it must be false
    if code_list.count('1') % 2 != 0:
        return None

    # if 0 is at the end of the integer, the num of 0 before the last 1 must be even
    code_list.reverse()
    for i in range(len(code_list)):
        if code_list[i] == '1':
            flag = i
            break

    if code_list[:flag].count('0') % 2 != 0:
        return None

    code_list.reverse()

    for i in range(0, len(code_list) - 1, 1):
        if code_list[i] == code_list[i + 1]:
            code_list[i + 1] = '2'
            decode_list_0b.append(code_list[i])
        elif code_list[i] == '1' and code_list[i + 1] == '0':
            return None
        elif code_list[i] == '0' and code_list[i + 1] == '1':
            decode_list_0b.append(',')
            continue
    # split different num
    decode_list_0b = ''.join(decode_list_0b).split(',')
    # turn int to 0b
    if decode_list_0b != ['']:
        for i in range(len(decode_list_0b)):
            deco = int(decode_list_0b[i], 2)
            decode_list.append(deco)
        return decode_list
    # REPLACE pass ABOVE WITH YOUR CODE


# We assume that user input is valid. No need to check
# for validity, nor to take action in case it is invalid.
print('Input either a strictly positive integer')
the_input = eval(input('or a nonempty list of strictly positive integers: '))
if type(the_input) is int:
    print('  In base 2,', the_input, 'reads as', bin(the_input)[2:])
    decoding = decode(the_input)
    if decoding is None:
        print('Incorrect encoding!')
    else:
        print('  It encodes: ', decode(the_input))
else:
    print('  In base 2,', the_input, 'reads as',
          f'[{", ".join(bin(e)[2:] for e in the_input)}]'
          )
    print('  It is encoded by', encode(the_input))
