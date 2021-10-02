# @file = roman_arabic.py
# @author = Hongxiao Jin
# @creat_time = 2019/10/5 15:04

import sys


def match_requirement(request):
    if len(request) == 3:
        if request[0] == "Please" and request[1] == "convert":
            return 1
    if len(request) == 5:
        if request[0] == "Please" and request[1] == "convert" and request[3] == "using":
            return 2
    if len(request) == 4:
        if request[0] == "Please" and request[1] == "convert" and request[3] == "minimally":
            return 3


# transfer arabic to roman For First kind of input.
def transfer_arabic_to_roman(arabic):
    res = ''
    roman_dict = {'M': 1000, 'CM': 900, 'D': 500, 'CD': 400, 'C': 100, 'XC': 90, 'L': 50, 'XL': 40, 'X': 10, 'IX': 9,
                  'V': 5, 'IV': 4, 'I': 1}
    for char in roman_dict:
        while arabic >= roman_dict[char]:
            arabic -= roman_dict[char]
            res += char
    return res


# transfer roman to arabic For First kind of input.
def transfer_roman_to_arabic(roman):
    res = 0
    roman_dict = {'M': 1000, 'CM': 900, 'D': 500, 'CD': 400, 'C': 100, 'XC': 90, 'L': 50, 'XL': 40, 'X': 10, 'IX': 9,
                  'V': 5, 'IV': 4, 'I': 1}
    for i in range(len(roman)):
        if roman[i] in roman_dict:
            if i + 1 < len(roman) and roman_dict[roman[i]] < roman_dict[roman[i + 1]]:
                res -= roman_dict[roman[i]]
            else:
                res += roman_dict[roman[i]]
    # check if the roman is valid
    if transfer_arabic_to_roman(res) == roman:
        return res
    else:
        return None


# create a generalised roman format
def create_roman_format(roman_symbol):
    roman_symbol_list = list(roman_symbol)
    roman_symbol_list.reverse()
    digit = 1
    roman_format = {}
    for i in range(0, len(roman_symbol_list), 2):
        roman_format[roman_symbol_list[i]] = digit
        if i + 1 < len(roman_symbol_list):
            roman_format[roman_symbol_list[i] + roman_symbol_list[i + 1]] = digit * 4  # the char like IV
            roman_format[roman_symbol_list[i + 1]] = digit * 5  # the char like V
        if i + 2 < len(roman_symbol_list):
            roman_format[roman_symbol_list[i] + roman_symbol_list[i + 2]] = digit * 9  # the char like IX
            roman_format[roman_symbol_list[i + 2]] = digit * 10  # the char like X
        digit *= 10
    return roman_format


# transfer arabic to roman For Second kind of input.
def transfer_arabic_to_generalised_roman(arabic, roman_symbol):
    res = ''
    input_string = str(arabic)
    roman_list = []
    arabic_list = []
    roman_dict = create_roman_format(list(roman_symbol))
    for char in roman_dict:
        roman_list.append(char)
        arabic_list.append(roman_dict[char])
    roman_list.reverse()
    arabic_list.reverse()
    # the number end with 9 must be 10 or other similar num - 1
    input_list = list(input_string)
    input_list.reverse()
    for i in range(len(input_list)):
        if input_list[i] == '9':
            if 10 ** (i + 1) not in arabic_list:
                return None
    for i in range(len(roman_list)):
        while arabic >= arabic_list[i]:
            arabic -= arabic_list[i]
            res += roman_list[i]
    # one symbol can not repeat more than 3 times
    for i in range(0, len(res)):
        check_repeat = res[i] + res[i] + res[i] + res[i]
        if check_repeat in res:
            return None
    return res


# transfer roman to arabic For Second kind of input.
def transfer_generalised_roman_to_arabic(roman, roman_symbol):
    res = 0
    roman_dict = create_roman_format(roman_symbol)
    # if the symbol starts with 5, then it can not double or more
    for i in range(0, len(roman)):
        if roman[i] in roman_dict:
            if str(roman_dict[roman[i]])[0] == '5':
                if i + 1 < len(roman) and roman[i] == roman[i + 1]:
                    return None
    # one symbol can not repeat more than 3 times
    for i in range(0, len(roman)):
        check_repeat = roman[i] + roman[i] + roman[i] + roman[i]
        if check_repeat in roman:
            return None
    # calculate the arabic
    for i in range(0, len(roman)):
        if roman[i] in roman_dict:
            if i + 1 < len(roman) and roman_dict[roman[i]] < roman_dict[roman[i + 1]]:
                res -= roman_dict[roman[i]]  # if the char is on the left and less than i+1, minus it
            else:
                res += roman_dict[roman[i]]
    # check if the roman is valid: such as VV
    if transfer_arabic_to_generalised_roman(res, roman_symbol) == roman:
        return res
    else:
        return None


def get_roman_format(roman_string):
    diff_char = []
    least_part_roman_ident = ''
    least_part_roman = ''
    ROMAN_ARABIC = {'I': 1, 'V': 5, 'X': 10, 'L': 50}
    ARABIC_ROMAN = {1: 'I', 5: 'V', 10: 'X', 50: 'L'}
    basis_arabic_roman = {}
    basis_arabic_count = {}

    position_count_dict = {}

    first_flag = {}
    last_flag = {}

    roman_format_list = []
    roman_format_string = ''

    # make a basis dict which includes 1-99 roman:arabic
    # count every basis char in basis roman arabic:count
    for i in range(1, 100):
        transferred_roman = transfer_arabic_to_roman(i)
        basis_arabic_roman[i] = transferred_roman
        ident = ''
        for j in range(len(transferred_roman)):
            ident = ident + str(transferred_roman.count(transferred_roman[j]))
        basis_arabic_count[i] = ident
    basis_count_list = list(basis_arabic_count.values())

    # one symbol can not repeat more than 3 times continuously
    for i in range(len(roman_string)):
        check_repeat = roman_string[i] + roman_string[i] + roman_string[i] + roman_string[i]
        if check_repeat in roman_string:
            return None

    # count different char in roman_string
    for i in range(len(roman_string)):
        if roman_string[i] not in diff_char:
            diff_char.append(roman_string[i])

    # get the position of each appear times
    for i in range(len(diff_char)):
        for j in range(len(roman_string)):
            if roman_string[j] == diff_char[i]:
                position_count_dict[j] = (diff_char[i], roman_string.count(diff_char[i]))

    # put the appear times into a list
    position_count_list = list(position_count_dict.values())

    for i in range(len(roman_string)):
        if roman_string[i] not in first_flag:
            first_flag[roman_string[i]] = i

    for i in range(-1, -len(roman_string) - 1, -1):
        if roman_string[i] not in last_flag:
            last_flag[roman_string[i]] = i + len(roman_string)

    # match the basic char with input char
    i = -1
    while i > -len(roman_string) - 1:
        char = roman_string[i]
        f_flag = first_flag[char]
        l_flag = last_flag[char]
        count_match = ''
        roman_match = ''
        if f_flag != l_flag:
            for j in range(f_flag, l_flag + 1):
                count_match += str(position_count_dict[j][1])
                roman_match += position_count_dict[j][0]

            if count_match in basis_count_list:
                least_part_roman_ident = count_match + ' ' + least_part_roman_ident
                least_part_roman = roman_match + ' ' + least_part_roman
                i = (f_flag - len(roman_string)) - 1
                continue
            elif count_match not in basis_count_list:
                in_part_time = 0
                out_part_time = 0

                in_part_time_index = []
                out_part_time_index = []

                for k in range(0, f_flag):
                    if roman_string[k] in roman_match:
                        out_part_time = roman_string[0:f_flag].count(roman_string[k])
                        in_part_time = position_count_dict[k][1] - out_part_time
                        out_part_time_index.append(k)
                    for m in range(f_flag, l_flag + 1):
                        if roman_string[m] == roman_string[k]:
                            in_part_time_index.append(m)

                for k in range(len(in_part_time_index)):
                    position_count_dict[in_part_time_index[k]] = (roman_string[in_part_time_index[k]], in_part_time)

                count_match = ''
                roman_match = ''
                for j in range(f_flag, l_flag + 1):
                    count_match += str(position_count_dict[j][1])
                    roman_match += position_count_dict[j][0]

                if count_match in basis_count_list:
                    least_part_roman_ident = count_match + ' ' + least_part_roman_ident
                    least_part_roman = roman_match + ' ' + least_part_roman
                    i = (f_flag - len(roman_string)) - 1
                    for k in range(len(out_part_time_index)):
                        position_count_dict[out_part_time_index[k]] = (
                            roman_string[out_part_time_index[k]], out_part_time)
                        last_flag[roman_string[out_part_time_index[k]]] = out_part_time_index[-1]
                else:
                    return None
        else:

            least_part_roman_ident = str(position_count_dict[f_flag][1]) + ' ' + least_part_roman_ident
            least_part_roman = position_count_dict[f_flag][0] + ' ' + least_part_roman
            i = (f_flag - len(roman_string)) - 1
            continue

    least_part_roman_ident_list = least_part_roman_ident.split()
    least_part_roman_list = least_part_roman.split()

    while least_part_roman_ident_list:
        least_part_roman_ident = least_part_roman_ident_list[-1]
        least_part_roman = least_part_roman_list[-1]
        finished_ident = []
        finished_char = []
        mix_to_basis = {}
        finished_ident.append(least_part_roman_ident_list[-1])
        finished_char.append(least_part_roman_list[-1])

        for i in range(-2, -len(least_part_roman_list) - 1, -1):
            temp_ident = least_part_roman_ident
            temp_roman = least_part_roman
            least_part_roman_ident = least_part_roman_ident_list[i] + least_part_roman_ident
            least_part_roman = least_part_roman_list[i] + least_part_roman
            if least_part_roman_ident in basis_count_list:
                finished_ident.append(least_part_roman_ident_list[i])
                finished_char.append(least_part_roman_list[i])
                continue
            else:
                least_part_roman_ident = temp_ident
                least_part_roman = temp_roman
                break

        for ara in basis_arabic_count:
            if basis_arabic_count[ara] == least_part_roman_ident:
                for i in range(len(least_part_roman_ident)):
                    mix_to_basis[ROMAN_ARABIC[basis_arabic_roman[ara][i]]] = least_part_roman[i]
                break

        # ROMAN_ARABIC = {'I': 1, 'V': 5, 'X': 10, 'L': 50}
        # ARABIC_ROMAN = {1: 'I', 5: 'V', 10: 'X', 50: 'L'}

        for ara in ARABIC_ROMAN:
            if ara in mix_to_basis:
                roman_format_list.insert(0, mix_to_basis[ara])
            else:
                roman_format_list.insert(0, '_')

        flag = 0
        for i in range(len(roman_format_list)):
            if roman_format_list[i].isalpha():
                break
            else:
                flag += 1

        if flag > 1:
            if flag == 2:
                for i in range(0, flag):
                    roman_format_list.remove('_')
            else:
                for i in range(0, flag - 1):
                    roman_format_list.remove('_')

        least_part_roman_ident_list.reverse()
        least_part_roman_list.reverse()
        for i in range(len(finished_ident)):
            least_part_roman_ident_list.remove(finished_ident[i])
            least_part_roman_list.remove(finished_char[i])

        least_part_roman_ident_list.reverse()
        least_part_roman_list.reverse()

    for i in range(-1, -len(roman_format_list) - 1, -1):
        if roman_format_list.count(roman_format_list[i]) > 1:
            roman_format_list[i] = '_'

    roman_format_string = ''.join(roman_format_list)
    roman_format_string = roman_format_string.strip('_')

    return roman_format_string


requirement = input('How can I help you? ').strip()

try:
    require_split = requirement.split(' ')  # split requirement with blank to get the requirement
    n = match_requirement(require_split)
    result = arabic_result = format_result = None
    if n is None or not requirement.startswith('Please convert '):
        raise NotImplementedError

    # First kind of input
    if n == 1:
        if require_split[2].isdigit():  # check the requirement is arabic to roman
            if require_split[2][0] == '0':  # if the integer starts with 0, it is invalid
                raise ValueError
            else:
                require_integer = int(require_split[2])
                if require_integer < 1 or require_integer > 3999:  # the integer is at most equal to 3999
                    raise ValueError
                else:
                    result = transfer_arabic_to_roman(require_integer)
        elif require_split[2].isalpha():  # check the requirement is roman to arabic
            result = transfer_roman_to_arabic(require_split[2])
        # output for first
        if result is not None:
            print('Sure! It is', result)
        else:
            raise ValueError

    # Second kind of input
    if n == 2:
        if require_split[2].isdigit():  # check the requirement is arabic to roman
            if require_split[2][0] == '0':  # if the integer starts with 0, it is invalid
                raise ValueError
            else:
                require_integer = int(require_split[2])
                result = transfer_arabic_to_generalised_roman(require_integer, require_split[4])
        elif require_split[2].isalpha():  # Second kind of input.
            result = transfer_generalised_roman_to_arabic(require_split[2], require_split[4])
        # output for second
        if result is not None:
            print('Sure! It is', result)
        else:
            raise ValueError

    # Second kind of input.
    if n == 3:
        if require_split[2].isalpha():
            format_result = get_roman_format(require_split[2])
            if format_result is None:
                raise ValueError
            else:
                arabic_result = transfer_generalised_roman_to_arabic(require_split[2], format_result)
                if arabic_result is None:
                    raise ValueError
        else:
            raise ValueError
        # output for third
        if arabic_result is not None and format_result is not None:
            print('Sure! It is', arabic_result, 'using', format_result)
        else:
            raise ValueError

except NotImplementedError:
    print("I don't get what you want, sorry mate!")
    sys.exit()
except ValueError:
    print("Hey, ask me something that's not impossible to do!")
    sys.exit()
