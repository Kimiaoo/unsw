## import modules here 

################# Question 1 #################
from collections import defaultdict


def multinomial_nb(training_data, sms):  # do not change the heading of the function
    # a set with all words
    all_words = set()
    for eml in training_data:
        for word in eml[0]:
            all_words.add(word)

    # initialize two dictionaries with all words
    spam, ham = {}, {}
    for word in all_words:
        spam[word] = 1
        ham[word] = 1

    spam_num, ham_num = 0, 0  # the number of spam and ham respectively
    # get dictionaries with words appear times in spam and ham respectively
    for eml in training_data:
        if eml[1] == 'spam':
            spam_num += 1
            for word in eml[0]:
                spam[word] += eml[0][word]
        else:
            ham_num += 1
            for word in eml[0]:
                ham[word] += eml[0][word]

    # calculate prior probability
    total_ems = spam_num + ham_num
    prior_spam = spam_num / total_ems
    prior_ham = ham_num / total_ems

    # calculate total appear times of words in spam and ham respectively
    sum_spam_words = sum(spam.values())
    sum_ham_words = sum(ham.values())

    # calculate the probability of words given spam or ham
    pro_spam, pro_ham = {}, {}
    for word in spam:
        pro_spam[word] = spam[word] / sum_spam_words
        pro_ham[word] = ham[word] / sum_ham_words

    sms_dic = {}
    for word in sms:
        if word in all_words:
            if word not in sms_dic:
                sms_dic[word] = 1
            else:
                sms_dic[word] += 1

    class_spam, class_ham = 1, 1
    for word in sms_dic:
        class_spam *= pro_spam[word] ** sms_dic[word]
        class_ham *= pro_ham[word] ** sms_dic[word]

    return (class_spam * prior_spam) / (class_ham * prior_ham)
