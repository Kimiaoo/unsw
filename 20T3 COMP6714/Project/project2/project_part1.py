def WAND_Algo(query_terms, top_k, inverted_index):
    # in pseudo code
    # query_terms: q    top_k: k  inverted_index: I

    term_info = {}  # {t: [[doc_id], [weight], upper_bound, [(c,w)], cw_index]}
    candidates = {}  # current post [t:(c_t,w_t)]
    for t in range(0, len(query_terms)):
        doc_id = []  # the documents which the term appears
        weight = []  # all the weights of term in document
        cw = []  # save [(doc_id, weight)]
        # get all weights of each term in different doc in inverted_index
        for tup in inverted_index[query_terms[t]]:
            doc_id.append(tup[0])
            weight.append(tup[1])
            cw.append((tup[0], tup[1]))
        upper_bound = max(weight)  # the biggest weight of term
        cw_index = 0
        term_info[t] = [doc_id, weight, upper_bound, cw, cw_index]
        candidates[t] = (term_info[t][3][cw_index][0], term_info[t][3][cw_index][1])
    # print('term_info: ', term_info)
    theta = -1  # current threshold
    Ans = []  # k-set of (d,Sd) values
    Evaluation = 0
    check_out = 0
    # while candidates != {}:
    while -1 * len(query_terms) != check_out:
        for term in term_info:
            if term_info[term][4] == -1 and term in candidates.keys():
                del candidates[term]
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=False)
        candidates = {}
        for item in sorted_candidates:
            candidates[item[0]] = item[1]
        score_limit = 0
        pivot = 0
        # print('sorted_candidates: ', sorted_candidates)
        while pivot < len(query_terms) - 1 and pivot < len(sorted_candidates) - 1:
            tmp_s_lim = score_limit + term_info[sorted_candidates[pivot][0]][2]
            if tmp_s_lim > theta:
                break
            score_limit = tmp_s_lim
            pivot = pivot + 1
        caculate_eva = 0
        # print('pivot: ', pivot)

        for i in range(0, (pivot + 1)):
            caculate_eva = caculate_eva + term_info[sorted_candidates[i][0]][2]
            # print('ub: ', caculate_eva)
            # print('theta: ', theta)
        if caculate_eva > theta:
            Evaluation = Evaluation + 1

        if sorted_candidates[0][1][0] == sorted_candidates[pivot][1][0]:
            s = 0  # score document c_pivot
            t = 0
            # Evaluation = Evaluation + 1
            while t < len(query_terms) and t < len(sorted_candidates) \
                    and sorted_candidates[t][1][0] == sorted_candidates[pivot][1][0]:
                term = sorted_candidates[t][0]
                cw_index = term_info[term][4]
                s = s + term_info[term][3][cw_index][1]
                term_info[term][4] = cw_index + 1
                if term_info[term][4] >= len(term_info[term][3]):
                    term_info[term][4] = -1
                    check_out = check_out + term_info[term][4]
                else:
                    candidates[term] = (term_info[term][3][term_info[term][4]][0],
                                        term_info[term][3][term_info[term][4]][1])
                t = t + 1
            if s > theta:
                Ans.append((s, sorted_candidates[pivot][1][0]))
                if len(Ans) > top_k:
                    smallest_index = 0
                    for i in range(1, len(Ans)):
                        if Ans[i][0] < Ans[smallest_index][0]:
                            smallest_index = i
                        elif Ans[i][0] == Ans[smallest_index][0]:
                            if Ans[i][1] > Ans[smallest_index][1]:
                                smallest_index = i
                    Ans.pop(smallest_index)
                    min_score_list = []
                    for item in Ans:
                        min_score_list.append(item[0])
                    theta = min(min_score_list)
                # print('Ans: ', Ans)
                # print('theta: ', theta)
        else:
            if caculate_eva > theta:
                Evaluation = Evaluation - 1
            for t in range(0, pivot):
                term = sorted_candidates[t][0]
                cw_index = term_info[term][4]
                if sorted_candidates[t][1][0] < sorted_candidates[pivot][1][0]:
                    term_info[term][4] = cw_index + 1
                    if term_info[term][4] >= len(term_info[term][3]):
                        term_info[term][4] = -1
                        check_out = check_out + term_info[term][4]
                    else:
                        candidates[term] = (term_info[term][3][term_info[term][4]][0],
                                            term_info[term][3][term_info[term][4]][1])
        Ans = sorted(Ans, key=lambda can: (-can[0], can[1]))
    return Ans, Evaluation
