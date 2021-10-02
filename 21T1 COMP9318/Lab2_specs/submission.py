## import modules here
import pandas as pd
import numpy as np
import helper


################### Question 1 ###################
def single_dataframe(p_v, df, res):
    df_list = df.values.tolist()[0]
    cols = df.columns.to_list()[:-1]
    res_df = pd.DataFrame()
    n = len(cols)
    # create truth table
    for col in cols:
        n = n - 1
        i = len(cols) - n - 1
        temp_str = ('1' * np.power(2, n) + '0' * np.power(2, n)) * np.power(2, i)
        res_df[col] = list(temp_str)
        res_df[col] = res_df[col].map(lambda x: df_list[i] if x == '1' else 'ALL')
    for row in res_df.values.tolist():
        row.append(str(df_list[-1]))
        res.append(p_v + row)
    return res


def buc_sr(p_v, df, res):
    # get current column names
    cols = df.columns.to_list()
    if df.shape[0] == 1 and len(cols) != 1:
        res = single_dataframe(p_v, df, res)
    else:
        if len(cols) == 1:  # The rest col is M
            p_v.append(str(sum(helper.project_data(df, 0))))
            res.append(p_v)
        else:  # recursion to get the deepest data
            temp_pre = p_v[:]
            # get values in the first column
            diff_values = list(set(helper.project_data(df, 0)))
            for val in diff_values:
                p_v = temp_pre[:]
                p_v.append(val)
                slice_val = helper.slice_data_dim0(df, val)
                res = buc_sr(p_v, slice_val, res)
            # integrate the column, add 'ALL'
            p_v = temp_pre[:]
            p_v.append('ALL')
            new_data = helper.remove_first_dim(df)
            # print(new_data)
            res = buc_sr(p_v, new_data, res)
    return res


def buc_rec_optimized(df):  # do not change the heading of the function
    pre_val = []  # save the previous val
    res_list = []  # save result for every line

    res_list = buc_sr(pre_val, df, res_list)

    # turn result to df
    res_df = pd.DataFrame(res_list)
    # update column names
    res_df.columns = df.columns.to_list()
    # print(res_df)
    return res_df
