import pandas as pd


# #########################################
# ####隣合うパルス数の差分値とる
# #########################################
def make_dataframe_diff_of_slots(df):
    """
    （スロット１のデータ）−（スロット２のデータ）、（スロット２のデータ）－（スロット３のデータ）
    、、、と，隣り合うパルス数を引き算していく．最後だけ，スロット１０のデータのまま．
    columnの名前は，最初のスロットの名前のままにしている
    （つまり，hp_Tot(1)-hp_Tot(2)のデータのcolnameを，形式的にhp_Tot(1）のままにしている）
    """
    # name[1]にはスロット１のデータ、name[2]には、スロット２のデータ、name[3]に、、、が入っている
    names = [0]*10
    for j in range(1, 11):
        names[j-1] = [i for i in df.columns if '({})'.format(j) in i]

    # （スロット１のデータ）−（スロット２のデータ）、（スロット２のデータ）－（スロット３のデータ）、、、としていき、
    # 粒径ごとのデータに分類する
    df_dia = [0]*10
    for i in range(1, 10):
        df_dia[i-1] = df[names[i-1]] - df[names[i]].values
#         df_dia[i-1].name = str(df_dia[i-1].name) + '-({})'.format(i)
    df_dia[9] = df[names[-1]]

    # 全てのデータを繋げる
    df_dia_all = pd.DataFrame()
    for i in range(1, 10):
        df_dia_all = pd.concat([df_dia_all, df_dia[i-1]], axis=1)
    df_dia_all = pd.concat([df_dia_all, df_dia[-1]], axis=1)
    
    ###########################################################
    #ハイドロフォンデータ以外で追加したいものがあればここに追加
    df_dia_all['Load_Avg'] = df['Load_Avg']
    df_dia_all['Load_Avg_difference'] = df['Load_Avg_difference']
    df_dia_all['WL_FMR_Avg'] = df['WL_FMR_Avg']
    df_dia_all['Velocity(m/s)'] = df['vel_P_Tot']
    ###########################################################
    
    return df_dia_all

def drop_untrusted_pit_data(df, min_pit, max_pit):
    """
    Load_Avg(ピット内)がmin_pit[kg]以上max_pit[kg]以下のデータのみを抽出
    注意：単位はkg
    """
    
    df_dropped = df[(df['Load_Avg'] > min_pit)&(df['Load_Avg'] < max_pit)]
    return df_dropped

def mean_of_df(df, meantime=30):
    """
    dfを30分間隔平均にする。
    'Load_Avg'は平均間隔にするとおかしくなるのでしない
    他にも平均にするとおかしいデータがあるはず？
    少なくともハイドロフォンデータは大丈夫だからそのまま行きます
    """
    sum_interval = meantime
    df_mean = df.resample('{}T'.format(sum_interval)).sum() / sum_interval
    df_mean['Load_Avg'] = df['Load_Avg']
#     df_mean['Load_Avg_difference'] = df['Load_Avg_difference'].resample(
#         '{}T'.format(sum_interval)).sum() / sum_interval
    
    return df_mean

def pit_positive(df):
    '''
    df has to have Load_Avg_difference column
    '''
    return df[df['Load_Avg_difference']>=0]

def waterlevel_clean(df, min_wl=2, max_wl=30):
    '''
    df has to have WL_FMR_Avg column.
    '''
    df = df[df['WL_FMR_Avg']>min_wl]
    df = df[df['WL_FMR_Avg']<max_wl]
    return df

def organize_dfs(df):
    df = pit_positive(df)
    df = waterlevel_clean(df)
    #NANが一つでもあれば削除
    df = df.dropna(how='all')
    return df

def splitdf_dfoptimize_wl_qobs(df, use_tots):
    '''
    df has to have only use_tots, WL_FMR_Avg, Load_Avg_difference columns
    '''
    df_optimize = df[use_tots]
    waterlevel = df['WL_FMR_Avg']
    qobs = df['Load_Avg_difference']
    return df_optimize, waterlevel, qobs