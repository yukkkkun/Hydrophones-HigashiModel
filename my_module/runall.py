import pandas as pd
import numpy as np
import os
import math

from matplotlib import pyplot as plt


#自作モジュールと、モジュールアップデート用
#(importlib.reloadをしないとモジュールを書き換えても反映されないため)
import getdfs
import dataconversion
import trendline
import dispgraphs
import importlib
import graph_settings
import calcdischarge
import sklardietrich
import furui
import optimize
##################################################

#データ保存場所
data_path = os.getcwd()[:-len('notebook')] + 'data/'

# ピット直上ハイドロフォン長さ[m]
length_hp_m = 0.275
# 5本のハイドロフォンの長さ[m]
length_C_m = 0.5
#　ピットの開口幅[m]
pit_width = 0.2

# それぞれの粒径界の対象重量（左から1024倍，512倍，256倍，，，2倍）
W_IDEAL = np.array([0.15, 0.22, 0.29, 0.65, 0.91, 1.96, 3.01, 6.91, 10.81, 50])*0.001
#　それぞれの粒径界の対象粒径
TARGET_TOT = ['-2mm', '3-5mm', '5-6mm', '6-7mm', '7-8.5mm', '8.5-10mm', '10-12.5mm', '12.5-15mm', '15-20mm', '20-30mm', '30mm-']

event_marker = ['.', 'x', 'v', '1', 'D']

# ハイドロフォンのそれぞれのチャンネルの名前
suffix = ['_Tot(1)', '_Tot(2)', '_Tot(3)', '_Tot(4)', '_Tot(5)',
            '_Tot(6)', '_Tot(7)', '_Tot(8)', '_Tot(9)', '_Tot(10)']

# 直上中央ハイドロフォン
names_of_center = ['hp'+ s for s in suffix]
# 中央ハイドロフォン
names_of_C = ['C'+ s for s in suffix]
# 中央右ハイドロフォン
names_of_RC = ['RC'+ s for s in suffix]
# 中央左ハイドロフォン
names_of_LC = ['LC'+ s for s in suffix]
# 右ハイドロフォン
names_of_R = ['R'+ s for s in suffix]
# 左ハイドロフォン
names_of_L = ['L'+ s for s in suffix]

# 右鉛直ハイドロフォン
names_of_VR = ['VR'+ s for s in suffix]
# 左鉛直ハイドロフォン
names_of_VL = ['VL'+ s for s in suffix]


# Corrected直上中央ハイドロフォン
names_of_center_Corrected = ['Corrected_hp'+ s for s in suffix]
# Corrected中央ハイドロフォン
names_of_C_Corrected = ['Corrected_C'+ s for s in suffix]
# Corrected中央右ハイドロフォン
names_of_RC_Corrected = ['Corrected_RC'+ s for s in suffix]
# Corrected中央左ハイドロフォン
names_of_LC_Corrected = ['Corrected_LC'+ s for s in suffix]
# Corrected右ハイドロフォン
names_of_R_Corrected = ['Corrected_R'+ s for s in suffix]
# Corrected左ハイドロフォン
names_of_L_Corrected = ['Corrected_L'+ s for s in suffix]



# スロットナンバーと倍率を対応させる
amplification_factor = {'Tot(1)': '1024', 'Tot(2)': '512', 'Tot(3)': '256', 'Tot(4)': '128',
                        'Tot(5)': '64', 'Tot(6)': '32', 'Tot(7)': '16', 'Tot(8)': '8',
                        'Tot(9)': '4', 'Tot(10)': '2'}

#furui基本設定
x_furui = [1, 2, 5, 7, 9, 15, 31.5, 50] #furui粒径界の上限
x_tot = [2, 5, 6, 7, 8.5, 10, 12.5, 15, 20, 30, 50] #Tot粒径界の上限
g_furui = ['-1mm(g)', '1-2mm(g)', '2-5mm(g)', '5-7mm(g)', '7-9mm(g)', '9-15mm(g)', '19-31.5mm(g)', '31.5mm-(g)'] #furui粒径界(g)
percent_furui = ['-1mm(%)', '1-2mm(%)', '2-5mm(%)', '5-7mm(%)', '7-9mm(%)', '9-15mm(%)', '19-31.5mm(%)', '31.5mm-(%)'] #furui粒径界(%)

##################################################

def runall_on_jupyternotebook(num_tots, list_start_furui, list_end_furui, evaluate='MAE'):

    #import originaldata
    df_all = getdfs.getalloriginal(unit='m')
    #差分値をとる
    df_all_diff = dataconversion.make_dataframe_diff_of_slots(df=df_all)
    #df_allは使わないので削除
    del df_all

    #import traindate,testdate
    df_traindate = getdfs.gettraindate()
    df_testdate = getdfs.gettestdate()

    print(df_traindate)
    print('Date Of Train Data')

    print(df_testdate)
    print('Date Of Test Data')

    #train_dateとtest_dateをリストに格納する
    list_traindate = df_traindate.astype(str).values.tolist()
    list_testdate = df_testdate.astype(str).values.tolist()

    #trainとtestの期間のdfデータをlistに格納
    list_df_train = []
    for traindate in list_traindate:
        start_train = traindate[0]
        end_train = traindate[1]
        list_df_train.append(df_all_diff[start_train:end_train]) 
        
    list_df_test = []
    for testdate in list_testdate:
        start_test = testdate[0]
        end_test = testdate[1]
        list_df_test.append(df_all_diff[start_test:end_test])


    # 関数drop_untrusted_pit_dataを用いてtrainデータをtestデータそれぞれカットアウト
    for i, df_train in enumerate(list_df_train):
        list_df_train[i] = dataconversion.drop_untrusted_pit_data(df_train, min_pit=321.1, max_pit=1606.1)
    
    for i, df_test in enumerate(list_df_test):
        list_df_test[i] = dataconversion.drop_untrusted_pit_data(df_test, min_pit=321.1, max_pit=1606.1)

    #30分平均したデータをリストに格納
    list_df_train_mean = []
    list_df_test_mean = []
    for df_train in list_df_train:
        list_df_train_mean.append(dataconversion.mean_of_df(df_train, meantime=30))
    for df_test in list_df_test:
        list_df_test_mean.append(dataconversion.mean_of_df(df_test, meantime=30))

    #一旦list_df_train_meanとlist_df_test_meanを繋げる
    df_train_mean = pd.concat(list_df_train_mean)
    df_test_mean = pd.concat(list_df_test_mean)
    df_events_mean = pd.concat([df_train_mean, df_test_mean])



    use_tots = names_of_center[-num_tots:]
    coluses = use_tots + ['Load_Avg', 'Load_Avg_difference', 'WL_FMR_Avg']
    #使うtotの数に合わせてW_IDEALも選択
    W_IDEAL_use = W_IDEAL[-len(use_tots):]
    #データ整理
    df_train_mean = dataconversion.organize_dfs(df_train_mean)
    df_test_mean = dataconversion.organize_dfs(df_test_mean)


    #trainデータ
    df_optimize_train, waterlevel_train, qobs_train = dataconversion.splitdf_dfoptimize_wl_qobs(df_train_mean, use_tots=use_tots)
    #testデータ
    df_optimize_test, waterlevel_test, qobs_test = dataconversion.splitdf_dfoptimize_wl_qobs(df_test_mean, use_tots=use_tots)

    #ふるい結果display
    furui.disp_furuidata()

    alpha = furui.calc_alpha(max_dia=50, min_dia=x_tot[-num_tots-1])


    #f(Nr)は1としているため
    f_n = 1
    # alphaの値は上で決めてる
    # alpha=0.3
    #粒径を算出 
    D_IDEALm = 2*((W_IDEAL/2650)*(3/4)*(1/math.pi))**(1/3)
    D_IDEALm_use = 2*((W_IDEAL_use/2650)*(3/4)*(1/math.pi))**(1/3)
    D_IDEALcm_use = D_IDEALm_use*100

    best_result, results = optimize.optimize_calcCbeta(df_optimize_train, use_tots, waterlevel_train, qobs_train, W_IDEAL_use, D_IDEALcm_use, alpha, f_n, evaluate, repeat=100, error_less=0.5, error_more=1.5)

    list_beta = best_result.x[:-1]
    C = best_result.x[-1]
    print('beta:', list_beta)
    print('C:', C)


    #df_all_diffを使うデータだけに絞る
    df_all_diff_use = df_all_diff[coluses]
    #30分平均をとる
    df_all_diff_use = dataconversion.mean_of_df(df_all_diff_use, meantime=30)

    # 整理
    df_all_diff_use = dataconversion.organize_dfs(df_all_diff_use)

    #trainデータ
    df_optimize_all, waterleve_all, qobs_all = dataconversion.splitdf_dfoptimize_wl_qobs(df_all_diff_use, use_tots=use_tots)

    qcalc_all, df_qcalc_each_all = optimize.calc_q(df_optimize_all, waterleve_all, list_beta, C, W_IDEAL_use, D_IDEALcm_use, f_n=f_n, alpha=alpha)

    # trainのピット内増減を表示
    plt.figure(figsize=(15,5))
    qcalc_all.plot()
    plt.ylabel('Qcalc [kg m$^{-1}$]')
    plt.xlabel('')
    plt.show()

    #例えば，2017年の流砂量と粒径別流砂量
    optimize.print_qcalc_each(qcalc_all, df_qcalc_each_all, use_tots, start='2017-01-01 0:00', end='2017-12-31 23:59')

    furui.display_results_comparing_furui(qcalc_all, df_qcalc_each_all, qobs_test, list_start_furui, list_end_furui, alpha)

if __name__ == "__main__":
    num_tots=6
    print(x_tot[-num_tots-1])