import pandas as pd
import numpy as np
import os
import math

from matplotlib import pyplot as plt
from logging import getLogger, StreamHandler, DEBUG, INFO, WARNING
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#自作モジュール
import getdfs
import sklardietrich
import dataconversion

#############
#ログ設定
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False
#############
###########################################################

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


###########################################################
#calc gamma by sklardietrich
#sklardietrichからgammaを計算。あとで使用
def calc_gamma(list_h_s, bump_hp_m):
    list_gamma = []
    for h_s in list_h_s:
        #データがエラー値なら1
        if np.isnan(h_s):
            gamma = 1
            list_gamma.append(gamma)
            
        #ハイドロフォン高さ以下なら1
        elif h_s <= bump_hp_m:
            gamma = 1
            list_gamma.append(gamma)
            
        #ハイドロフォン高さ以上なら割り算
        else:
            gamma = bump_hp_m/h_s
            list_gamma.append(gamma)

    return list_gamma 

def calc_gammas(series_wl, D_IDEALcm_use):
    '''
    calc_gammaを使って，全ての粒径界，水深データに対してgammaを算出する
    sklardietrichは全てcm, gで計算しているためそれに合わせている．
    '''
    #gammaは無次元
    list2d_gamma = []
    bump_hp_m = 0.025 * 100 #cm
    width = 500#cm

    for wl in series_wl:
        #wl is in cm
        #径深Rを算出
        R = sklardietrich.calc_R(wl, width)
        #h_s is in cm
        list_h_s = [sklardietrich.calc_h_s(R=R, d=D_IDEALcm_use[i]) for i in range(len(D_IDEALcm_use))]

        #gammaを計算
        list_gamma = calc_gamma(list_h_s, bump_hp_m)
        list2d_gamma.append(list_gamma)

    return list2d_gamma



def optimize_calcCbeta(df_optimize_train, use_tots, waterlevel_train, qobs_train, W_IDEAL_use, D_IDEALcm_use, alpha, f_n, evaluate, repeat, error_less=0.5, error_more=1.5):
    evaluate = 'MAE'
    gammas = calc_gammas(series_wl=waterlevel_train, D_IDEALcm_use=D_IDEALcm_use)
    df_qcalc_train_nocorrected = (df_optimize_train.mul(W_IDEAL_use.reshape(1,len(W_IDEAL_use))) / np.array(gammas))


    def func(Const):
        '''
        ここに最小化するコスト関数を設定する。
        コスト関数をreturnで返す
        Const[:-1]: beta
        Const[-1] : C 
        '''
        df_w = df_qcalc_train_nocorrected.mul(Const[:-1])
        df_w_sum = df_w.sum(axis=1)
        df_qcalc = df_w_sum * (Const[-1]/(f_n*alpha))
        # cost = np.square(df_qcalc-df_qobs)
        # Cost = cost.sum()

        if evaluate == "MAE":
            cost = mean_absolute_error(qobs_train, df_qcalc)

        elif evaluate =='RMSE':
            cost = np.sqrt(mean_squared_error(qobs_train, df_qcalc))

        else:
            logger.exception("Evaluate func doesn't match with our choices ")

        return cost


    #     # βの制約条件
    # error_less = 0.5
    # error_more = 1.5

    #今回はこれ
    #チャンネル数増やすならそれ用に作らないといけない
    if len(use_tots)==6:
        
        cons = (
            {'type': 'ineq', 'fun': lambda Const: Const[0]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[0]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[1]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[1]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[2]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[2]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[3]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[3]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[4]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[4]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[5]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[5]+error_more},

            {'type': 'ineq', 'fun': lambda Const:  Const[6]},
            # {'type': 'ineq', 'fun': lambda Const:  1-Const[6]},

            {'type': 'ineq', 'fun': lambda Const: Const[1]*W_IDEAL[1] - Const[0]*W_IDEAL[0]},
            {'type': 'ineq', 'fun': lambda Const: Const[2]*W_IDEAL[2] - Const[1]*W_IDEAL[1]},
            {'type': 'ineq', 'fun': lambda Const: Const[3]*W_IDEAL[3] - Const[2]*W_IDEAL[2]},
            {'type': 'ineq', 'fun': lambda Const: Const[4]*W_IDEAL[4] - Const[3]*W_IDEAL[3]},
            {'type': 'ineq', 'fun': lambda Const: Const[5]*W_IDEAL[5] - Const[4]*W_IDEAL[4]},
            )

    #その他の時
    if len(use_tots)==7:
        
        cons = (
            {'type': 'ineq', 'fun': lambda Const: Const[0]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[0]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[1]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[1]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[2]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[2]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[3]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[3]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[4]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[4]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[5]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[5]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[6]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[6]+error_more},
            
            {'type': 'ineq', 'fun': lambda Const:  Const[7]},
            # {'type': 'ineq', 'fun': lambda Const:  1-Const[6]},

            {'type': 'ineq', 'fun': lambda Const: Const[1]*W_IDEAL[1] - Const[0]*W_IDEAL[0]},
            {'type': 'ineq', 'fun': lambda Const: Const[2]*W_IDEAL[2] - Const[1]*W_IDEAL[1]},
            {'type': 'ineq', 'fun': lambda Const: Const[3]*W_IDEAL[3] - Const[2]*W_IDEAL[2]},
            {'type': 'ineq', 'fun': lambda Const: Const[4]*W_IDEAL[4] - Const[3]*W_IDEAL[3]},
            {'type': 'ineq', 'fun': lambda Const: Const[5]*W_IDEAL[5] - Const[4]*W_IDEAL[4]},
            {'type': 'ineq', 'fun': lambda Const: Const[6]*W_IDEAL[6] - Const[5]*W_IDEAL[5]},
            )
    if len(use_tots)==8:
        cons = (
            {'type': 'ineq', 'fun': lambda Const: Const[0]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[0]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[1]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[1]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[2]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[2]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[3]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[3]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[4]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[4]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[5]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[5]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[6]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[6]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[7]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[7]+error_more},
            
            {'type': 'ineq', 'fun': lambda Const:  Const[8]},
            # {'type': 'ineq', 'fun': lambda Const:  1-Const[6]},

            {'type': 'ineq', 'fun': lambda Const: Const[1]*W_IDEAL[1] - Const[0]*W_IDEAL[0]},
            {'type': 'ineq', 'fun': lambda Const: Const[2]*W_IDEAL[2] - Const[1]*W_IDEAL[1]},
            {'type': 'ineq', 'fun': lambda Const: Const[3]*W_IDEAL[3] - Const[2]*W_IDEAL[2]},
            {'type': 'ineq', 'fun': lambda Const: Const[4]*W_IDEAL[4] - Const[3]*W_IDEAL[3]},
            {'type': 'ineq', 'fun': lambda Const: Const[5]*W_IDEAL[5] - Const[4]*W_IDEAL[4]},
            {'type': 'ineq', 'fun': lambda Const: Const[6]*W_IDEAL[6] - Const[5]*W_IDEAL[5]},
            {'type': 'ineq', 'fun': lambda Const: Const[7]*W_IDEAL[7] - Const[6]*W_IDEAL[6]},

            )
    if len(use_tots)==9:
        cons = (
            {'type': 'ineq', 'fun': lambda Const: Const[0]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[0]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[1]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[1]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[2]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[2]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[3]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[3]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[4]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[4]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[5]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[5]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[6]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[6]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[7]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[7]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[8]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[8]+error_more},
            
            {'type': 'ineq', 'fun': lambda Const:  Const[9]},
            # {'type': 'ineq', 'fun': lambda Const:  1-Const[6]},

            {'type': 'ineq', 'fun': lambda Const: Const[1]*W_IDEAL[1] - Const[0]*W_IDEAL[0]},
            {'type': 'ineq', 'fun': lambda Const: Const[2]*W_IDEAL[2] - Const[1]*W_IDEAL[1]},
            {'type': 'ineq', 'fun': lambda Const: Const[3]*W_IDEAL[3] - Const[2]*W_IDEAL[2]},
            {'type': 'ineq', 'fun': lambda Const: Const[4]*W_IDEAL[4] - Const[3]*W_IDEAL[3]},
            {'type': 'ineq', 'fun': lambda Const: Const[5]*W_IDEAL[5] - Const[4]*W_IDEAL[4]},
            {'type': 'ineq', 'fun': lambda Const: Const[6]*W_IDEAL[6] - Const[5]*W_IDEAL[5]},
            {'type': 'ineq', 'fun': lambda Const: Const[7]*W_IDEAL[7] - Const[6]*W_IDEAL[6]},
            {'type': 'ineq', 'fun': lambda Const: Const[8]*W_IDEAL[8] - Const[7]*W_IDEAL[7]},
            )  
    if len(use_tots)==10:
        cons = (
            {'type': 'ineq', 'fun': lambda Const: Const[0]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[0]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[1]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[1]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[2]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[2]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[3]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[3]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[4]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[4]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[5]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[5]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[6]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[6]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[7]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[7]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[8]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[8]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[9]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[9]+error_more},
            
            {'type': 'ineq', 'fun': lambda Const:  Const[10]},
            # {'type': 'ineq', 'fun': lambda Const:  1-Const[6]},

            {'type': 'ineq', 'fun': lambda Const: Const[1]*W_IDEAL[1] - Const[0]*W_IDEAL[0]},
            {'type': 'ineq', 'fun': lambda Const: Const[2]*W_IDEAL[2] - Const[1]*W_IDEAL[1]},
            {'type': 'ineq', 'fun': lambda Const: Const[3]*W_IDEAL[3] - Const[2]*W_IDEAL[2]},
            {'type': 'ineq', 'fun': lambda Const: Const[4]*W_IDEAL[4] - Const[3]*W_IDEAL[3]},
            {'type': 'ineq', 'fun': lambda Const: Const[5]*W_IDEAL[5] - Const[4]*W_IDEAL[4]},
            {'type': 'ineq', 'fun': lambda Const: Const[6]*W_IDEAL[6] - Const[5]*W_IDEAL[5]},
            {'type': 'ineq', 'fun': lambda Const: Const[7]*W_IDEAL[7] - Const[6]*W_IDEAL[6]},
            {'type': 'ineq', 'fun': lambda Const: Const[8]*W_IDEAL[8] - Const[7]*W_IDEAL[7]},
            {'type': 'ineq', 'fun': lambda Const: Const[9]*W_IDEAL[9] - Const[8]*W_IDEAL[8]},

            )    
    if len(use_tots)==5:
            cons = (
            {'type': 'ineq', 'fun': lambda Const: Const[0]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[0]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[1]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[1]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[2]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[2]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[3]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[3]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[4]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[4]+error_more},


            {'type': 'ineq', 'fun': lambda Const:  Const[5]},
            # {'type': 'ineq', 'fun': lambda Const:  1-Const[6]},

            {'type': 'ineq', 'fun': lambda Const: Const[1]*W_IDEAL[1] - Const[0]*W_IDEAL[0]},
            {'type': 'ineq', 'fun': lambda Const: Const[2]*W_IDEAL[2] - Const[1]*W_IDEAL[1]},
            {'type': 'ineq', 'fun': lambda Const: Const[3]*W_IDEAL[3] - Const[2]*W_IDEAL[2]},
            {'type': 'ineq', 'fun': lambda Const: Const[4]*W_IDEAL[4] - Const[3]*W_IDEAL[3]},
            )
    if len(use_tots)==4:
            cons = (
            {'type': 'ineq', 'fun': lambda Const: Const[0]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[0]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[1]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[1]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[2]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[2]+error_more},
            {'type': 'ineq', 'fun': lambda Const: Const[3]-error_less},
            {'type': 'ineq', 'fun': lambda Const: -Const[3]+error_more},

            {'type': 'ineq', 'fun': lambda Const:  Const[4]},
            # {'type': 'ineq', 'fun': lambda Const:  1-Const[6]},

            {'type': 'ineq', 'fun': lambda Const: Const[1]*W_IDEAL[1] - Const[0]*W_IDEAL[0]},
            {'type': 'ineq', 'fun': lambda Const: Const[2]*W_IDEAL[2] - Const[1]*W_IDEAL[1]},
            {'type': 'ineq', 'fun': lambda Const: Const[3]*W_IDEAL[3] - Const[2]*W_IDEAL[2]},
            )

    def opt_func(func, cons):
        Const0 = np.append(np.random.rand(len(use_tots)), 2)# 初期値を設定．値は適当
        result = minimize(func, x0=Const0, constraints=cons, method="COBYLA", options={'maxiter':10000})
        return result
    
    def repeat_opt_func(func, repeat, cons):
        '''
        repeatの数だけopt_funcを繰り返す．一番fungが最小の物がbest_result
        '''

        temp = 1000000000 
        results = []
        for _ in range(repeat):
            result = opt_func(func, cons)
            
            if result.fun < temp:
                results.append(result)
                temp = result.fun
    #             print('func : ', temp)
    #             print('x : ', result.x)
        best_result = results[-1]
        
        return best_result, results

    #こっからがoptimize_calcCbeta
    #大体おおよそ同じくらいの結果が毎回出るようになるはず．出ない場合はrepeatを1000とかに増やす
    best_result, results = repeat_opt_func(func, repeat=100, cons=cons)
    
    return best_result, results


def calc_q(df_optimize, waterlevel, list_beta, C, W_IDEAL_use, D_IDEALcm_use, f_n, alpha):
    
    
    #計算しやすいよう変換
    array_beta = np.array(list_beta)
    #gammaを算出

    gammas = calc_gammas(series_wl=waterlevel, D_IDEALcm_use=D_IDEALcm_use)
    df_qcalc_each = (df_optimize.mul(W_IDEAL_use.reshape(1,len(W_IDEAL_use))).mul(array_beta.reshape(1,len(W_IDEAL_use)))) / np.array(gammas) 
    df_qcalc_each_sum = df_qcalc_each.sum(axis=1)
    df_qcalc = df_qcalc_each_sum * (C/(f_n*alpha))
    
    return df_qcalc, df_qcalc_each
    

def qcalc_sum_period(qcalc, start=None, end=None):
    '''
    start,endは指定なければqcalc全体の総量になる
    '''
    if start == None:
        start = qcalc.index[0]
    if end == None:
        end = qcalc.index[-1]
    qcalc_sum = qcalc[start:end].sum()
    
    return qcalc_sum

def qcalc_each_sum_period(df_qcalc_each, start=None, end=None):
    '''
    start,endは指定なければqcalc全体の総量になる
    '''
    if start == None:
        start = df_qcalc_each.index[0]
    if end == None:
        end = df_qcalc_each.index[-1]
        
    qcalc_sum_each = df_qcalc_each[start:end].sum()
    
    return qcalc_sum_each

def print_qcalc_each(qcalc_all, df_qcalc_each_all, use_tots, start, end):
    qcalc_sum_all = qcalc_sum_period(qcalc_all, start=start, end=end)
    qcalc_sum_each_all = qcalc_each_sum_period(df_qcalc_each_all, start=start, end=end)
    print('流砂量({}~{})：'.format(start, end),qcalc_sum_all)
    TARGET_TOT_use = TARGET_TOT[-len(use_tots):]
    for i in range(len(TARGET_TOT_use)):
        print('粒径別{}'.format(TARGET_TOT_use[i]), qcalc_sum_each_all[i], 'kg')


def compare_qcalc_qobs(qcalc_all, qobs, start, end, zoomout=True):
    '''
    この期間は，PitTrueの期間でないと意味ない
    zoomout:Trueのとき，qcalcとqobsが両方ある区間のみの比較
    '''
    qobs_all_selected = qobs[start:end]
    qcalc_selected = qcalc_all[start:end]

    if zoomout:
        qcalc_selected = qcalc_selected[qobs_all_selected.index]
        
    qobs_sum = qobs_all_selected.sum()
    qcalc_sum = qcalc_selected.sum()
    print('qobs : ', qobs_sum)
    print('qcalc : ', qcalc_sum)
    
    plt.figure(figsize=(6,4))
    ax = plt.subplot(1,1,1) 

    ax.plot(qcalc_selected, label='calc.')
    ax.plot(qobs_all_selected, label='obs.')
    ax.set_ylabel('Bedload discharge\n[kg m$^{-1}$ min$^{-1}$]')
    ax.set_xlabel('')
    ax.legend()
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45)

    plt.show()

#########################
if __name__ == "__main__":
    pass



