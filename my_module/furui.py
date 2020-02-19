import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

#自作モジュール
import getdfs
import optimize
import graph_settings

#furui基本設定
x_furui = [1, 2, 5, 7, 9, 15, 31.5, 50] #furui粒径界の上限
x_tot = [2, 5, 6, 7, 8.5, 10, 12.5, 15, 20, 30, 50] #Tot粒径界の上限
g_furui = ['-1mm(g)', '1-2mm(g)', '2-5mm(g)', '5-7mm(g)', '7-9mm(g)', '9-15mm(g)', '19-31.5mm(g)', '31.5mm-(g)'] #furui粒径界(g)
percent_furui = ['-1mm(%)', '1-2mm(%)', '2-5mm(%)', '5-7mm(%)', '7-9mm(%)', '9-15mm(%)', '19-31.5mm(%)', '31.5mm-(%)'] #furui粒径界(%)
#ふるい結果の日数
df_furui = getdfs.get_furui()
num_furui_count = len(df_furui)
num_furui_date = int(num_furui_count/3)

#それぞれのふるい日の-45cmの％データをリストに格納
list_furui_percent45cm = []
for i in range(num_furui_date):
    list_furui_percent45cm.append(df_furui[3*i+1:3*i+2][percent_furui])

#list_furui_percent45cmの％を累積％に変換
list_furui_percent45cm_cumsum = []
for furui_percent45cm in list_furui_percent45cm:
    list_furui_percent45cm_cumsum.append(np.cumsum(furui_percent45cm, axis=1))

x_each = []
y_each = []
for i in range(num_furui_date):
    x_each.append(np.insert(x_furui, 0, 0))
    y_each.append(np.insert(np.array(list_furui_percent45cm_cumsum[i]).flatten(), 0, 0))


###すべてグラフ出力######
def calc_furui_mean_45cm():
    ###３つの平均を算出######
    furui_percents = [0]*len(percent_furui)
    for i in range(1, num_furui_count, 3):
        furui_percents = furui_percents + np.array(df_furui[i:i+1][percent_furui])
    furui_percent_cumsums = np.cumsum(furui_percents, axis=1).flatten()
    furui_percent_cumsum_mean =  furui_percent_cumsums/num_furui_date

    return furui_percent_cumsum_mean


def disp_furuidata():
    fig = plt.figure(figsize=(12,4))
    ax1 = plt.subplot(1,2,1) 

    for i in range(num_furui_count):
        furui_percent_cumsum = np.cumsum(df_furui[i:i+1][percent_furui], axis=1)
        ax1.scatter(x_furui, furui_percent_cumsum)
        ax1.set_xscale('log')
        ax1.set_ylim(0,105)
        ax1.set_xlim((0.1), 100)
    #########################

    #３つの平均を算出
    furui_percent_cumsum_mean = calc_furui_mean_45cm()
    ###真ん中のデータのみグラフ出力######
    ax2 = plt.subplot(1,2,2) 
    for i in range(1, num_furui_count, 3):
        furui_percent_cumsum = np.cumsum(df_furui[i:i+1][percent_furui], axis=1)
        ax2.scatter(x_furui, furui_percent_cumsum)
        ax2.set_xscale('log')
        ax2.set_ylim(0,105)
        ax2.set_xlim((0.1), 100)

    # ３つの平均をグラフ出力(赤)##
    ax2.scatter(x_furui, furui_percent_cumsum_mean, c="red")
    
    plt.show()
#########################

#直線近似
# y=mx+n
# ２点を通る方程式を返却
# (y=数値) or (x=数値) or (y=mx+n)
def makeLinearEquation(x1, y1, x2, y2):
    line = {}
    if y1 == y2:
        # y軸に平行な直線
        line["y"] = y1
    elif x1 == x2:
        # x軸に平行な直線
        line["x"] = x1
    else:
        # y = mx + n
        line["m"] = (y1 - y2) / (x1 - x2)
        line["n"] = y1 - (line["m"] * x1)
#     print(line)
    return line

def calc_y(line, dia):
#     print("y=mx+n : ",line)
    if 'y' in line:
        percent = line["y"]
    elif 'x' in line:
        print('Error, x is something wrong')
    else:
        percent = line["m"]*dia + line["n"]
#     print("pecent : ", percent)
    return percent
    

def calcPercent(dia, x, y):
    """
    粒径から通過百分率のパーセントを算出する
    各プロット毎に直線近似
    """
    if (dia >= x[0]) & (dia < x[1]):
        line = makeLinearEquation(x[0], y[0], x[1], y[1])
        percent = calc_y(line, dia)

    elif (dia >= x[1]) & (dia < x[2]):
        line = makeLinearEquation(x[1], y[1], x[2], y[2])
        percent = calc_y(line, dia)

    elif (dia >= x[2]) & (dia < x[3]):
        line = makeLinearEquation(x[2], y[2], x[3], y[3])
        percent = calc_y(line, dia)

    elif (dia >= x[3]) & (dia < x[4]):
        line = makeLinearEquation(x[3], y[3], x[4], y[4])
        percent = calc_y(line, dia)

    elif (dia >= x[4]) & (dia < x[5]):
        line = makeLinearEquation(x[4], y[4], x[5], y[5])
        percent = calc_y(line, dia)

    elif (dia >= x[5]) & (dia < x[6]):
        line = makeLinearEquation(x[5], y[5], x[6], y[6])
        percent = calc_y(line, dia)

    elif (dia >= x[6]) & (dia < x[7]):
        line = makeLinearEquation(x[6], y[6], x[7], y[7])
        percent = calc_y(line, dia)

    elif (dia >= x[7]) & (dia <= x[8]):
        line = makeLinearEquation(x[7], y[7], x[8], y[8])
        percent = calc_y(line, dia)
        
    else:
        print("Diameter is out of range")
        
    return percent

def calc_dia_rate(max_dia, min_dia, x, y):
    rate = calcPercent(max_dia, x, y) - calcPercent(min_dia, x, y)
    return rate/100

def calc_dia_from_percent(percent, x, y):
    try_dia = np.arange(0, 50, 0.1)
    
    dia_hold = []
    for dia in try_dia:
        dia_hold.append(calcPercent(dia, x, y))
    dia_hold = np.array(dia_hold)
    dia_hold = np.abs(dia_hold-percent)
    key = np.where(dia_hold == np.min(dia_hold))
    dia = try_dia[key]
    
    return dia


def calc_alpha(max_dia, min_dia): 
    furui_percent_cumsum_mean = calc_furui_mean_45cm()
    x_mean = np.insert(x_furui, 0, 0)
    y_mean = np.insert(furui_percent_cumsum_mean, 0, 0)
    alpha = calc_dia_rate(max_dia, min_dia, x_mean, y_mean)
    print("alpha:", alpha)
    return alpha



def display_sediment_distribution(qcalc_sum_each, percent_ini, furui_no):
    '''
    display_sediment_distributions用のグラフ描くコード
    '''
    percents = [100-percent_ini]
    percents.extend(list(np.array(qcalc_sum_each)/qcalc_sum_each.sum()*(percent_ini)))
    percents_cumsum = np.cumsum(percents)
    x_tot_use = x_tot[-len(qcalc_sum_each)-1:]

    plt.figure(figsize=(6,4))
    ax = plt.subplot(1,1,1) 

    ax.scatter(x_tot_use, percents_cumsum, label='calc.')
    ax.scatter(x_furui, list_furui_percent45cm_cumsum[furui_no], label='obs.')

    ax.set_xscale('log')
    ax.set_ylim(0,105)
    ax.set_xlim((0.1), 100)
    ax.set_ylabel('Percentage passing\n[%]')
    ax.set_xlabel('Diameter[mm]')

    ax.legend()
    
    return ax
    
#     plt.show()   


def display_sediment_distributions(df_qcalc_each_all, furui_no, alpha, list_start_furui, list_end_furui):
    '''
    furui_no:ふるい結果番号．順に０，１，２
    '''
    #alphaの値を%に直す
    percent_ini = alpha*100
    start = list_start_furui[furui_no]
    end = list_end_furui[furui_no]
    print('##'*10)    
    print('期間：', start, '~', end)
    #粒径別流砂量対象期間全量
    qcalc_sum_each = optimize.qcalc_each_sum_period(df_qcalc_each_all, start=start, end=end)
    
    #alphaを補正無し
    print('I set the initial alpha rate as {} here, because you set the alpha when optimizing'.format(percent_ini))
    display_sediment_distribution(qcalc_sum_each=qcalc_sum_each, percent_ini=percent_ini, furui_no=furui_no)
    plt.show()
    #alphaを補正したもの
    alpha_corrected = calc_dia_rate(50, 8.5, x_each[furui_no], y_each[furui_no])
    print('I set the initial alpha rate as {} here so that you can see how accurate the sediment distribution given by method2 even though I set the alpha as {} when optimizing'.format(alpha_corrected, alpha))
    percent_ini_corrected = alpha_corrected*100
    display_sediment_distribution(qcalc_sum_each=qcalc_sum_each, percent_ini=percent_ini_corrected, furui_no=furui_no)
    plt.show()
    #alphaを補正したもののアップ
    print('Zoomed out')
    percent_ini_corrected = alpha_corrected*100
    ax = display_sediment_distribution(qcalc_sum_each=qcalc_sum_each, percent_ini=percent_ini_corrected, furui_no=furui_no)
    ax.set_ylim((100-percent_ini_corrected)-10,105)
    plt.show()


def display_results_comparing_furui(qcalc_all, df_qcalc_each_all, qobs_test, list_start_furui, list_end_furui, alpha):
    for i in range(num_furui_date):
        display_sediment_distributions(df_qcalc_each_all, furui_no=i, alpha=alpha, list_start_furui=list_start_furui, list_end_furui=list_end_furui)
        optimize.compare_qcalc_qobs(qcalc_all, qobs=qobs_test, start=list_start_furui[i], end=list_end_furui[i])

if __name__ == "__main__":
    disp_furuidata()