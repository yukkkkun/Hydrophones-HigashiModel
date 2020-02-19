
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

#インポートの仕方が毎回混乱する。要勉強
# from . import trendline
#これが正しいはずなのだが、これだとうまく行かない
import trendline
import dispgraphs
import getdfs

######################################################
# ハイドロフォン長さ
length_hp_m = 0.275
length_C_m = 0.5
pit_width = 0.2
Dv = 0.0125
Dh = 0.0125

W_IDEAL = np.array([0.15, 0.22, 0.29, 0.65, 0.91, 1.96, 3.01, 6.91, 10.81, 50])*0.001
TARGET_TOT = ['-2mm', '3-5mm', '5-6mm', '6-7mm', '7-8.5mm', '8.5-10mm', '10-12.5mm', '12.5-15mm', '15-20mm', '25-30mm', '30mm-']


event_marker = ['.', 'x', 'v', '1', 'D']

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

######################################################



def set_fig_title_labels(ax, title='Title', ylabel='y_name', xlabel='x_name'):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
        
def add_least_squares(x, y, ax):
    result = trendline.least_squares_method(x, y)
    trendline.plot_least_squares_result(result, x, ax)      
        
        
def time_series_graph(df, col_name='Load_Avg'):
 
    ax = plt.subplot(1, 1, 1)
    df[col_name].plot(ax=ax, ls='none', marker='.', markersize=1)

def time_series_graph_series(series):
 
    ax = plt.subplot(1, 1, 1)
    series.plot(ax=ax, ls='none', marker='.', markersize=1)


    
def scatter_graph(df, y_name, x_name, figsize=(5,5), title='Title',
                      ylabel='y_name', xlabel='x_name', alpha=1.0, least_squares=False, linear_regression=False):

    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    ax.scatter(df[x_name],df[y_name], alpha=alpha)
    set_fig_title_labels(ax, title=title, ylabel=ylabel, xlabel=xlabel)
        
    if least_squares:
            add_least_squares(df[x_name], df[y_name], ax)
            
    if linear_regression:
            trendline.add_linear_regression(df[x_name], df[y_name], color='r')
            
        
def scatter_graphs(df, list_y_names, list_x_names, figsize=(5,5), title='Title', 
                        ylabel=None, xlabel=None, least_squares=False, linear_regression=False,
                        alpha=1.0, overlap=False, axes=None):
    """
    前にあるグラフに重ねたいとき：overlap=True, かつ、axesを設定する。
    axesはoverlap=Falseのときにreturnで取得できる。
    """
    if ylabel == None:
        list_ylables = list_y_names
    
    else:
        list_ylables = [ylabel]*len(list_y_names)
    
    if xlabel == None:
        list_xlables = list_x_names
    
    else:
        list_xlables = [xlabel]*len(list_x_names)
  
    
    num_graphs = len(list_y_names)
    if overlap:
        for i in range(num_graphs):
            axes[i].scatter(df[list_x_names[i]], df[list_y_names[i]], alpha=alpha)
            set_fig_title_labels(axes[i], title=list(amplification_factor.keys())[i] + ' : ' + list(amplification_factor.values())[i], ylabel=list_ylables[i], xlabel=list_xlables[i])
                    
        pass
    else:
        plt.figure(figsize=figsize)
        axes = []
        for i in range(num_graphs):
            ax = plt.subplot(3, 4, i+1)
            ax.scatter(df[list_x_names[i]], df[list_y_names[i]], alpha=alpha)
            set_fig_title_labels(ax, title=list(amplification_factor.keys())[i] + ' : ' + list(amplification_factor.values())[i], ylabel=list_ylables[i], xlabel=list_xlables[i])
            axes.append(ax)

            if least_squares:
                add_least_squares(df[list_x_names[i]], df[list_y_names[i]], ax)
        #             result = least_squares_method(df[list_x_names[i]], df[list_y_names[i]])
        #             plot_least_squares_result(result, df[list_x_names[i]], ax)

            if linear_regression:
                    trendline.add_linear_regression(df[list_x_names[i]], df[list_y_names[i]], color='r')
            
    plt.tight_layout()

    if overlap:
        return
    else:
        return axes
    
def compare_graphs_time_series(df1, df2, start=None, end=None, figsize=(6,4)):
    if start== None:
        start = str(df1.axes[0][0])
        print(start)
    else:
        pass

    if end == None:
        end = str(df1.axes[0][-1])
        print(end)

    else:
        pass

    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    ax.plot(df1[start:end])
    ax.plot(df2[start:end])

def compare_graphs_scatter(df1, df2, start=None, end=None, figsize=(5,5), alpha=1.0, least_squares=None, linear_regression=None, weight_linear_regression=None):
    if start== None:
        start = str(df1.axes[0][0])
        print(start)
    else:
        pass

    if end == None:
        end = str(df1.axes[0][-1])
        print(end)

    else:
        pass

    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    ax.scatter(df1[start:end], df2[start:end], alpha=alpha)

    if least_squares:
        add_least_squares(df1[start:end], df2[start:end], ax)
            
    if linear_regression:
        trendline.add_linear_regression(df1[start:end], df2[start:end], color='r')
    
    if weight_linear_regression:
        trendline.add_weight_linear_regression(df1[start:end], df2[start:end], color='r')






def set_xy_lims_for_current_graph(xlim, ylim):
    plt.ylim(ylim)
    plt.xlim(xlim)



###################################################
###こっからはハイドロフォン研究用
#############################
def disp_qcal_monthly(qcalc, df_waterlevel):

    qcalc = pd.DataFrame(qcalc)
    qcalc.columns = ['qcalc']

    def extract_by_month(df, month):
        return df[df.index.month == month]

    def extract_by_year(df, year):
        return df[df.index.year == year]

    list_monthly_qcalc = []
    list_monthly_waterlevel = []
    for i in range(12):
        list_monthly_qcalc.append(extract_by_month(qcalc, i+1))
        list_monthly_waterlevel.append(pd.DataFrame(extract_by_month(df_waterlevel, i+1)))

    plt.figure(figsize=(5,5))
    colorlist = ["0.05", "0.1", "0.15", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    months = ["January", "February", "March",  "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    plt.figure(figsize=(12, 12))
    axis = []

    max_qcalc = int(qcalc.max())

    for i in range(12):
        ax = plt.subplot(4,4,i+1)

        temp_num = np.array(list(range(len(list_monthly_waterlevel[i]))))
    #     temp_num = np.random.rand(len(list_monthly_waterlevel[i]))
        ax.scatter(np.array(list_monthly_waterlevel[i]).flatten(), np.array(list_monthly_qcalc[i]).flatten(), c=temp_num, cmap='Blues', alpha=0.5)
        
        ax.set_xlim(-0.2,1.5)
        ax.set_ylim(-max_qcalc*0.1 ,max_qcalc + max_qcalc*0.1)

        ax.set_title(months[i])
        ax.set_ylabel('Bedload Dischaege(kg/min)')
        ax.set_xlabel('Q(m**3/min)')
        
        axis.append(ax)
        
        plt.tight_layout()


    ax = plt.subplot(4,4,13)
    for monthly_qcalc, monthly_waterlevel, color in zip(list_monthly_qcalc, list_monthly_waterlevel, colorlist):
        ax.scatter(monthly_waterlevel, monthly_qcalc, color=color)
        ax.set_xlim(-0.2,1.5)
        ax.set_ylim(-max_qcalc*0.1 ,max_qcalc + max_qcalc*0.1)

    axis.append(ax)

    # plt.show()
    

def disp_qcaleach_monthly(qcalc_each, df_waterlevel, tot_name='hp_Tot(10)'):
    qcalc = qcalc_each[tot_name]
    disp_qcal_monthly(qcalc, df_waterlevel)