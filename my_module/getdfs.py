import pandas as pd
import numpy as np
import os

from logging import getLogger, StreamHandler, DEBUG, INFO


#自作モジュール
import dispgraphs


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

length_hp_m = 0.3 # ピット直上ハイドロフォン長さ
length_C_m = 0.5 # 5本の真ん中ハイドロフォン長さ
pit_width = 0.2 #ピット流入口長さ

suffix = ['_Tot(1)', '_Tot(2)', '_Tot(3)', '_Tot(4)', '_Tot(5)',
            '_Tot(6)', '_Tot(7)', '_Tot(8)', '_Tot(9)', '_Tot(10)']

names_of_center = ['hp'+ s for s in suffix] # 直上中央ハイドロフォン
names_of_C = ['C'+ s for s in suffix] # 中央ハイドロフォン
names_of_RC = ['RC'+ s for s in suffix] # 中央右ハイドロフォン
names_of_LC = ['LC'+ s for s in suffix] # 中央左ハイドロフォン
names_of_R = ['R'+ s for s in suffix] # 右ハイドロフォン
names_of_L = ['L'+ s for s in suffix] # 左ハイドロフォン
names_of_VR = ['VR'+ s for s in suffix] # 右鉛直ハイドロフォン
names_of_VL = ['VL'+ s for s in suffix] # 左鉛直ハイドロフォン

# data_path = os.getcwd()[:-len('notebook')] + 'data\\'
data_path = os.path.abspath(__file__)[:-len('my_module/getdfs.py')] + 'data\\'


# #########################################
# ####オリジナルデータのインポート（1分毎）
# #########################################
def getalloriginal(unit='m'):
    '''
    オリジナルデータを読み込み，DataFrameで返す．
    unitは今はmでしか指定できません．
    '''
    df_all = pd.read_csv(data_path + 'ashiaraidani_all.csv',
                       index_col='TIMESTAMP', parse_dates=True)

    #水の単位体積重量を1000 kgf/m3，土砂の単位体積重量を2650 kgf/m3とし質量に変換した
    df_all['Load_Avg'] = df_all['Load_Avg']*1.65
    #ピット内の差分値をとった特徴量（流砂量）を増やす
    df_all['Load_Avg_difference'] = df_all['Load_Avg'].diff()

    #もし単位がmであれば，ハイドロフォンデータを長さで割る，ピットデータは開口幅で割る
    if unit=="m":
        #5本のハイドロフォンの名前取得
        colnames_of_RRCCLCL = names_of_R + names_of_RC + names_of_C + names_of_LC + names_of_L

        df_all[names_of_center] = df_all[names_of_center]/length_hp_m
        df_all[colnames_of_RRCCLCL] = df_all[colnames_of_RRCCLCL]/length_C_m
        df_all['Load_Avg_difference'] = df_all['Load_Avg_difference']/pit_width       
    else:
        logger.info('Units are not in the same unit. You have to match units among all data you will use by yourself')

        pass

    return df_all

def gettraindate():
    df_traindate = pd.read_csv(data_path + 'traindata.csv')
    df_traindate['starttime'] = pd.to_datetime(df_traindate['starttime'])
    df_traindate['endtime'] = pd.to_datetime(df_traindate['endtime'])
    return df_traindate

def gettestdate():
    df_testdate = pd.read_csv(data_path + 'testdata.csv')
    df_testdate['starttime'] = pd.to_datetime(df_testdate['starttime'])
    df_testdate['endtime'] = pd.to_datetime(df_testdate['endtime'])
    return df_testdate

def get_furui():
    df_furui= pd.read_csv(data_path + 'furui_results.csv', index_col='TIMESTAMP', parse_dates=True)
    return df_furui


###########################################################


if __name__ == "__main__":
    pass